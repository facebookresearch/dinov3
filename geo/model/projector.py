from torch import nn
import math
import torch

'''
code from transformers package https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
'''

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    

class Qwen2_5_VLPatchMerger(nn.Module):
    """Merges every spatial_merge_size x spatial_merge_size window of patch tokens (b x n x context_dim, row-major
    square grid as returned by the encoders) into one token of the decoder dimension: b x n / s^2 x dim.
    Qwen2.5-VL's vision tower orders the patches so that consecutive tokens form a window; DINO returns them row by
    row, so the windows are gathered here from the grid."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        s = self.spatial_merge_size
        side = math.isqrt(n)
        if side * side != n or side % s:
            raise ValueError(f'merger needs a square grid of patches with a side divisible by {s}, got {n} patches '
                             '(images are center-cropped to a square by default)')
        x = self.ln_q(x)
        # (b, rows, s, cols, s, d) -> one row per window, its s x s patches concatenated
        x = x.view(b, side // s, s, side // s, s, d).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, (side // s) ** 2, self.hidden_size)
        return self.mlp(x)
    

class Mapper(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        self.k = k
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * k * 2),
            nn.GELU(),
            nn.Linear(output_dim * k * 2, output_dim * k),
        )

    def forward(self, x):
        _x = self.mlp(x)
        return _x.view(x.shape[0], self.k, self.output_dim)
        

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class QueryExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.queries = nn.Parameter(torch.randn(k, input_dim))
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        

    def forward(self, x, return_weights=False):
        b = x.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        attn_scores, attn_weights = self.attention(query=q, key=x, value=x)
        x = self.norm(attn_scores)
        x = self.proj(x)
    
        if return_weights:
            return x, attn_weights
        
        return x

class QFormerBlock(nn.Module):
    """One Q-Former layer: self-attention among the queries, cross-attention from the queries to the visual tokens and
    an MLP, each pre-norm with a residual connection."""

    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, queries, visual, return_weights=False):
        h = self.norm_self(queries)
        queries = queries + self.self_attention(h, h, h, need_weights=False)[0]
        h = self.norm_cross(queries)
        attended, weights = self.cross_attention(h, visual, visual, need_weights=return_weights)
        queries = queries + attended
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries, weights


class QFormer(nn.Module):
    """Q-Former-like bridge (BLIP-2) without the text branch: k learned queries go through a stack of QFormerBlocks that
    read the visual tokens (patch embeddings, b x n x input_dim) by cross-attention; the k outputs are projected to the
    decoder dimension and become the k visual tokens of the prompt, whatever the number of patches (image size).
    Trained only with the captioning loss of the decoder."""

    def __init__(self, input_dim, output_dim, k, layers=2, dim=768, heads=8, dropout=0.0, condition_dim=None):
        super().__init__()
        self.visual_norm = nn.LayerNorm(input_dim)
        self.visual_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.blocks = nn.ModuleList(QFormerBlock(dim, heads, dropout=dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)
        # condition_dim: a per-image vector (e.g. the scale embedding) added to every query; zero-initialised, so the
        # Q-Former starts as without it
        self.conditioned = condition_dim is not None
        if self.conditioned:
            self.condition = nn.Linear(condition_dim, dim)
            nn.init.zeros_(self.condition.weight)
            nn.init.zeros_(self.condition.bias)

    def forward(self, x, return_weights=False, condition=None):
        visual = self.visual_proj(self.visual_norm(x))
        queries = self.queries.expand(x.shape[0], -1, -1)
        if self.conditioned and condition is not None:
            queries = queries + self.condition(condition.to(queries.dtype))[:, None]
        weights = []
        for block in self.blocks:
            queries, w = block(queries, visual, return_weights)
            weights.append(w)
        out = self.proj(self.norm(queries))
        # cross-attention maps per layer (b x k x n), e.g. to see where each query looks
        return (out, weights) if return_weights else out


class ScaleEmbedding(nn.Module):
    """Embedding of the physical scale of an image, given as log(mm per encoder input pixel): sinusoidal features at
    several frequencies (so both small and large scale differences are resolved) and an MLP to output_dim."""

    def __init__(self, output_dim, frequencies=8, hidden=256):
        super().__init__()
        self.register_buffer('frequencies', 2.0 ** torch.arange(-3, frequencies - 3, dtype=torch.float32),
                             persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * frequencies, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, log_scale):
        x = log_scale.float()[:, None]
        angles = x * self.frequencies
        features = torch.cat([x, angles.sin(), angles.cos()], dim=-1)
        return self.mlp(features.to(self.mlp[0].weight.dtype))


def multimodal_factory(name, input_dim, output_dim, condition_dim=None):
    # condition_dim: size of a per-image condition (the scale embedding) the projection can use (qformer only)
    name = name.split(':')
    if name[0] == 'qformer':
        # qformer:<queries>[:<layers>[:<width>]], e.g. qformer:32 or qformer:32:4:1024
        k = int(name[1])
        layers = int(name[2]) if len(name) > 2 else 2
        dim = int(name[3]) if len(name) > 3 else 768
        return QFormer(input_dim, output_dim, k, layers=layers, dim=dim, condition_dim=condition_dim)
    if name[0] == 'mapper':
        return Mapper(input_dim, output_dim, int(name[1]))
    if name[0] == 'projector':
        return Projector(input_dim, output_dim)
    if name[0] == 'merger':
        return Qwen2_5_VLPatchMerger(output_dim, input_dim, int(name[1]))
    if name[0] == 'attention':
        return QueryExtractor(input_dim, output_dim, int(name[1]))
    raise ValueError(f'unknown multimodal projection {":".join(name)}')


if __name__ == '__main__':
    x = torch.rand((32, 128, 768))
    ext = QueryExtractor(768, 1024, 16)
      
    print(ext(x).shape)

    map = Mapper(768, 1024, 16)
    x = torch.randn((32, 768))
    print(map(x).shape)