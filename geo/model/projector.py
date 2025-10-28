from torch import nn
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
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)
        return x
    
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


def multimodal_factory(name, input_dim, output_dim):
    name = name.split(':')
    if name[0] == 'mapper':
        return Mapper(input_dim, output_dim, int(name[1]))
    if name[0] == 'projector':
        return Projector(input_dim, output_dim)
    if name[0] == 'merger':
        return Qwen2_5_VLPatchMerger(output_dim, input_dim, int(name[1]))

if __name__ == '__main__':
    mapper = Mapper(768, 1024*5, 1024, 10)
    x = torch.rand((8, 768))
    x = mapper(x)
    print(x.shape)

    proj = Projector(768, 2048, 1024)
    x = torch.rand((8, 64, 768))
    x = proj(x)
    print(x.shape)

