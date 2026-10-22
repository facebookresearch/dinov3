import os, sys
sys.path.append(os.path.normpath(os.path.join(__file__, '../../../')))
import torch
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.layers.dino_head import DINOHead
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms.functional as TF


def create_model(cfg_path):
    conf = OmegaConf.load(cfg_path)
    return DinoVisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layerscale_init=1e-05, 
        **conf.student,
    )


def create_and_load_model(cfg_path, weight_path, branch='teacher'):
    """ViT-L/16 of the SSL training config with the backbone of the checkpoint: the teacher (EMA of the student, the
    one DINOv3 evaluates and releases) by default, or the student."""
    conf = OmegaConf.load(cfg_path)
    model = DinoVisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layerscale_init=1e-05,
        **conf.student,
    )
    return load_backbone_weights(model, weight_path, branch)


def load_backbone_weights(backbone, weight_path, branch='teacher'):
    """Loads the backbone of a DINOv3 SSL checkpoint of this project (state['model'] with '<branch>.backbone.' keys,
    branch 'teacher' or 'student') into backbone, a ViT-L/16 of the same architecture, e.g. DINO-TXT's vision backbone.
    rope_embed.periods is computed from the RoPE settings, not learned, so the backbone's own copy is kept (the SSL
    checkpoint stores it in bf16)."""
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    prefix = f'{branch}.backbone.'
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix) and not k.endswith('rope_embed.periods')}
    if not weights:
        prefixes = sorted({'.'.join(k.split('.')[:2]) for k in state})[:10]
        raise RuntimeError(f'no {prefix}* weights in {weight_path}; it has {prefixes}')
    missing, unexpected = backbone.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.endswith('rope_embed.periods')]
    if missing or unexpected:
        raise RuntimeError(f'{weight_path} ({branch}) does not match the backbone: missing {missing[:5]}, '
                           f'unexpected {unexpected[:5]}')
    print(f'loaded the {branch} backbone ({len(weights)} tensors) from {weight_path}')
    return backbone


def load_dino_head(cfg_path, weight_path):
    conf = OmegaConf.load(cfg_path)
    dino = DINOHead(
        in_dim=1024,
        out_dim=conf.dino.head_n_prototypes,
        nlayers=conf.dino.head_nlayers,
        hidden_dim=conf.dino.head_hidden_dim,
        bottleneck_dim=conf.dino.head_bottleneck_dim,
        mlp_bias=True,
    )
    state = torch.load(weight_path, weights_only=False)
    new_state = {}
    for k, v in state['model'].items():
        if 'teacher' in k:
            if 'dino_head' in k:
                name = k.replace('teacher.dino_head.', '')
                new_state[name] = v
            
    dino.load_state_dict(new_state)
    # print(dino)
    return dino


def resize_transform(mask_image: Image, image_size: int = 512, patch_size: int = 16) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        input = TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
        input = TF.normalize(input, mean=mean, std=std)
        
        return input 
    

def preprocess(images: list, image_size: int)-> torch.Tensor:
    tensors = []
    for image in images:
        im = Image.open(image).convert('RGB')
        tensors.append(resize_transform(im, image_size=image_size)[0])
    return tensors


if __name__ == '__main__':
    weight_path = '/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin'
    cfg_path = 'dinov3/configs/train/dinov3_vitl16_geo.yaml'
    model = create_and_load_model(cfg_path, weight_path)
    dino = load_dino_head(cfg_path, weight_path)
    # print(model)
    
    path = '/nethome/atena_projetos/fibz/data/Dataset/images/9A5EB7D9EFEC0ED9E0534EEB1D0A1100.png'
    img = Image.open(path).convert('RGB')
    input = resize_transform(img)
    
    # print(dir(model))
   
    with torch.no_grad(): 
        outputs = model.forward(input.unsqueeze(0), is_training=False, return_grid=True)
        print(outputs)