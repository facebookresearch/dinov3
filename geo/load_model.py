import os, sys
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
import torch
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.layers.dino_head import DINOHead
from omegaconf import OmegaConf


def load_trained_model(cfg_path, weight_path):
    conf = OmegaConf.load(cfg_path)
    model = DinoVisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layerscale_init=1e-05, 
        **conf.student,
    )
    state = torch.load(weight_path, weights_only=False)
    new_state = {}
    for k, v in state['model'].items():
        if 'dino_head' in k or 'ibot_head' in k:
            pass
        elif 'student' in k:
            name = k.replace('student.backbone.', '')
            new_state[name] = v
            
    model.load_state_dict(new_state)
    return model

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



if __name__ == '__main__':
    weight_path = '/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin'
    cfg_path = 'dinov3/configs/train/dinov3_vitl16_geo.yaml'
    model = load_trained_model(cfg_path, weight_path)
    dino = load_dino_head(cfg_path, weight_path)
    # print(model)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    from PIL import Image
    path = '/nethome/atena_projetos/fibz/data/Dataset/images/9A5EB7D9EFEC0ED9E0534EEB1D0A1100.png'
    img = Image.open(path).convert('RGB')

    def resize_transform(
        mask_image: Image,
        image_size: int = 512,
        patch_size: int = 16,
    ) -> torch.Tensor:
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
    
    # print(dir(model))
    input = resize_transform(img)
    input = TF.normalize(input, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    with torch.no_grad(): 
        grid = False
        outputs = model.forward(torch.cat([input.unsqueeze(0), input.unsqueeze(0)]), is_training=False, return_grid=grid)
        if grid:
            print(outputs['cls'].shape)
        else:
            print(outputs.shape)
            prototipes_logits = dino(outputs)
            print(prototipes_logits.shape)
            print(prototipes_logits.argmax())