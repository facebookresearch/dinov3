import os, sys
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
import torch
from omegaconf import DictConfig, OmegaConf
from dinov3.models.vision_transformer import DinoVisionTransformer

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


if __name__ == '__main__':
    weight_path = '/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin'
    cfg_path = 'dinov3/configs/train/dinov3_vitl16_geo.yaml'
    model = load_model(cfg_path, weight_path)