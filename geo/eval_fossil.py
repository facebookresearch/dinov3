from fossilVL import FossilVL
from omegaconf import OmegaConf
import os
from dataset import ConversationDataset
import torch 


def load_fossil(path):
    conf = OmegaConf.load(os.path.join(path, 'config.yaml'))
    model = FossilVL(conf.decoder.name, conf.encoder.weight_path, conf.encoder.config_path)
    ckpt = torch.load(os.path.join(path, 'checkpoint.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)

    return model


if __name__ == '__main__':
    images = ['E45D13E63E7908AEE044002128300A66', 'B166354B2A50549FE04400144F24C75C', 
              'E1080C2441974FE8E0533A821E0A1C6C', 'AD40F23BD1274A48E05354EB1D0AB94C']
    model = load_fossil('/nethome/recpinfo/users/fibz/data/checkpoints/fossil_vl')
        
    for im in images:
        image = f"/nethome/atena_projetos/fibz/images/{im}.png"
        output = model.generate(image)
        print('output')
        print(output)
