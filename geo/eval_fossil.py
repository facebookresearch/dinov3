from model.fossilVL import FossilVL
from omegaconf import OmegaConf
import os
from dataset import ConversationDataset
import torch 


def load_fossil(path):
    conf = OmegaConf.load(os.path.join(path, 'config.yaml'))
    model = FossilVL(conf)
    ckpt = torch.load(os.path.join(path, 'checkpoint.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)

    return model


if __name__ == '__main__':
    images = ['B166354B2A50549FE04400144F24C75C', 'E45D13E63E7908AEE044002128300A66', 
              'E1080C2441974FE8E0533A821E0A1C6C', 'AD40F23BD1274A48E05354EB1D0AB94C']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_fossil('/nethome/recpinfo/users/fibz/data/checkpoints/fossil_vl').to(device)
    
    for im in images:
        image = f"/nethome/atena_projetos/fibz/images/{im}.png"
        # print(image)
        prompt = 'Descreva as características da rocha de acordo com a imagem microscópica.'
        output = model.generate([image], prompt, device)
        print('output')
        print(output)
