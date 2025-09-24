import os, sys
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
import torch
from omegaconf import OmegaConf
from dinov3.models.vision_transformer import DinoVisionTransformer
from load_model import load_dino_head, load_trained_model
import torchvision.transforms.functional as TF
import json
from tqdm import tqdm
from PIL import Image
from dinov3_visualization import resize_transform
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None



if __name__ == '__main__':
    weight_path = '/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin'
    cfg_path = 'dinov3/configs/train/dinov3_vitl16_geo.yaml'
    
    dataset = json.load(open('/nethome/atena_projetos/fibz/data/Dataset/caption_dataset_train.json', 'r'))
    model = load_trained_model(cfg_path, weight_path)
    dino_head = load_dino_head(cfg_path, weight_path)
    counts = [0]*dino_head.last_layer.out_features
    
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        guid = sample['cd_guid']
        image = Image.open(f'/nethome/atena_projetos/fibz/data/Dataset/images/{guid}.png').convert('RGB')
        image = resize_transform(image)
        embeddings = model.forward(image.unsqueeze(0), is_training=False)
        output = dino_head(embeddings)
        counts[output.argmax()] += 1
        # if i > 10:
        #     break

    plt.plot(range(len(counts)), counts)
    plt.xlabel('prototipes')
    plt.ylabel('counts')
    plt.savefig('PROTOTIPES.png')
    
    json.dump( counts, open('counts.json', 'w'))        