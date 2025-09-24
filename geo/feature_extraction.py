import argparse
import os
import sys
import pickle
from tqdm import tqdm
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from load_model import load_trained_model
Image.MAX_IMAGE_PIXELS = None


class PetroXLSLoader(Dataset):
    def __init__(self,filepath):
        assert os.path.exists(filepath)
        df = pd.read_excel(filepath)
        self.text = df['text'].tolist()
        self.cd_guid = df['cd_guid'].tolist()

    def __len__(self):
        return len(self.cd_guid)

    def __getitem__(self, index):
        return {'text': self.text[index], 'cd_guid': self.cd_guid[index]}

    def get_loader(self, batch_size):
        indices = np.arange(len(self.cd_guid))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)

def preprocess(mask_image: Image, image_size: int = 512, patch_size: int = 16,) -> torch.Tensor:
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    im =  TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
    return TF.normalize(im, mean=IMAGE_MEAN, std=IMAGE_STD).unsqueeze(0)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download and extract features from petro dataset')
    parser.add_argument('--root', type=str, required=True, help='folder containing petro dataset images')
    parser.add_argument('--output', type=str, required=True, help='path to save the extracted features')
    parser.add_argument('--data', type=str, required=True, help='path to xlsx file with text and ids')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--dim', default=512, type=int, help='image dimension')
    parser.add_argument('--config', type=str, default='dinov3/configs/train/dinov3_vitl16_geo.yaml', help='dino cfg file')
    parser.add_argument('--checkpoint', type=str, default='/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin', help='checkpoint file')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    # model init
    model = load_trained_model(args.config, args.checkpoint)
    model.cuda()

    dataset_petro = PetroXLSLoader(args.data)
    
    data = {'captions': dataset_petro.text,
            'image_id': dataset_petro.cd_guid,
            'image_embeddings': [],}

    loader = dataset_petro.get_loader(16)
    # extraction loop
    for batch in tqdm(loader):
        imgs = []
        for id in batch['cd_guid']:
            imgs.append(preprocess(Image.open(f'{args.root}/{id}.png').convert('RGB')))
        
        with torch.no_grad():
            vis_embed = model.forward(torch.cat(imgs).cuda(), is_training=False, return_grid=False)
            print(vis_embed.shape)
            data['image_embeddings'] += vis_embed.detach().cpu()
            # to avoid compatibility issues
            logging.debug(f'image embeddings shape: {vis_embed.shape}')

    logging.debug(f'caption sample {data["captions"][0]}')
    data['image_embeddings'] = torch.stack(data['image_embeddings'])

    print(data['image_embeddings'].shape)
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)



