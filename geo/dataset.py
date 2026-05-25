from PIL import Image
import json
import torch
import os
import numpy as np
from torch.utils.data import Dataset
Image.MAX_IMAGE_PIXELS = None


class ConversationDataset(Dataset):
    def __init__(self, root, annotation):
            data = json.load(open(annotation, 'r'))
            self.root = root
            self.image = []
            self.conversation = []
            for sample in data:
                if 'cd_guid' in sample:
                    self.image.append(os.path.join(self.root, '{}.png'.format(sample['cd_guid'])))
                
                elif 'image_name' in sample:
                    self.image.append(os.path.join(self.root, sample['image_name'].replace('\\', '/')))
                else:
                    raise ValueError('there is no image in the dataset')

                self.conversation.append(json.dumps(sample['conversation']))

    def __getitem__(self, index):
        return {
            'image': self.image[index],
            'conversation': self.conversation[index],
        
        }
        
    def __len__(self):
        return len(self.image)
    
    def colate__fn(self, batch):
        payload = {'image': [], 'conversation': []}
        for sample in batch:
            payload['image'].append(sample['image'])
            payload['conversation'].append(json.loads(sample['conversation']))
            # print(sample['conversation'])
        return payload
            

    def get_loader(self, batch_size:int, shuffle:bool):
        '''
        get torch dataloader
        :param batch_size: batch size for the dataloader
        :return: dataloader
        '''
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.colate__fn)
    



if __name__ == "__main__":
    dataset = ConversationDataset('/nethome/atena_projetos/fibz/images', '/nethome/atena_projetos/fibz/data/Dataset/simple_conversation/conv_test.json')
    loader = dataset.get_loader(2, True)
    # print(dataset[0])
    for batch in loader:
        print(batch)
        break
