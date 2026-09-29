from PIL import Image
import json
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random
import math
from torch.utils.data import Sampler

Image.MAX_IMAGE_PIXELS = None

def conversation_collate(batch):
    payload = {'image': [], 'conversation': []}
    sizes = []
    for sample in batch:
        payload['image'].append(sample['image'])
        payload['conversation'].append(json.loads(sample['conversation']))
        if 'size' in sample.keys():
            sizes.append(sample['size'])

    if len(sizes) > 0:
        payload['size'] = np.array(sizes)
    return payload


class ConversationDataset(Dataset):
    def __init__(self, root, annotation, groups=None, max_size=20000000, min_size=224, campo=None):
            data = json.load(open(annotation, 'r'))
            self.root = root
            self.image = []
            self.conversation = []
            self.sizes = []
            print('campo:', campo)
            print('groups', groups)

            for sample in data:
                if 'user_group' not in sample.keys() or sample['user_group'] in groups:
                    if 'size' not in sample.keys() or (sample['size'] >= min_size and sample['size'] < max_size):
                        if campo is None or ('campo' in sample.keys() and campo == sample['campo']):
                            if 'cd_guid' in sample:
                                self.image.append(os.path.join(self.root, '{}.png'.format(sample['cd_guid'])))

                            elif 'image_name' in sample:
                                self.image.append(os.path.join(self.root, sample['image_name'].replace('\\', '/')))

                            elif 'image_id' in sample:
                                self.image.append(os.path.join(self.root, '{}.png'.format(sample['image_id'])))
                                
                            else:
                                raise ValueError('there is no image in the dataset')

                            self.conversation.append(json.dumps(sample['conversation']))

                if 'size' in sample.keys():
                    self.sizes.append(sample['size'])

    def __getitem__(self, index):
        payload = {
            'image': self.image[index],
            'conversation': self.conversation[index],
        }

        if len(self.sizes) > 0:
            payload['size'] = self.sizes[index]

        return payload
    
    def __len__(self):
        return len(self.image)

    def get_loader(self, collate_fn, batch_size:int, shuffle:bool,):
        '''
        get torch dataloader
        :param batch_size: batch size for the dataloader
        :return: dataloader
        '''
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


class DatasetBatchSampler(Sampler):
    def __init__(self, len1, len2, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices1 = list(range(len1))
        self.indices2 = list(range(len1, len1 + len2))

    def __iter__(self):
        idx1 = self.indices1.copy()
        idx2 = self.indices2.copy()

        if self.shuffle:
            random.shuffle(idx1)
            random.shuffle(idx2)

        batches = []

        for i in range(0, len(idx1), self.batch_size):
            batches.append(idx1[i:i+self.batch_size])

        for i in range(0, len(idx2), self.batch_size):
            batches.append(idx2[i:i+self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        return (
            (len(self.indices1) + self.batch_size - 1) // self.batch_size +
            (len(self.indices2) + self.batch_size - 1) // self.batch_size
        )

class DistributedDatasetBatchSampler(Sampler):
    def __init__(
        self,
        len1,
        len2,
        batch_size,
        shuffle=True,
        drop_last=False,
        num_replicas=None,
        rank=None,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.len1 = len1
        self.len2 = len2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):

        rng = random.Random(self.epoch)

        idx1 = list(range(self.len1))
        idx2 = list(range(self.len1, self.len1 + self.len2))

        if self.shuffle:
            rng.shuffle(idx1)
            rng.shuffle(idx2)

        batches = []

        def make_batches(indices):
            bs = self.batch_size
            if self.drop_last:
                end = len(indices) // bs * bs
            else:
                end = len(indices)

            return [indices[i:i+bs] for i in range(0, end, bs)
                    if len(indices[i:i+bs]) == bs or not self.drop_last]

        batches.extend(make_batches(idx1))
        batches.extend(make_batches(idx2))

        if self.shuffle:
            rng.shuffle(batches)

        # pad so every rank gets the same number of batches
        total_batches = len(batches)
        per_rank = math.ceil(total_batches / self.num_replicas)
        total_size = per_rank * self.num_replicas

        if total_size > total_batches:
            batches += batches[: total_size - total_batches]

        # split among ranks
        batches = batches[self.rank:total_size:self.num_replicas]

        return iter(batches)

    def __len__(self):
        total_batches = (
            math.ceil(self.len1 / self.batch_size)
            + math.ceil(self.len2 / self.batch_size)
        )
        return math.ceil(total_batches / self.num_replicas)

if __name__ == "__main__":
    dataset1 = ConversationDataset(
        '/nethome/atena_projetos/fvlk/data/Dataset/v4/images/', 
        "/nethome/atena_projetos/fibz/data/Dataset/v4/conv_train.json",
        groups=[2, 3],
        min_size=224,
        max_size=512
        )
    
    dataset2 = ConversationDataset(
            '/nethome/atena_projetos/fvlk/data/Dataset/v4/images/', 
            "/nethome/atena_projetos/fibz/data/Dataset/v4/conv_train.json",
            groups=[2, 3],
            min_size=512,
            )
            
    from torch.utils.data import DataLoader
    from torch.utils.data import ConcatDataset

    train_dataset = ConcatDataset([dataset1, dataset2])

    print(len(dataset1), len(dataset2), len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=DatasetBatchSampler(
            len1=len(dataset1),
            len2=len(dataset2),
            batch_size=32,
        ),
        # collate_fn=data_collator,
    )

    for batch in train_loader:
        nwpu_count = 0
        geo_count = 0
        print(batch['conversation'])
        break