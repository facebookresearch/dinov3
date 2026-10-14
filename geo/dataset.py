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
    scales = []
    for sample in batch:
        payload['image'].append(sample['image'])
        payload['conversation'].append(json.loads(sample['conversation']))
        if 'size' in sample.keys():
            sizes.append(sample['size'])
        if 'scale' in sample.keys():
            scales.append(sample['scale'])

    if len(sizes) > 0:
        payload['size'] = np.array(sizes)
    if len(scales) > 0:
        payload['scale'] = np.array(scales)
    return payload


# conversions of data.scale_unit to millimetres per pixel
MM_PER_PIXEL = {
    'px_per_mm': lambda v: 1.0 / v,
    'mm_per_px': lambda v: v,
    'um_per_px': lambda v: v / 1000.0,
}


def data_options(conf):
    """Image source and scale options of conf.data, as ConversationDataset arguments:
        small_root   folder with the non-standardized images, same file names as data.root (default: data.root)
        scale_field  sample field with the scale of the image actually loaded, in scale_unit (default: no scale)
        scale_unit   px_per_mm (default), mm_per_px or um_per_px"""
    data = conf.data
    return {'small_root': data.get('small_root'), 'scale_field': data.get('scale_field'),
            'scale_unit': data.get('scale_unit', 'px_per_mm')}


class ConversationDataset(Dataset):
    def __init__(self, root, annotation, groups=None, max_size=20000000, min_size=224, campo=None, campos=None,
                 small_root=None, scale_field=None, scale_unit='px_per_mm'):
            # campo: a single description type; campos: a list of them. Both None keeps every sample
            # small_root: images read from this folder instead of root (same names); their size (shorter side) is
            #   read from the file, since the split's 'size' describes the standardized images
            # scale_field: sample field with the image scale; every sample then has 'scale' in mm per pixel
            data = json.load(open(annotation, 'r'))
            self.root = small_root or root
            self.image = []
            self.conversation = []
            self.labels = []  # per sample, None when the split has no labels
            self.campos = []
            self.sizes = []
            self.scales = []  # mm per pixel per sample, None without scale_field
            self.scale_field = scale_field
            if scale_field is not None and scale_unit not in MM_PER_PIXEL:
                raise ValueError(f'scale_unit must be one of {sorted(MM_PER_PIXEL)}, got {scale_unit}')
            selected = set(campos or []) | ({campo} if campo is not None else set())
            print('campos:', sorted(selected) if selected else 'all')
            print('groups', groups)
            if small_root:
                print('images from', small_root)
            file_sizes = {}  # shorter side per image file, read once per image
            missing_scale = 0

            for sample in data:
                if 'user_group' not in sample.keys() or sample['user_group'] in groups:
                    if not selected or sample.get('campo') in selected:
                        if 'cd_guid' in sample:
                            image = os.path.join(self.root, '{}.png'.format(sample['cd_guid']))

                        elif 'image_name' in sample:
                            image = os.path.join(self.root, sample['image_name'].replace('\\', '/'))

                        elif 'image_id' in sample:
                            image = os.path.join(self.root, '{}.png'.format(sample['image_id']))

                        else:
                            raise ValueError('there is no image in the dataset')

                        size = sample.get('size')
                        if small_root:
                            if image not in file_sizes:
                                with Image.open(image) as im:  # reads the header only
                                    file_sizes[image] = min(im.size)
                            size = file_sizes[image]
                        if size is not None and not (size >= min_size and size < max_size):
                            continue

                        scale = None
                        if scale_field is not None:
                            if sample.get(scale_field) is None:
                                missing_scale += 1
                                continue
                            scale = MM_PER_PIXEL[scale_unit](float(sample[scale_field]))

                        self.image.append(image)
                        self.conversation.append(json.dumps(sample['conversation']))
                        self.labels.append(sample.get('labels'))
                        self.campos.append(sample.get('campo'))
                        self.scales.append(scale)
                        # only for kept samples, so sizes[i] belongs to image[i]
                        if size is not None:
                            self.sizes.append(size)

            if missing_scale:
                print(f'skipped {missing_scale} samples without {scale_field}')

    def __getitem__(self, index):
        payload = {
            'image': self.image[index],
            'conversation': self.conversation[index],
        }

        if len(self.sizes) > 0:
            payload['size'] = self.sizes[index]
        if self.scale_field is not None:
            payload['scale'] = self.scales[index]

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
        # a single process (no torchrun) is one replica
        distributed = torch.distributed.is_initialized()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if distributed else 1
        if rank is None:
            rank = torch.distributed.get_rank() if distributed else 0

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