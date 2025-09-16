import json
import os
import random
from enum import Enum
from typing import Callable, Dict, List, Optional, Union
from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class _Split(Enum):
    TRAIN = "train"
    TEST = "test"


def read_data(root: str, split: _Split) -> List[Dict]:
    annotation = os.path.join(root, f'caption_dataset_{split.value}.json')
    annotation = os.path.abspath(os.path.normpath(annotation))
    print(annotation)
    data = {}
    json_data = json.load(open(annotation, 'r'))

    for sample in json_data:
        id = sample['cd_guid']
        data[id] = {'id': id,
                    'captions': sample['descricao'],
                    'image': os.path.abspath(os.path.join(root, 'images', f'{id}.png'))}
    return list(data.values())


class GeoDataset(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CocoCaptions.Split",
        root: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self.image_captions = read_data(root, split)

    def get_image_relpath(self, index: int) -> str:
        image_path = self.image_captions[index]["image"]
        return image_path

    def get_image_data(self, index: int) -> bytes:
        image_path = self.get_image_relpath(index)
        with open(image_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> str:
        return self.image_captions[index]["captions"]

    def __len__(self) -> int:
        return len(self.image_captions)
