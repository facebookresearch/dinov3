import os.path

from dinov3.data import (
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
    CombinedDataLoader,
)

if __name__ == "__main__":
    dataset = make_dataset(
        dataset_str=f'GeoDataset:split=TEST:root=../../geo',
        )
