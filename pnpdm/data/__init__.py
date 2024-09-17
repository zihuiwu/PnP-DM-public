import torch
import numpy as np
from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(
        dataset: VisionDataset,
        batch_size: int, 
        num_workers: int, 
        train: bool
    ):
    return DataLoader(
        dataset,
        batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train
    )


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, head: int = None, grayscale: bool = False, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.grayscale = grayscale
        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        if head is not None:
            self.fpaths = self.fpaths[:head]
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    @property
    def display_name(self):
        if self.grayscale:
            return 'ffhq-gs'
        else:
            return 'ffhq'

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        if self.grayscale:
            img = Image.open(fpath).convert("L")
            img = torch.from_numpy(np.array(img).astype(np.float32))[None] / 255.0
        else:
            img = Image.open(fpath).convert('RGB')
            img = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1) / 255.0
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
