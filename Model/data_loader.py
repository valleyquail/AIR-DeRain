import matplotlib.pyplot as plt
import torch
import os
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(42)


class ImagePairsDataset(VisionDataset):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)

    def __len__(self):
        path = self.root
        # print(len(os.listdir(path)) - 1)
        return len(os.listdir(path)) - 1

    def __getitem__(self, idx):
        blurred_name = os.path.join(self.root + 'blurred/' + str(idx) + '_blur.png')
        original_name = os.path.join(self.root + 'clean/' + str(idx) + '_mask.png')

        blurred = read_image(blurred_name)
        original = read_image(original_name)

        sample = {'image': blurred, 'original': original}
        # sample = blurred
        if self.transform:
            sample = self.transform(sample)

        return sample


def create_data_loader(path: str):
    transform = v2.Compose(
        [
            v2.Resize(240),
            v2.RandomRotation(degrees=(0, 180)),
        ]
    )
    dataset = ImagePairsDataset(path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader
