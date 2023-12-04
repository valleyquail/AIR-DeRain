import numpy as np
import torch
import os
import pandas as pd
import torchvision.models
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class SDDTrainingData(Dataset):
    def __init__(self, csv_file, images_root, transform=None):
        self.annotations = pd.read_csv('../BoundingBoxPairs/Annotations/annotations.csv')
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_name = os.path.join(self.images_root, self.annotations.iloc[item, 0])
        image = read_image(img_name)

        boxes = self.annotations.iloc[item, 1:]
        boxes = np.array([boxes], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'annotations': boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample


transforms = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.transforms



model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    weights_backbone=MobileNet_V3_Large_Weights,
    progress=True,
    num_classes=2)
