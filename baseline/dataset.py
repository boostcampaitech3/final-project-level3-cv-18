import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize=(512,512), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), **args):
        self.transform = A.Compose([
            A.Resize(resize[0], resize[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform(image)


class TrainDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        label = self.labels[index]
        return image, label


class ValidDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        label = self.labels[index]
        return image, label