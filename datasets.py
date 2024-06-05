from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
import h5py
import random
import cv2
from tqdm import tqdm
from collections import Counter
import utils


class WatertankDataset(Dataset):
    def __init__(self, path: Path, transforms=False):
        self.path_to_images = path
        self.classes = {dir.name: i for i, dir in
                        enumerate(self.path_to_images.iterdir())}
        self.transforms = transforms
        self.images, self.labels = utils.get_images_labels(self.path_to_images,
                                                           self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sonar_image = self.images[index]
        if self.transforms:
            rdeg = (np.random.random() - 0.5) * 20
            sonar_image = utils.rotate_cv(sonar_image, rdeg)
            if np.random.random() > 0.5:
                sonar_image = np.fliplr(sonar_image.copy())
        sonar_image = sonar_image - 84.5  # 84.5 is mean of pixels in dataset
        return np.rollaxis(sonar_image, 2), self.labels[index]


class RotNetData(Dataset):
    def __init__(self, dir: Path, transforms=False):
        self.image_dir = dir
        self.classes = {dir.name: i for i, dir in
                        enumerate(self.image_dir.iterdir())}
        self.transforms = transforms
        self.images = utils.get_images(self.image_dir)
        self.images, self.labels = utils.get_rotations(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sonar_image = self.images[index]
        if self.transforms:
            rdeg = (np.random.random() - 0.5) * 20
            sonar_image = utils.rotate_cv(sonar_image, rdeg)
            # if np.random.random() > .5:
            #     sonar_image = np.fliplr(sonar_image.copy())
        sonar_image = sonar_image - 84.5
        return np.rollaxis(sonar_image, 2).copy(), self.labels[index]


class MuraData(Dataset):
    def __init__(self, dir_path: Path, transform: bool = False,
                 img_size: int = 320):
        self.image_dir = dir_path
        self.images, self.labels, self.cats = utils.get_mura_images(
            self.image_dir)
        if transform:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        self.pos_weights, self.neg_weights = utils.get_category_counts(
            self.labels, self.cats)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        images = self.images[index]
        image_arr = utils.transform_image(images, self.transforms)
        pos_cat_wt = self.pos_weights[self.cats[index]]
        neg_cat_wt = self.neg_weights[self.cats[index]]
        return image_arr, self.labels[index], (pos_cat_wt, neg_cat_wt)
    
    def train_dataloader(self, batch_size):
        return DataLoader(dataset=self, batch_size=batch_size,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self, batch_size=8, shuffle=False)


class CheXpertData(Dataset):
    def __init__(self, dir_path: Path, label_csv: Path,
                 transform: bool = False, data_type: str = 'train',
                 uncertainty_type: str = 'own_class'):
        self.image_dir = dir_path
        self.label_csv = label_csv
        self.data_type = data_type
        self.images, self.labels = utils.get_chexpert_images(
            path=self.image_dir, label_csv=label_csv,
            data_type=self.data_type, uncertainty_type=uncertainty_type)
        if transform:
            self.transforms = A.Compose([
                A.Resize(320, 320),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
                ])
        else:
            self.transforms = A.Compose([
                A.Resize(320, 320),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
                ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image_arr = utils.transform_image(image, self.transforms)
        return image_arr, self.labels[index]
