from typing import List
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

def combine_labels_images(images_path: str, labels_path: str):
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)

    combined = []

    for i in range(0, len(images)):
        image = os.path.join(image_path, images[i])
        label = os.path.join(label_path, labels[i])
        pair = (image, label)
        combined.append(pair)

    return combined



def to_channels (arr: np.ndarray, dtype = np.uint8) -> np.ndarray :
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype = dtype)
    for c in channels:
        c = int(c)
        res[...,c:c+1][arr == c] = 1

    return res

class Prostate3DDataset(Dataset):
    """Dataset for color images and binary masks."""

    def __init__(self, image_path=r"N:\code\prostate_data\data\semantic_MRs_anon", label_path=r"N:\code\prostate_data\data\semantic_labels_anon"):
        self.dataset = combine_labels_images(image_path, label_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get image and mask
        image, mask = self.dataset[index]
        image = nib.load(image).get_fdata()
        mask = nib.load(mask).get_fdata()

        mask = to_channels(mask)

        # Convert to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.squeeze(0) 
        mask = mask.squeeze(0)

        return image, mask

image_path = r"N:\code\prostate_data\data\semantic_MRs_anon"
label_path = r"N:\code\prostate_data\data\semantic_labels_anon"