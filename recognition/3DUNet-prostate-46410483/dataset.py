from typing import List
import numpy as np
import nibabel as nib
import os
from torch.utils.data import Dataset
import torch

def combine_labels_images(images_path: str, labels_path: str, size: int):
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)

    combined = []

    image_size = len(images)

    if size > 0 and size < image_size:
        image_size = size

    for i in range(0, image_size):
        image = os.path.join(images_path, images[i])
        label = os.path.join(labels_path, labels[i])
        pair = (image, label)
        combined.append(pair)

    return combined

def pad_image(image):
    """
    Pad the inputted image to be divisable by 8 to work with the 3D UNet
    """
    _, _, depth, height, width = image.shape
    depth_size = (8 - (depth % 8))
    depth_before = depth_size // 2
    depth_after = depth_size - depth_before
    height_size = (8 - (height % 8))
    height_before = height_size // 2
    height_after = height_size - height_before
    width_size = (8 - (width % 8))
    width_before = width_size // 2
    width_after = width_size - width_before
    padding = [width_before, width_after, height_before, height_after, depth_before, depth_after]
    image = torch.nn.functional.pad(image, padding)
    return image


class Prostate3DDataset(Dataset):
    """Dataset for color images and binary masks."""

    def __init__(self, image_path=r"N:\code\prostate_data\data\semantic_MRs_anon", label_path=r"N:\code\prostate_data\data\semantic_labels_anon", training=True, size=-1):
        self.dataset = combine_labels_images(image_path, label_path, size)
        self.downsample = 0.5
        self.training = training

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get image and mask
        image, mask = self.dataset[index]
        image = nib.load(image).get_fdata().astype(np.float32)
        mask = nib.load(mask).get_fdata().astype(np.uint8)

        # Normalise image
        image = (image - image.mean()) / (image.std() + 1e-7)

        # Convert to tensor
        image = torch.from_numpy(image)[None, None] 
        mask = torch.from_numpy(mask)[None, None].float()

        if self.training:
            image = torch.nn.functional.interpolate(image, scale_factor=self.downsample, mode='trilinear', align_corners=False)
            mask = torch.nn.functional.interpolate(mask, scale_factor=self.downsample, mode='trilinear')

            image = pad_image(image)
            mask = pad_image(mask)

        image = image.squeeze(0)
        mask = mask.squeeze(0).long()

        return image, mask