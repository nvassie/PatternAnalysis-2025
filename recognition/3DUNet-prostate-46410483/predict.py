import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from dataset import Prostate3DDataset
from modules import ThreeDUNet
import torchvision.transforms.functional as TF
from train import train, test

import numpy as np
import matplotlib.pyplot as plt
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

image_path = r"N:\code\prostate_data\data\semantic_MRs_anon"
label_path = r"N:\code\prostate_data\data\semantic_labels_anon"
#model_path = r"N:\code\PatternAnalysis-2025\recognition\3DUNet-prostate-46410483\models\2025-11-01_0.0636.pth"

def generate_indices_for_datasets(image_path):
    """
    generates random three lists of indices for the training (80%), testing (10%) and 
    validation (10%) datasets, based on provided dataset size.

    The indices are unique to prevent data leakage.
    """
    images = os.listdir(image_path)
    num_of_images = len(images)
    list_of_indices = list(range(num_of_images))

    random.shuffle(list_of_indices)

    train_num = int(0.8 * num_of_images)
    val_num = int(0.1 * num_of_images)

    train_indices = list_of_indices[:train_num]
    val_indices = list_of_indices[train_num:train_num + val_num]
    test_indices = list_of_indices[train_num + val_num:]

    return train_indices, test_indices, val_indices

train_indices, test_indices, val_indices = generate_indices_for_datasets(image_path)

training_dataset = Prostate3DDataset(image_path=image_path, label_path=label_path, indices=train_indices)
test_dataset = Prostate3DDataset(image_path=image_path, label_path=label_path, indices=test_indices)


train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ThreeDUNet(in_channels=1, out_channels=6)
# if model_path:
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
losses = train(device, model, train_loader, epochs=10, lr=0.001, save=True, plot_epoch_results=False)
test(device, model, test_loader, plot=True)

generate_indices_for_datasets(image_path)