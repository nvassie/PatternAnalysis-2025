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
print(f'Using device: {device}')

image_path = r"N:\code\prostate_data\data\semantic_MRs_anon"
label_path = r"N:\code\prostate_data\data\semantic_labels_anon"

dataset = Prostate3DDataset(image_path=image_path, label_path=label_path)


train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = ThreeDUNet(in_channels=1, out_channels=6)
losses = train(device, model, train_loader, epochs=2, lr=0.001, plot_epoch_results=True)
test(device, model, test_loader)

