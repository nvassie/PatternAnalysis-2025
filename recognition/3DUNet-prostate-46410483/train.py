import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from modules import ThreeDUNet
from datetime import date
from dataset import Prostate3DDataset

# Constant
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

# Change the following to your valid path
IMAGE_PATH = r"N:\code\prostate_data\data\semantic_MRs_anon"
LABEL_PATH = r"N:\code\prostate_data\data\semantic_labels_anon"

# Hyper Parameters
DOWNSAMPLE_FACTOR = 0.5
LR = 0.001
EPOCHS = 15
IN_CHANNELS = 1
OUT_CHANNELS = 6
CE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
SMOOTHING = 1e-6
CLASS_NUMBER = 6

def epoch_plot(images, masks, outputs, dice_values):
        with torch.no_grad():
            image = images[0]
            mask = masks[0]
            output = outputs[0]

            prediction = torch.argmax(output, dim=0)

            # middle slice
            mid_slice = image.shape[1] // 2

            image_slice   = image[0, mid_slice].detach().cpu().numpy()
            mask_slice  = mask[mid_slice].detach().cpu().numpy()
            prediction_slice  = prediction[mid_slice].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12,4))

        # image plot
        axes[0].set_title(f"Image slice {mid_slice}")
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].axis("off")

        # ground truth mask
        axes[1].set_title("Ground Truth Mask")
        axes[1].imshow(image_slice, cmap="gray")
        axes[1].imshow(mask_slice, cmap="turbo", alpha=0.4, vmin=0, vmax=5)
        axes[1].axis("off")

        # model mask
        axes[2].set_title("Model Mask")
        axes[2].imshow(image_slice, cmap="gray")
        axes[2].imshow(prediction_slice, cmap="turbo", alpha=0.4, vmin=0, vmax=5)
        axes[2].axis("off")

        num_classes = 6
        cmap = plt.get_cmap("turbo")
        legend_labels = []

        for i in range(num_classes):
            label = mpatches.Patch(color=cmap(i / (num_classes - 1)), label=CLASS_NAMES[i] + f" ({dice_values[i]:.4f})")
            legend_labels.append(label)

        fig.legend(handles=legend_labels, loc='outside right')

        plt.tight_layout()
        plt.show()

def loss_plot(epoch_num, train_losses, val_losses):
    """
    Plots the average training and validation loss per epoch
    """

    x_axis = []

    for i in range(epoch_num):
        x_axis.append(i)

    plt.plot(x_axis, train_losses, color="Red", label="Average Training Loss")
    plt.plot(x_axis, val_losses, color="Blue", label="Average Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

def dice_plot(training_dice_scores, validation_dice_scores):
    """
    Plots each class's Dice similarity coefficients for training and validation per epoch.

    training_dice_scores: list of length num_epochs
        each element is a list [c1, c2, c3, ..., cK] for that epoch
    validation_dice_scores: same structure as training_dice_scores
    """

    # Convert to numpy for easy transpose
    training_arr = np.array(training_dice_scores)       # shape [num_epochs, num_classes]
    validation_arr = np.array(validation_dice_scores)   # shape [num_epochs, num_classes]

    num_epochs = training_arr.shape[0]
    num_classes = training_arr.shape[1]

    # x-axis is epochs 0..num_epochs-1
    x_axis = np.arange(num_epochs)

    # one color per class, consistent across both subplots
    cmap = plt.cm.turbo
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    # Create two vertically stacked subplots that share x
    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Training subplot ---
    for class_idx in range(num_classes):
        ax_train.plot(
            x_axis,
            training_arr[:, class_idx],          # all epochs for this class
            color=colors[class_idx],
            label=CLASS_NAMES[class_idx]
        )

    ax_train.set_title("Training Dice Similarity Coefficients")
    ax_train.set_ylabel("Dice Coefficient")
    ax_train.legend(loc="lower right")
    ax_train.grid(True, linestyle="--", alpha=0.4)

    # --- Validation subplot ---
    for class_idx in range(num_classes):
        ax_val.plot(
            x_axis,
            validation_arr[:, class_idx],
            color=colors[class_idx],
            label=CLASS_NAMES[class_idx]
        )

    ax_val.set_title("Validation Dice Similarity Coefficients")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Dice Coefficient")
    ax_val.legend(loc="lower right")
    ax_val.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def multiclass_dice_plot(epoch_num, training_mcds, validation_mcdc):
    """
    Plots the multiclass dice coefficients for training and validation per epoch
    """

    x_axis = []

    for i in range(epoch_num):
        x_axis.append(i)

    plt.plot(x_axis, training_mcds, color="Red", label="Training Multiclass Dice Coefficient")
    plt.plot(x_axis, validation_mcdc, color="Blue", label="Validation Multiclass Dice Coefficient")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Multiclass Dice Coefficient")
    plt.title("Training vs Validation Multiclass Dice Coefficients")
    plt.legend()
    plt.show()

class DiceCELoss(nn.Module):
    """
    A loss function that combines different weights of dice loss and cross entropy loss
    """
    def __init__(self, smoothing=1e-6, ce_weight=0.5, dice_weight=0.5):
        super(DiceCELoss, self).__init__()
        self.smoothing = smoothing
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        """
        """
        ce_loss = self.ce(prediction, target) * self.ce_weight
        probability = torch.softmax(prediction, dim=1)

        one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        one_hot = one_hot.permute(0,4,1,2,3)
        dice_loss, dice_per_class = calc_dice_loss(probability, one_hot, self.smoothing)
        dice_loss = dice_loss * self.dice_weight

        return (ce_loss + dice_loss), dice_per_class.detach().cpu().numpy()

def calc_dice_loss(probability, one_hot, smoothing):
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probability * one_hot, dims)
    denominator = torch.sum(probability + one_hot, dims)
    dice_per_class = (2 * intersection + smoothing) / (denominator + smoothing)
    dice_mean = dice_per_class.mean()
    return (1 - dice_mean), dice_per_class

def generate_datasets(image_path, label_path, downsample_factor):
    """
    generates random three lists of indices for the training (80%), testing (10%) and 
    validation (10%) datasets, based on provided dataset size.

    The indices are unique to prevent data leakage.
    """
    images = os.listdir(image_path)
    num_of_images = len(images)

    full_dataset = Prostate3DDataset(image_path, label_path, downsample_factor)

    train_num = int(0.8 * num_of_images)
    val_num = int(0.1 * num_of_images)
    test_num = num_of_images - (train_num + val_num)

    training_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_num, val_num, test_num])

    return training_dataset, validation_dataset, test_dataset

def train(device, model, train_loader, validation_loader, epochs=10, lr=0.001, save=False, plot_epoch_results=False):
    model.to(device)
    criterion = DiceCELoss(SMOOTHING, CE_WEIGHT, DICE_WEIGHT)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    training_dice_scores = []
    validation_dice_scores = []
    multiclass_training_dice_scores = []
    multiclass_validation_dice_scores = []

    print(f"Starting training 3D UNet\n")

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        validation_loss = 0
        count = 0
        for _, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device).float(), masks.to(device).long()


            if masks.dim() == 5 and masks.size(0) == 1:
                masks = masks.squeeze(0)

            optimizer.zero_grad()
            outputs = model(images)

            loss, dice_per_class = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            count += 1
            if (count % 50 == 0):
                print(f"          Epoch: {epoch}, Steps Completed: {count}/{len(train_loader)}")

        validation_epoch_loss, validation_dice_per_class = validate(device, model, validation_loader)
        validation_loss += validation_epoch_loss
        model.train()

        if plot_epoch_results:
            model.eval()
            epoch_plot(images, masks, outputs, dice_per_class)
            model.train()


        avg_loss = training_loss / len(train_loader)
        avg_val_loss = validation_loss / len(validation_loader)
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        epoch_training_dice_scores = [dice_per_class[0], dice_per_class[1], dice_per_class[2], dice_per_class[3], dice_per_class[4], dice_per_class[5]]
        training_dice_scores.append(epoch_training_dice_scores)

        epoch_validation_dice_scores = [validation_dice_per_class[0], validation_dice_per_class[1], validation_dice_per_class[2], validation_dice_per_class[3], validation_dice_per_class[4], validation_dice_per_class[5]]
        validation_dice_scores.append(epoch_validation_dice_scores)

        training_epoch_msdc = (dice_per_class[0] + dice_per_class[1] + dice_per_class[2] + dice_per_class[3] + dice_per_class[4] + dice_per_class[5]) / CLASS_NUMBER
        multiclass_training_dice_scores.append(training_epoch_msdc)

        validation_epoch_msdc = (validation_dice_per_class[0] + validation_dice_per_class[1] + validation_dice_per_class[2] + validation_dice_per_class[3] + validation_dice_per_class[4] + validation_dice_per_class[5]) / CLASS_NUMBER
        multiclass_validation_dice_scores.append(validation_epoch_msdc)

        print(f"\n    📍 Epoch {epoch+1}/{epochs} Complete: Avg Training Loss = {avg_loss:.4f}, Avg Validation Loss = {avg_val_loss:.4f}")
        print(f"          Dice Similarity Coefficients:")
        print(f"             Multiclass: {training_epoch_msdc:.4f}\n"
            f"             Background: {dice_per_class[0]:.4f}\n"
            f"             Body: {dice_per_class[1]:.4f}\n"
            f"             Bone: {dice_per_class[2]:.4f}\n"
            f"             Bladder: {dice_per_class[3]:.4f}\n"
            f"             Rectum: {dice_per_class[4]:.4f}\n"
            f"             Prostate: {dice_per_class[5]:.4f}\n")

    print(f"Training complete with 3D UNet\n")
    print(f"Final average training loss: {avg_loss:.4f}")
    print(f"Final average training loss: {avg_val_loss:.4f}")
    print(f"Final multiclass dice similartiy coefficient: {training_epoch_msdc:.4f}\n")
    loss_plot(epochs, train_losses, val_losses)
    dice_plot(training_dice_scores, validation_dice_scores)
    multiclass_dice_plot(epochs, multiclass_training_dice_scores, multiclass_validation_dice_scores)
    
    if save:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, "models")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{date.today()}_{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}\n")

def validate(device, model, validation_loader):
    model.eval()
    criterion = DiceCELoss(SMOOTHING, CE_WEIGHT, DICE_WEIGHT)
    with torch.no_grad():
        total_loss = 0

        for images, masks in validation_loader:
            images = images.to(device)
            masks = masks.to(device)

            if masks.dim() == 5 and masks.size(0) == 1:
                masks = masks.squeeze(0)

            outputs = model(images)
            loss, dice_per_class = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss, dice_per_class


def test(device, model, test_loader, plot=False):
    model.eval()
    criterion = DiceCELoss(SMOOTHING, CE_WEIGHT, DICE_WEIGHT)
    print(f"Starting Testing 3D UNet\n")
    with torch.no_grad():
        total_loss = 0
        count = 0
        plot_image = None
        plot_mask = None
        plot_output = None

        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            if masks.dim() == 5 and masks.size(0) == 1:
                masks = masks.squeeze(0)

            outputs = model(images)
            loss, dice_per_class = criterion(outputs, masks)

            total_loss += loss.item()
            
            if count == len(test_loader)-1:
                plot_image = images
                plot_mask = masks
                plot_output = outputs
            count += 1
            print(f"       Steps Completed: {count}/{len(test_loader)}")
            
    if plot and plot_image != None and plot_mask != None and plot_output != None:
        epoch_plot(plot_image, plot_mask, plot_output, dice_per_class)

    average_loss = total_loss / len(test_loader)
    print(f"\nAverage loss while testing: {average_loss:.4f}")

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    training_dataset, validation_dataset, test_dataset = generate_datasets(IMAGE_PATH, LABEL_PATH, DOWNSAMPLE_FACTOR)

    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = ThreeDUNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    train(device, model, train_loader, validation_loader, epochs=EPOCHS, lr=LR, save=True, plot_epoch_results=False)
    test(device, model, test_loader, plot=True)