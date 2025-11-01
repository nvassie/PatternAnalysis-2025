import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import date
from plotting import epoch_plot

class DiceCELoss(nn.Module):
    """
    """
    def __init__(self, smoothing=1e-5, ce_weight=0.5, dice_weight=0.5):
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
        dice_loss = calc_dice_loss(probability, one_hot, self.smoothing)
        dice_loss = dice_loss * self.dice_weight

        return ce_loss + dice_loss

def calc_dice_loss(probability, one_hot, smoothing):
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probability * one_hot, dims)
    denominator = torch.sum(probability + one_hot, dims)
    dice_per_class = (2 * intersection + smoothing) / (denominator + smoothing)
    dice_mean = dice_per_class.mean() 
    return (1 - dice_mean)

def train(device, model, train_loader, epochs=3, lr=0.001, save=False, plot_epoch_results=False):
    model.to(device)
    criterion = DiceCELoss(1e-6)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []

    print(f"Starting training 3D UNet")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        count = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device).float(), masks.to(device).long()


            if masks.dim() == 5 and masks.size(0) == 1:
                masks = masks.squeeze(0)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            count += 1
            if (count % 50 == 0):
                print(f"       Epoch: {epoch}, Steps Completed: {count}/{len(train_loader)}")

        if plot_epoch_results:
            model.eval()
            epoch_plot(images, masks, outputs)
            model.train()        


        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"    📍 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

    print(f"Training complete with 3D UNet\n")
    
    if save:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, "models")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{date.today()}_{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}\n")

    return losses

def test(device, model, test_loader, plot=False):
    model.eval()
    criterion = DiceCELoss(1e-6)
    print(f"Starting Testing 3D UNet")
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
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            
            if count == len(test_loader)-1:
                plot_image = images
                plot_mask = masks
                plot_output = outputs
            count += 1
            print(f"       Steps Completed: {count}/{len(test_loader)}")
            
    if plot and plot_image != None and plot_mask != None and plot_output != None:
        epoch_plot(plot_image, plot_mask, plot_output)

    average_loss = total_loss / len(test_loader)
    print(f"Average loss while testing: {average_loss:.4f}")