import torch
import torch.nn as nn
import torch.optim as optim

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

def train(device, model, train_loader, epochs=3, lr=0.001, plot_epoch_results=False):
    model.to(device)
    criterion = DiceCELoss(1e-6)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []

    print("Starting training 3D UNet")
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
                print(f"{count}/{len(train_loader)} Completed")

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

    print(" Training complete with 3D UNet")
    return losses