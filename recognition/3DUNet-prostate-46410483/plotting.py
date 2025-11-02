import imageio
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

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


def prepare_volumes_any(image_tensor: torch.Tensor,
                        mask_tensor: torch.Tensor):
    """
    Inputs can be:
      image_tensor: 
        - (B, C, H, W, D)
        - (C, H, W, D)
        - (H, W, D)
      mask_tensor:
        - label map (H, W, D)
        - (B, 1, H, W, D)
        - one-hot / logits (B, Cclasses, H, W, D) or (Cclasses, H, W, D)

    Returns:
      vol_zhw: (D, H, W) float32  (intensities)
      seg_zhw: (D, H, W) int64    (class ids 0..K-1)
    """

    # ---- 1. IMAGE SHAPE NORMALIZATION ----
    img = image_tensor

    # remove batch dim if present
    if img.dim() == 5:
        # assume (B, C, H, W, D)
        img = img[0]  # take first in batch -> (C, H, W, D)

    if img.dim() == 4:
        # could be (C, H, W, D) or (H, W, D, 1)
        # we want to end up with (H, W, D)
        if img.shape[0] == 1:
            # (1, H, W, D) -> squeeze channel
            img = img[0]  # (H, W, D)
        elif img.shape[-1] == 1:
            # (H, W, D, 1) rare, but just in case
            img = img[..., 0]  # (H, W, D)

    # by now we expect img to be (H, W, D)
    assert img.dim() == 3, f"Image after squeeze is {img.shape}, expected 3D (H,W,D)"

    # ---- 2. MASK SHAPE NORMALIZATION ----
    seg = mask_tensor

    # remove batch if present
    if seg.dim() == 5:
        # assume (B, C_or_1, H, W, D)
        seg = seg[0]  # (C_or_1, H, W, D)

    if seg.dim() == 4:
        # could be:
        #   (1, H, W, D)               -> single-channel labels
        #   (Cclasses, H, W, D)        -> per-class scores / one-hot
        if seg.shape[0] == 1:
            seg = seg[0]  # (H, W, D)
        else:
            # assume channel-first classes
            # convert to argmax class map
            seg = torch.argmax(seg, dim=0)  # (H, W, D)

    if seg.dim() == 3 and seg.shape[0] != img.shape[0] and seg.shape[0] == img.shape[-1]:
        # edge case: seg is (D,H,W) but img is (H,W,D); rotate seg to (H,W,D)
        # but let's handle that after we align axes below, so we won't do anything here
        pass

    # now seg should be (H, W, D) as class indices per voxel
    assert seg.dim() == 3, f"Mask after squeeze/argmax is {seg.shape}, expected 3D (H,W,D)"

    # ---- 3. PUT DEPTH FIRST -> (D, H, W) ----
    # Right now img, seg are (H, W, D)
    # We convert both to (D, H, W)
    vol_zhw = img.permute(2, 0, 1).contiguous().float()
    seg_zhw = seg.permute(2, 0, 1).contiguous().long()

    return vol_zhw, seg_zhw


def make_multiclass_overlay_gif(
    volume,   # (D,H,W) float
    segmask,  # (D,H,W) int labels, 0 = background
    out_path="overlay.gif",
    fps=10,
    alpha=0.5
):
    volume = volume.detach().cpu()
    segmask = segmask.detach().cpu()

    assert volume.shape == segmask.shape, f"vol {volume.shape} vs seg {segmask.shape} mismatch"

    D, H, W = volume.shape

    vmin = float(volume.min())
    vmax = float(volume.max())
    denom = (vmax - vmin) + 1e-8

    class_colors = np.array([
        [0.0, 0.0, 0.0],   # class 0 = background (no tint)
        [1.0, 0.0, 0.0],   # class 1 = red
        [0.0, 1.0, 0.0],   # class 2 = green
        [0.0, 0.0, 1.0],   # class 3 = blue
        [1.0, 1.0, 0.0],   # class 4 = yellow
        [1.0, 0.0, 1.0],   # class 5 = magenta
    ], dtype=np.float32)

    frames = []

    for i in range(D):
        slc = volume[i].numpy()      # (H,W)
        seg = segmask[i].numpy()     # (H,W), ints 0..5

        base = (slc - vmin) / denom
        base = np.clip(base, 0.0, 1.0)

        rgb = np.stack([base, base, base], axis=-1)  # (H,W,3)

        overlay_colors = class_colors[seg]           # (H,W,3)

        non_bg = seg != 0
        non_bg_3 = np.stack([non_bg]*3, axis=-1)

        rgb[non_bg_3] = (
            (1 - alpha) * rgb[non_bg_3] +
            alpha      * overlay_colors[non_bg_3]
        )

        rgb_uint8 = (rgb * 255).astype(np.uint8)
        frames.append(rgb_uint8)

    imageio.mimsave(out_path, frames, duration=1.0/fps, loop=0)
    print(f"Saved GIF to {out_path} ({D} slices).")