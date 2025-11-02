from typing import Tuple
import torch
import os
import imageio
from torch.utils.data import DataLoader, random_split
from dataset import Prostate3DDataset
from modules import ThreeDUNet
from train import DiceCELoss, mask_plot
import numpy as np

def prepare_volumes_any(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
    vol_zhw: (D, H, W) float32  (intensities)
    seg_zhw: (D, H, W) int64    (class ids 0..K-1)
    """
    img = image_tensor
    if img.dim() == 5:
        img = img[0]
    if img.dim() == 4:
        img = img[0]

    seg = mask_tensor
    if seg.dim() == 5:
        seg = seg[0]
    if seg.dim() == 4:
        if seg.shape[0] == 1:
            seg = seg[0]
        else:
            print(4)
            seg = torch.argmax(seg, dim=0)
    if seg.dim() == 3 and seg.shape[0] != img.shape[0] and seg.shape[0] == img.shape[-1]:
        pass

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

def generate_predict_dataset(image_path, label_path, size):
    """
    Generates a dataset with the provided size to run with model with
    """
    images = os.listdir(image_path)
    num_of_images = len(images)

    full_dataset = Prostate3DDataset(image_path, label_path, 0.5)

    predict_num = size
    unused_num = num_of_images - predict_num

    predict_dataset, unused_dataset = random_split(full_dataset, [predict_num, unused_num])

    return predict_dataset

def evaulate(device, model, loader):
    """
    Runs the loaded model
    """
    model.eval()
    criterion = DiceCELoss(1e-6)
    print(f"Starting Evaulating 3D UNet\n")
    with torch.no_grad():
        total_loss = 0
        count = 0
        plot_image = None
        plot_mask = None
        plot_output = None

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if masks.dim() == 5 and masks.size(0) == 1:
                masks = masks.squeeze(0)

            outputs = model(images)
            loss, dice_per_class = criterion(outputs, masks)

            total_loss += loss.item()
            
            if count == len(loader)-1:
                plot_image = images
                plot_mask = masks
                plot_output = outputs
            count += 1
            print(f"       Steps Completed: {count}/{len(loader)}")

    mask_plot(plot_image, plot_mask, plot_output, dice_per_class)

    image_vol1, output_vol = prepare_volumes_any(plot_image, plot_output)
    make_multiclass_overlay_gif(image_vol1, output_vol, out_path="model.gif", fps=8, alpha=0.2)

    image_vol2, mask_vol = prepare_volumes_any(plot_image, plot_mask)
    make_multiclass_overlay_gif(image_vol2, mask_vol, out_path="ground_truth.gif", fps=8, alpha=0.2)

    average_loss = total_loss / len(loader)
    print(f"\nAverage loss while evaulating: {average_loss:.4f}")

def predict():
    """
    Loads a model and creates a dataset then evaulates the model
    and then outputs metrics, plots and gifs
    """
    image_path = r"N:\code\prostate_data\data\semantic_MRs_anon"
    label_path = r"N:\code\prostate_data\data\semantic_labels_anon"

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    #get most recent model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError("The is no models folder in the current directory")
    
    models_list = os.listdir(models_dir)

    if len(models_list) == 0:
        raise FileNotFoundError("The are no models saved")
    
    models_list.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
    model_name = models_list[0]


    print(f"Loading model: {model_name}\n")
    model_path = os.path.join(models_dir, model_name)

    model = ThreeDUNet(in_channels=1, out_channels=6)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    size = 30

    dataset = generate_predict_dataset(image_path, label_path, size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    evaulate(device, model, loader)

if __name__ == "__main__":
    predict()