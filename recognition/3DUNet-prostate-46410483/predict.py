import random
import torch
import os
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset
from modules import ThreeDUNet
from train import DiceCELoss
from plotting import epoch_plot, make_multiclass_overlay_gif, prepare_volumes_any

def generate_predict_dataset(image_path, label_path, size):
    """
    Generates a dataset with the provided size to run with model with
    """
    images = os.listdir(image_path)
    num_of_images = len(images)
    list_of_indices = list(range(num_of_images))

    random.shuffle(list_of_indices)

    dataset_indices = list_of_indices[:size]
    dataset = Prostate3DDataset(image_path, label_path, dataset_indices)

    return dataset

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

    epoch_plot(plot_image, plot_mask, plot_output, dice_per_class)
    vol1, seg1 = prepare_volumes_any(plot_image, plot_output)   # -> (72,136,136) each
    make_multiclass_overlay_gif(vol1, seg1, out_path="my_volume1.gif", fps=8, alpha=0.2)
    vol2, seg2 = prepare_volumes_any(plot_image, plot_mask)   # -> (72,136,136) each
    make_multiclass_overlay_gif(vol2, seg2, out_path="my_volume2.gif", fps=8, alpha=0.2)

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