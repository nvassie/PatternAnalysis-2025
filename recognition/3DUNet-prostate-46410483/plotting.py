import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def epoch_plot(images, masks, outputs, dice_values):
        CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
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