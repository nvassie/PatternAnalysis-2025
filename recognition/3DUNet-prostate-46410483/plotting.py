import torch
import matplotlib.pyplot as plt

def epoch_plot(images, masks, outputs):
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

        plt.figure(figsize=(12,4))

        # image plot
        plt.subplot(1,3,1)
        plt.title(f"Image slice {mid_slice}")
        plt.imshow(image_slice, cmap="gray")
        plt.axis("off")

        # ground truth mask
        plt.subplot(1,3,2)
        plt.title("Ground Truth Mask")
        plt.imshow(image_slice, cmap="gray")
        plt.imshow(mask_slice, cmap="turbo", alpha=0.4, vmin=0, vmax=5)
        plt.axis("off")

        # model mask
        plt.subplot(1,3,3)
        plt.title("Model Mask")
        plt.imshow(image_slice, cmap="gray")
        plt.imshow(prediction_slice, cmap="turbo", alpha=0.4, vmin=0, vmax=5)
        plt.axis("off")

        plt.tight_layout()
        plt.show()