## Description

This project implements a 3D-UNet to segment a downsampled Prostate 3D data set with the goal of having a minimum dice similarity coefficient of 0.7 when testing.

## Model Architecture

The model implemented is based on the 3D-UNet describes in [1] and has the following architecture:



## Dataset

## Usage

**Dependencies**

- **Python** 3.12.5
- **pytorch** 2.8.0
- **matplotlib** 3.10.6
- **numpy** 2.1.2
- **nibabel** 5.3.2
- **imageio** 2.37.0

### Training:

To create and train a model, running the following line in the 3DUNet-prostate-46410483 directory

```bash
python train.py
```

This will create, train, validate and test a model and save the resulting model into the /model directory

### Predict:

To use a previously created model to create visualations of the preformance of the model run the following line in the 3DUNet-prostate-46410483 directory

```bash
python predict.py
```

## Results

Below are two gifs of the same image with masks overlayed on them, the left gif uses the masks produced by the model and the right gif uses the ground truth masks provided by the dataset.

| Model | Ground Truth |
| :---: | :----------: |
| ![model](readme_images/model.gif) | ![ground_truth](readme_images/ground_truth.gif) |


## References
