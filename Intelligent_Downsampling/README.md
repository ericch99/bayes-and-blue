# Intelligent Downsampling

## Purpose
We use [CycleGAN](https://arxiv.org/abs/1703.10593) to learn a realistic degradation model from high-resolution images (ExtremeWeather dataset) to low-resolution images (NCEP dataset) by treating it as an unpaired image-to-image style transfer task. We demonstrate that this offers an improvement over "naive" downsampling methods such as bilinear or bicubic interpolation.

## Remarks
The notebook `downsampling.ipynb` contains examples of how to run the training and test code. The notebook, hosted on Google Colab, contains code to connect to the Google Drive directory and install the missing dependencies. 

The `checkpoints` folder contains the weights of the trained networks, and sample images generated during training. 

The `results` folder contains the final generated low resolution images.
