# Spatial Super-Resolution

## Purpose
We adapt [SRGAN](https://github.com/tensorlayer/srgan) to perform 8x spatial super-resolution of climate simulation images from the ExtremeWeather dataset. Depending on the resolution of the output images, such simulations can be computationally intensive; we demonstrate that learnable super-resolution methods can be used to achieve the desired spatial resolution and accuracy while still allowing the simulation to output images of lower resolution, thus reducing the resources needed to run the simulations.  

## Scripts
`model.py` defines the generator and discriminator networks used in the GAN.

`config_clim.py` contains hyperparamater settings such as batch size, learning rate, number of epochs, etc.

`train_clim.py` contains the data loading, training, and inference pipelines.

`SRGAN.ipynb` is the top-level Colab file from which all other scripts are called. The cells can simply be run in order.

## Running the Code
To train: `!python train_clim.py`<br/>
To evaluate: `!python train_clim.py --mode=evaluate`<br/>
To generate patches centered around extreme weather events (i.e bounding boxes): `!python train_clim.py --mode=generate`
