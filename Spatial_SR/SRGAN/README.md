We adapt [SRGAN](https://github.com/tensorlayer/srgan) to perform 8x spatial super-resolution of climate simulation images from the ExtremeWeather dataset. <br/>

`model.py` defines the generator and discriminator networks used in the GAN. <br/>
`config_clim.py` contains hyperparamater settings such as batch size, learning rate, number of epochs, etc.<br/>
`train_clim.py` contains the data loading, training, and inference pipelines<br/>
`SRGAN.ipynb` is the top-level Colab file from which all other scripts are called. The cells can simply be run in order.<br/>
To train: `!python train_clim.py`<br/>
To evaluate: `!python train_clim.py --mode=evaluate`<br/>
To generate patches centered around extreme weather events (i.e bounding boxes): `!python train_clim.py --mode=generate`
