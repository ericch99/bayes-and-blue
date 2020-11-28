#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import cv2
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D, get_D_Conditional
from config_clim import config
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import shutil


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
# create folders to save result images and trained models
now = datetime.now()
now = str(now).split('.')[0]

save_dir = now + "/samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = now + "/models"
tl.files.exists_or_mkdir(checkpoint_dir)

#make copies of training, config, and model scripts for version control	
shutil.copy('train_clim.py', now)	
shutil.copy('config_clim.py', now)	
shutil.copy('model.py', now)



def get_train_data():

    # load dataset
    
    train_path_1979_2 = config.TRAIN.hr_img_path + "2/CycleGAN_Data/ExtremeWeather/";
    train_path_1981_2 = "/content/drive/My Drive/ProjectX 2020/Data/1981/2/"
    train_path_1984_2 = "/content/drive/My Drive/ProjectX 2020/Data/1984/2/"
    
    train_path_1979_6 = config.TRAIN.hr_img_path + "6/";
    train_path_1981_6 = "/content/drive/My Drive/ProjectX 2020/Data/1981/6/";
    train_path_1984_6 = "/content/drive/My Drive/ProjectX 2020/Data/1984/6/";

    train_path_1979_8 = config.TRAIN.hr_img_path + "8/";
    train_path_1981_8 = "/content/drive/My Drive/ProjectX 2020/Data/1981/8/";
    train_path_1984_8 = "/content/drive/My Drive/ProjectX 2020/Data/1984/8/";

    print("Getting File Paths")
    hr_img_list_1979_2 = (tl.files.load_file_list(train_path_1979_2, regx='.*.npy', printable=False))[0:600]
    hr_img_list_1981_2 = (tl.files.load_file_list(train_path_1981_2, regx='.*.npy', printable=False))[0:600]
    hr_img_list_1984_2 = (tl.files.load_file_list(train_path_1984_2, regx='.*.npy', printable=False))[0:600]

    hr_img_list_1979_6 = (tl.files.load_file_list(train_path_1979_6, regx='.*.npy', printable=False))
    hr_img_list_1981_6 = (tl.files.load_file_list(train_path_1981_6, regx='.*.npy', printable=False))
    hr_img_list_1984_6 = (tl.files.load_file_list(train_path_1984_6, regx='.*.npy', printable=False))

    hr_img_list_1979_8 = (tl.files.load_file_list(train_path_1979_8, regx='.*.npy', printable=False))
    hr_img_list_1981_8 = (tl.files.load_file_list(train_path_1981_8, regx='.*.npy', printable=False))
    hr_img_list_1984_8 = (tl.files.load_file_list(train_path_1984_8, regx='.*.npy', printable=False))


    print("Loading Images")
    hr_imgs = [];
    min0= 999999999;
    min1= min0;
    min2= min0;

    max0 = -999999999;
    max1 = max0;
    max2 = max0;
    
    print("1979")
    for i in tqdm(range(len(hr_img_list_1979_2))):
      im = np.concatenate([np.load(train_path_1979_2 + hr_img_list_1979_2[i]),
                           np.load(train_path_1979_6 + hr_img_list_1979_6[i]),
                           np.load(train_path_1979_8 + hr_img_list_1979_8[i])], axis = 2);

      '''min0 = min(min0, np.min(im[:, :, 0]));
      min1 = min(min1, np.min(im[:, :, 1]));
      min2 = min(min2, np.min(im[:, :, 2]));
      max0 = max(max0, np.max(im[:, :, 0]));
      max1 = max(max1, np.max(im[:, :, 1]));
      max2 = max(max2, np.max(im[:, :, 2]));'''
      hr_imgs.append(im)
      del im;

    print("1981")
    for i in tqdm(range(len(hr_img_list_1981_2))):
      im = np.concatenate([np.load(train_path_1981_2 + hr_img_list_1981_2[i]),
                           np.load(train_path_1981_6 + hr_img_list_1981_6[i]),
                           np.load(train_path_1981_8 + hr_img_list_1981_8[i])], axis = 2);
      '''min0 = min(min0, np.min(im[:, :, 0]));
      min1 = min(min1, np.min(im[:, :, 1]));
      min2 = min(min2, np.min(im[:, :, 2]));
      max0 = max(max0, np.max(im[:, :, 0]));
      max1 = max(max1, np.max(im[:, :, 1]));
      max2 = max(max2, np.max(im[:, :, 2]));'''
      hr_imgs.append(im)
      del im;

    print("1984")
    for i in tqdm(range(len(hr_img_list_1984_2))):
      im = np.concatenate([np.load(train_path_1984_2 + hr_img_list_1984_2[i]),
                           np.load(train_path_1984_6 + hr_img_list_1984_6[i]),
                           np.load(train_path_1984_8 + hr_img_list_1984_8[i])], axis = 2);
      '''min0 = min(min0, np.min(im[:, :, 0]));
      min1 = min(min1, np.min(im[:, :, 1]));
      min2 = min(min2, np.min(im[:, :, 2]));
      max0 = max(max0, np.max(im[:, :, 0]));
      max1 = max(max1, np.max(im[:, :, 1]));
      max2 = max(max2, np.max(im[:, :, 2]));'''
      hr_imgs.append(im)
      del im;
    
    maxes = np.array([1.0867410e5, 9.5626381e1, 3.3686847e2]);
    mins = np.array([9.0560906e4, 3.2156777e-02, 1.5234143e2]);
    print(maxes)
    print(mins)

    train_hr_imgs = hr_imgs[0:1500];
    t = len(train_hr_imgs);
    print("Number of Training Images: ", t)

    valid_lr_imgs = hr_imgs[1500:];
    v = len(valid_lr_imgs)
    print("Number of Validation Images: ", v)
    
        
    # dataset API and augmentation
    def generator_train():
        for im in train_hr_imgs:
            yield im;
    def generator_valid():
        for im in valid_lr_imgs:
            yield im;
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [224, 224, 3])
        hr_patch  = tf.divide(tf.subtract(hr_patch, mins), tf.cast(tf.subtract(maxes, mins), tf.float32));         #min-max normalization
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        hr_patch = tf.image.random_flip_up_down(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[28, 28])
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)

    valid_ds = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32))
    valid_ds = valid_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    #valid_ds = valid_ds.shuffle(shuffle_buffer_size)
    valid_ds = valid_ds.prefetch(buffer_size=2)
    valid_ds = valid_ds.batch(batch_size)


    return train_ds, t , valid_ds, v, maxes, mins

def train(end):
    G = get_G((batch_size, 28, 28, 3))
    D = get_D_Conditional((batch_size, 224, 224, 3))
    VGG = tl.models.VGG16(pretrained = True, end_with = end)
    
    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()
    train_ds, num_images, valid_ds, num_valid_images, maxes, mins = get_train_data()

    ## initialize learning (G)
    print("Initialize Generator")

    mseloss = [];
    PSNR = [];
    SSIM = [];
    mseloss_valid = [];
    PSNR_valid = [];
    SSIM_valid = [];
    
    n_step_epoch = round(num_images // batch_size)
    for epoch in range(n_epoch_init):
        mse_avg = 0;
        PSNR_avg = 0;
        SSIM_avg = 0;
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
          
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
                mse_avg+=mse_loss;
                
                psnr = np.mean(np.array(tf.image.psnr(fake_hr_patchs, hr_patchs, max_val = 1)));
                PSNR_avg += psnr;
                
                ssim = np.mean(np.array(tf.image.ssim(fake_hr_patchs, hr_patchs, max_val = 1)));
                SSIM_avg+=ssim;

                

            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))

            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f}, psnr: {:.3f}, ssim: {:.3f} ".format(
                epoch+1, n_epoch_init, step+1, n_step_epoch, time.time() - step_time, mse_loss, psnr, ssim))
        if (epoch != 0) and (epoch % 10 == 0):
            #restore to original values before normalization
            '''fake_hr_patchs = fake_hr_patchs.numpy()
            hr_patchs = hr_patchs.numpy()
            fake_hr_patchs = np.add(np.multiply(fake_hr_patchs, np.subtract(maxes, mins)), mins);
            hr_patchs = np.add(np.multiply(hr_patchs, np.subtract(maxes, mins)), mins)'''


            save = np.concatenate([fake_hr_patchs, hr_patchs], axis = 0)
            tl.vis.save_images(save, [4, 4], os.path.join(save_dir, 'train_g_init{}.png'.format(epoch)))
            
        mseloss.append(mse_avg/(step+1));
        PSNR.append(PSNR_avg/(step+1));
        SSIM.append(SSIM_avg/(step+1));

        #validate
        mse_valid_loss = 0;
        psnr_valid = 0;
        ssim_valid = 0;
        for step, (lr_patchs, hr_patchs) in enumerate(valid_ds):
          fake_hr_patchs = G(lr_patchs)
          mse_valid_loss += tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True);
          psnr_valid += np.mean(np.array(tf.image.psnr(fake_hr_patchs, hr_patchs, max_val = 1)));
          ssim_valid += np.mean(np.array(tf.image.ssim(fake_hr_patchs, hr_patchs, max_val = 1)));

        if (epoch != 0) and (epoch % 10 == 0):
          #restore to original values before normalization
          '''fake_hr_patchs = fake_hr_patchs.numpy()
          hr_patchs = hr_patchs.numpy()
          fake_hr_patchs = np.add(np.multiply(fake_hr_patchs, np.subtract(maxes, mins)), mins);
          hr_patchs = np.add(np.multiply(hr_patchs, np.subtract(maxes, mins)), mins);'''
          save = np.concatenate([fake_hr_patchs, hr_patchs], axis = 0)
          tl.vis.save_images(save, [4, 4], os.path.join(save_dir, 'valid_g_init{}.png'.format(epoch)))
        
        mse_valid_loss /= (step+1);
        mseloss_valid.append(mse_valid_loss)
        psnr_valid /= (step+1);
        PSNR_valid.append(psnr_valid)
        ssim_valid /= (step+1)
        SSIM_valid.append(ssim_valid);


        print("Validation MSE: ", mse_valid_loss.numpy(), "Validation PSNR: ", psnr_valid, "Validation SSIM: ", ssim_valid)
        

    '''plot stuff'''
    '''plt.figure()
    epochs = np.linspace(1, n_epoch_init*n_step_epoch, num = n_epoch_init*n_step_epoch);
    plt.title("Generator Initialization")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.plot(epochs, np.array(mseloss))
    plt.plot(epochs, np.array(PSNR))
    #plt.plot(epochs, SSIM)
    plt.legend(("MSE", " PSNR"))
    plt.show()'''
    np.save(save_dir +'/mse_init_train.npy', mseloss)
    np.save(save_dir +'/psnr_init_train.npy', PSNR)
    np.save(save_dir +'/ssim_init_train.npy', SSIM)
    np.save(save_dir +'/mse_init_valid.npy', mseloss_valid)
    np.save(save_dir +'/psnr_init_valid.npy', PSNR_valid)
    np.save(save_dir +'/ssim_init_valid.npy', SSIM_valid)


    ## adversarial learning (G, D)
    print("Adversarial Learning")

    mseloss = [];
    mseloss_valid = [];
    PSNR = [];
    PSNR_valid = [];
    SSIM = [];
    SSIM_valid = []
    vggloss = [];
    vggloss_valid = [];
    advloss = [];
    dloss = [];
    min_val_mse = 9999;
    max_val_psnr = 0;
    max_val_ssim = 0;
    n_step_epoch = round(num_images // batch_size)
    for epoch in range(n_epoch):
        mse_avg = 0;
        PSNR_avg = 0;
        vgg_avg = 0;
        SSIM_avg = 0;
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D([lr_patchs, fake_patchs])
                logits_real = D([lr_patchs, hr_patchs])
      
                d_acc = (np.count_nonzero(tf.nn.sigmoid(logits_fake) < 0.5) + np.count_nonzero(tf.nn.sigmoid(logits_real) > 0.5))/16;
                feature_fake = VGG(fake_patchs) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG(hr_patchs)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 6e-5 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                vgg_avg +=vgg_loss;
                g_loss = mse_loss + vgg_loss + g_gan_loss

                mse_avg+=mse_loss;
                advloss.append(g_gan_loss);
                dloss.append(d_loss);
                psnr = np.mean(np.array(tf.image.psnr(fake_patchs, hr_patchs, max_val = 1)));
                PSNR_avg+=psnr;
                ssim = np.mean(np.array(tf.image.ssim(fake_patchs, hr_patchs, max_val = 1)));
                SSIM_avg+=ssim;
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}, d_acc: {:.3f}, psnr: {:.3f}, ssim: {:.3f}".format(
                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss, d_acc, psnr, ssim))


        if (epoch != 0) and (epoch % 10 == 0):
            #restore to original values before normalization
            '''fake_patchs = fake_patchs.numpy()
            hr_patchs = hr_patchs.numpy()
            fake_patchs = np.add(np.multiply(fake_patchs, np.subtract(maxes - mins)), mins);
            hr_patchs = np.add(np.multiply(hr_patchs, np.subtract(maxes - mins)), mins);'''
            save = np.concatenate([fake_patchs, hr_patchs], axis = 0)
            tl.vis.save_images(save, [4, 4], os.path.join(save_dir, 'train_g{}.png'.format(epoch)))
        
        mseloss.append(mse_avg/(step+1));
        PSNR.append(PSNR_avg/(step+1));
        vggloss.append(vgg_avg/(step+1));
        SSIM.append(SSIM_avg/(step+1));

        #validate
        mse_valid_loss = 0;
        vgg_valid_loss = 0;
        psnr_valid = 0;
        ssim_valid = 0;
        d_acc = 0;
        
        for step, (lr_patchs, hr_patchs) in enumerate(valid_ds):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
              break
            fake_patchs = G(lr_patchs)
            logits_fake = D([lr_patchs, fake_patchs])
            logits_real = D([lr_patchs, hr_patchs])
            d_acc += ((np.count_nonzero(tf.nn.sigmoid(logits_fake) < 0.5) + np.count_nonzero(tf.nn.sigmoid(logits_real) > 0.5)))
            feature_fake = VGG(fake_patchs) # the pre-trained VGG uses the input range of [0, 1]
            feature_real = VGG(hr_patchs)
            d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
            d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
            d_loss = d_loss1 + d_loss2
            g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
            mse_valid_loss += tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
            vgg_valid_loss += 6e-5 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
            
            g_loss = mse_loss + vgg_valid_loss + g_gan_loss
            psnr_valid += np.mean(np.array(tf.image.psnr(fake_patchs, hr_patchs, max_val = 1)));
            ssim_valid += np.mean(np.array(tf.image.ssim(fake_patchs, hr_patchs, max_val = 1)));
                

        mse_valid_loss /= (step+1);
        mseloss_valid.append(mse_valid_loss)
        vgg_valid_loss /= (step+1);
        vggloss_valid.append(vgg_valid_loss)
        psnr_valid /= (step+1);
        PSNR_valid.append(psnr_valid);
        ssim_valid /= (step+1);
        SSIM_valid.append(ssim_valid);

        
        d_acc /= (num_valid_images*2)
        print("Validation MSE: ", mse_valid_loss.numpy(), "Validation PSNR: ", psnr_valid, "Validation SSIM", ssim_valid, "Validation Disc Accuracy: ", d_acc)
        
        if (epoch != 0) and (epoch % 10 == 0):
          #restore to original values before normalization
            '''fake_patchs = fake_patchs.numpy()
            hr_patchs = hr_patchs.numpy()
            fake_patchs = np.add(np.multiply(fake_patchs, np.subtract(maxes, mins)), mins);
            hr_patchs = np.add(np.multiply(hr_patchs, np.subtract(maxes, mins)), mins);'''
            save = np.concatenate([fake_patchs, hr_patchs], axis = 0)
            tl.vis.save_images(save, [4, 4], os.path.join(save_dir, 'valid_g{}.png'.format(epoch)))


        #save models if metrics improve
        if(mse_valid_loss <= min_val_mse):
          print("Val loss improved from ", np.array(min_val_mse), " to " , np.array(mse_valid_loss))
          min_val_mse = mse_valid_loss;
          G.save_weights(os.path.join(checkpoint_dir, "g_mse_val.h5"))
          D.save_weights(os.path.join(checkpoint_dir, "d_mse_val.h5")) 
        else:
          print ("Val loss did not improve from ", np.array(min_val_mse))

        if(psnr_valid >= max_val_psnr):
          print("Val PSNR improved from ", max_val_psnr, " to " , psnr_valid)
          max_val_psnr = psnr_valid;
          G.save_weights(os.path.join(checkpoint_dir, "g_psnr_val.h5"))
          D.save_weights(os.path.join(checkpoint_dir, "d_psnr_val.h5"))
        else:
          print ("Val PSNR did not improve from ", max_val_psnr)

        if(ssim_valid >= max_val_ssim):
          print("Val SSIM improved from ", max_val_ssim, " to " , ssim_valid)
          max_val_ssim = ssim_valid;
          G.save_weights(os.path.join(checkpoint_dir, "g_ssim_val.h5"))
          D.save_weights(os.path.join(checkpoint_dir, "d_ssim_val.h5"))
        else:
          print ("Val SSIM did not improve from ", max_val_ssim)


        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
    
    np.save(save_dir + '/mse_train.npy', mseloss)
    np.save(save_dir + '/psnr_train.npy', PSNR)
    np.save(save_dir + '/ssim_train.npy', SSIM)
    np.save(save_dir + '/advloss_train.npy', advloss)
    np.save(save_dir + '/discloss_train.npy', dloss)
    np.save(save_dir + '/vgg_train.npy', vggloss)
    np.save(save_dir + '/mse_valid.npy', mseloss_valid)
    np.save(save_dir + '/psnr_valid.npy', PSNR_valid)
    np.save(save_dir + '/ssim_valid.npy', SSIM_valid)
    np.save(save_dir + '/vgg_valid.npy', vggloss_valid)

    return checkpoint_dir, save_dir

def evaluate(timestamp, model):
    
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # load dataset
    
    # load dataset
    
    train_path_1979_2 = config.TRAIN.hr_img_path + "2/CycleGAN_Data/ExtremeWeather/";
    train_path_1981_2 = "/content/drive/My Drive/ProjectX 2020/Data/1981/2/"
    train_path_1984_2 = "/content/drive/My Drive/ProjectX 2020/Data/1984/2/"
    
    train_path_1979_6 = config.TRAIN.hr_img_path + "6/";
    train_path_1981_6 = "/content/drive/My Drive/ProjectX 2020/Data/1981/6/";
    train_path_1984_6 = "/content/drive/My Drive/ProjectX 2020/Data/1984/6/";

    train_path_1979_8 = config.TRAIN.hr_img_path + "8/";
    train_path_1981_8 = "/content/drive/My Drive/ProjectX 2020/Data/1981/8/";
    train_path_1984_8 = "/content/drive/My Drive/ProjectX 2020/Data/1984/8/";

    print("Getting File Paths")
    #hr_img_list_1979_2 = (tl.files.load_file_list(train_path_1979_2, regx='.*.npy', printable=False))[0:500]
    #hr_img_list_1981_2 = (tl.files.load_file_list(train_path_1981_2, regx='.*.npy', printable=False))[0:500]
    hr_img_list_1984_2 = (tl.files.load_file_list(train_path_1984_2, regx='.*.npy', printable=False))[200:500]

    #hr_img_list_1979_6 = (tl.files.load_file_list(train_path_1979_6, regx='.*.npy', printable=False))
    #hr_img_list_1981_6 = (tl.files.load_file_list(train_path_1981_6, regx='.*.npy', printable=False))
    hr_img_list_1984_6 = (tl.files.load_file_list(train_path_1984_6, regx='.*.npy', printable=False))

    #hr_img_list_1979_8 = (tl.files.load_file_list(train_path_1979_8, regx='.*.npy', printable=False))
    #hr_img_list_1981_8 = (tl.files.load_file_list(train_path_1981_8, regx='.*.npy', printable=False))
    hr_img_list_1984_8 = (tl.files.load_file_list(train_path_1984_8, regx='.*.npy', printable=False))

    print("Loading Images")
    hr_imgs = [];
    min0= 999999999;
    min1= min0;
    min2= min0;

    max0 = -999999999;
    max1 = max0;
    max2 = max0;


    print("1984")
    for i in tqdm(range(len(hr_img_list_1984_2))):
      im = np.concatenate([np.load(train_path_1984_2 + hr_img_list_1984_2[i]),
                           np.load(train_path_1984_6 + hr_img_list_1984_6[i]),
                           np.load(train_path_1984_8 + hr_img_list_1984_8[i])], axis = 2);
      min0 = min(min0, np.min(im[:, :, 0]));
      min1 = min(min1, np.min(im[:, :, 1]));
      min2 = min(min2, np.min(im[:, :, 2]));
      max0 = max(max0, np.max(im[:, :, 0]));
      max1 = max(max1, np.max(im[:, :, 1]));
      max2 = max(max2, np.max(im[:, :, 2]));
      hr_imgs.append(im)
    
    maxes = np.array([max0, max1, max2]);
    mins = np.array([min0, min1, min2]);
    print(maxes)
    print(mins)


    valid_imgs = hr_imgs[0:];
    v = len(valid_imgs)
    print("Number of Validation Images: ", v)

    def generator_valid():
        for im in valid_imgs:
            yield im;
    def _map_fn_train(img):
        #hr_patch = tf.image.random_crop(img, [224, 224, 3])
        hr_patch = img;
        hr_patch  = tf.divide(tf.subtract(hr_patch, mins), tf.cast(tf.subtract(maxes, mins), tf.float32));         #min-max normalization
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        hr_patch = tf.image.random_flip_up_down(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 144])
        return lr_patch, hr_patch

    valid_ds = tf.data.Dataset.from_generator(generator_valid, output_types=(tf.float32))
    valid_ds = valid_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    valid_ds = valid_ds.prefetch(buffer_size=2)
    valid_ds = valid_ds.batch(1)
  
    ###========================== DEFINE MODEL ============================###
    G = get_G([1, 96, 144, 3])
    G.load_weights(os.path.join(timestamp, "models/g_" + model + ".h5"));
    G.eval()
    PSNR = [];
    #SSIM = [];
    MSE = [];
    pred_psnr = 0;
    pred_ssim = 0;
    pred_vgg = 0;
    bl_psnr = 0;
    bl_ssim = 0;
    bl_vgg = 0;


    '''imid = 10;
    h, w = valid_imgs[imid].shape[0], valid_imgs[imid].shape[1]
    valid_img  = tf.divide(tf.subtract(valid_imgs[imid], mins[0]), tf.cast(tf.subtract(maxes[0], mins[0]), tf.float32));         #min-max normalization
    valid_lr_img = tf.image.resize(valid_img, [int(h/8), int(w/8)]);
    valid_sr_img = G(tf.expand_dims(valid_lr_img, 0))
    valid_bicu_img = np.expand_dims(cv2.resize(np.float32(valid_lr_img), (w, h), interpolation= cv2.INTER_CUBIC), 2)

    
    print("LR size: %s /  generated HR size: %s" % ((h/8, w/8), valid_sr_img.shape))
      
    pred_psnr += np.mean(np.array(tf.image.psnr(valid_sr_img, valid_img, max_val = 1)))
    pred_ssim += np.mean(np.array(tf.image.ssim(valid_sr_img, valid_img, max_val = 1)))
    bl_psnr += np.mean(np.array(tf.image.psnr(valid_bicu_img, valid_img, max_val = 1)))
    bl_ssim += np.mean(np.array(tf.image.ssim(tf.convert_to_tensor(valid_bicu_img), valid_img, max_val = 1)))

    print(valid_lr_img.shape)
    print(valid_bicu_img.shape)
    print(valid_img.shape)
    print(valid_sr_img.shape)

    valid_lr_img = valid_lr_img[0:29, 0:29]
    valid_bicu_img = valid_bicu_img[0:225, 0:225]
    valid_img = valid_img[0:225, 0:225]
    valid_sr_img = valid_sr_img[0, 0:225, 0:225, :]
    #tl.vis.save_image(valid_sr_img[0].numpy(), os.path.join(timestamp, 'samples/valid_gen' + str(imid) + '.png'))
    tl.vis.save_image(valid_lr_img.numpy(), os.path.join(timestamp, 'samples/8x_LR.png'))
    tl.vis.save_image(valid_bicu_img, os.path.join(timestamp, 'samples/8x_bicubic.png'))
    tl.vis.save_image(valid_img.numpy(), os.path.join(timestamp, 'samples/GT.png'))
    tl.vis.save_image(valid_sr_img.numpy(), os.path.join(timestamp, 'samples/8x_HL.png'))'''
    VGG = tl.models.VGG16(pretrained = True, end_with = 'conv2_1')
    VGG.train()

    for step, (lr_patchs, hr_patchs) in enumerate(valid_ds):
      print(step, " out of ", len(valid_imgs))
      h, w = hr_patchs.shape[1], hr_patchs.shape[2]
      #print(w, h)
      valid_sr_img = G(lr_patchs)
      #print(lr_patchs.shape)
      valid_bicu_img = cv2.resize(np.float32(lr_patchs[0]), (w, h), interpolation= cv2.INTER_CUBIC)

    
      #print("LR size: %s /  generated HR size: %s" % ((h/8, w/8), valid_sr_img.shape))
      
      pred_psnr += np.mean(np.array(tf.image.psnr(valid_sr_img, hr_patchs, max_val = 1)))
      pred_ssim += np.mean(np.array(tf.image.ssim(valid_sr_img, hr_patchs, max_val = 1)))
      pred_vgg +=  tl.cost.mean_squared_error(VGG(hr_patchs), VGG(valid_sr_img), is_mean=True)

      bl_psnr += np.mean(np.array(tf.image.psnr(valid_bicu_img, hr_patchs, max_val = 1)))
      bl_ssim += np.mean(np.array(tf.image.ssim(tf.convert_to_tensor(valid_bicu_img), hr_patchs, max_val = 1)))
      bl_vgg +=  tl.cost.mean_squared_error(VGG(hr_patchs), VGG(valid_bicu_img), is_mean=True)


      triplet = np.concatenate([valid_bicu_img, np.zeros((768, 1, 3)), valid_sr_img[0].numpy(), np.zeros((768, 1, 3)), hr_patchs[0].numpy()], axis = 1)
      #tl.vis.save_image(valid_sr_img[0].numpy(), os.path.join(timestamp, 'samples/valid_gen' + str(imid) + '.png'))
      #tl.vis.save_image(valid_lr_img.numpy(), os.path.join(timestamp, 'samples/valid_lr' + str(imid) + '.png'))
      #tl.vis.save_image(valid_img.numpy(), os.path.join(timestamp, 'samples/valid_hr' + str(imid) + '.png'))
      tl.vis.save_image(triplet, os.path.join(timestamp, 'samples/triplet' + str(step) + '.png'))
    print (pred_psnr/v)
    print (pred_ssim/v)
    print (pred_vgg/v)
    print (bl_psnr/v)
    print (bl_ssim/v)
    print(bl_vgg/v)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
      for ending in ['conv2_1']:	
        checkpoint_dir, save_dir = train(ending)
        
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate("2020-11-20 00:17:57", "psnr_val")
    else:
        raise Exception("Unknown --mode")