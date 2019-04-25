""" Script to train model"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.24'


import os
import functools

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import cv2

import dataset_configs
import data_io
import u_net

import tensorflow as tf
from tensorflow.keras import models, layers, losses

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_path = r'/awlab/users/chsu/WorkSpace/tensorflow/segmentation'

MODE = 'EVAL' # 'TRAIN' or 'EVAL'

# For loading previously trained model
model_name = 'incucyte_nuc_unweighted_bce_20190425_08-47-43'
model_dir = os.path.join(root_path, 'models', model_name)
result_folder = os.path.join(root_path, 'results', model_name)
n_test = 5


#### Set up
# data IO
task = 'incucyte_nucleus'
test_size = 0.2
random_state = 423

match_pattern = '_[A-Z]([4]|10)_'

nuc_idx = 1
cell_idx = 0

# dataset config
train_cfg = {
    'resize': None, 
    'scale': 1/255.,
    'crop_size': [512, 512],
    'to_flip': True
}
val_cfg = {
    'resize': None, 
    'scale': 1/255.,
    'crop_size': [512, 512]
}

# building model
num_filters_list = [32, 64, 128, 256, 512]
n_classes = 2
w_cfg = {
    'nuc_ch': 1,
    'cell_ch': 0,
    'w0': 10,
    'sigma': 5
}

loss_fn_type = 'weighted_bce'
metrics = [u_net.dice_loss]
monitor = 'val_dice_loss'

# training model
model_tag = 'incucyte_nuc_' + loss_fn_type + '_'
epochs = 50


###############################################################################################
### Get file names
data_cfg = dataset_configs.get_dataset_config(task)
data_cfg['match_pattern'] = match_pattern

x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
    data_io.get_data_filenames(**data_cfg, test_size=test_size, random_state=random_state)

num_train_data = len(x_train_fnames)
num_val_data = len(x_val_fnames)


### Configure training and validation dataset
read_img_fn = functools.partial(data_io._get_image_from_path, channels=data_cfg['n_channels'], 
                                dtype=data_cfg['dtype'], crop_bd_width=data_cfg['crop_bd_width'])

batch_size = data_cfg['batch_size']

tr_preproc_fn = functools.partial(data_io._augment, **train_cfg)
val_preproc_fn = functools.partial(data_io._augment, **val_cfg)

train_ds = data_io.get_dataset(x_train_fnames, y_train_fnames, read_img_fn=read_img_fn,
                               preproc_fn=tr_preproc_fn, shuffle=True, batch_size=batch_size)
val_ds = data_io.get_dataset(x_val_fnames, y_val_fnames, read_img_fn=read_img_fn, 
                             preproc_fn=val_preproc_fn, shuffle=False, batch_size=batch_size)


### Build the model
model = u_net.Unet(num_filters_list, n_classes=n_classes, dynamic=True)

# loss functions
if loss_fn_type is 'weighted_bce':
    loss_fn = functools.partial(u_net.weighted_bce_loss, w0=w_cfg['w0'], sigma=w_cfg['sigma'])

elif loss_fn_type is 'weighted_bce_dice':
    loss_fn = functools.partial(u_net.weighted_bce_dice_loss, w0=w_cfg['w0'], sigma=w_cfg['sigma'])

elif loss_fn_type is 'unweighted_bce':
    loss_fn = losses.binary_crossentropy

elif loss_fn_type is 'unweighted_bce_dice':
    loss_fn = u_net.bce_dice_loss

elif loss_fn_type is 'unweighted_dice':
    loss_fn = u_net.dice_loss
# loss_fn = functools.partial(u_net.weighted_cce_loss, **w_cfg)

model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)


if MODE == 'TRAIN':
    ### Train the model
    # callbacks
    timestamp = '{}'.format(datetime.datetime.now()).split('.')[0]
    timestamp = timestamp.replace('-','').replace(':','-').replace(' ','_')
    foler_name = model_tag + timestamp

    # model weights
    weights_path = os.path.join(root_path, 'models', foler_name, 'weights-{epoch:04d}.ckpt')
    weights_dir = os.path.dirname(weights_path)
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor=monitor, 
                                            save_best_only=True, save_weights_only=True, verbose=1)
    # tensorboard
    log_dir = os.path.join(root_path, 'logs', foler_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # training
    history = model.fit(train_ds, epochs=epochs, 
                        steps_per_epoch=int(np.ceil(num_train_data / batch_size)),
                        validation_data=val_ds,
                        validation_steps=int(np.ceil(num_val_data / batch_size)),
                        callbacks=[cp, tb])

if MODE == 'EVAL':
    # Load the trained model
    latest = tf.train.latest_checkpoint(model_dir)
    print()
    print('Loading model from:')
    print('  ', latest)
    model.load_weights(latest)

    performance = model.evaluate(val_ds, steps=int(np.ceil(num_val_data / batch_size)))
    print()
    print("Performance: {}".format(performance))


    # Export predicted images
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    idx_to_plot = np.random.choice(num_val_data, n_test)
    for i, (img, mask) in enumerate(val_ds):
        if i in idx_to_plot:
            y_pred = model(img)        
            for j in range(batch_size-1):
                I = np.uint8(img[j].numpy()*255.)
                M = np.uint8(mask[j].numpy()*255.) 
                M_pred = np.uint8((y_pred[j].numpy() > 0.5) *255.)
                
                if task == 'both_seg':
                    I = np.uint8(img[j].numpy()*255.)
                    M = np.uint8(mask[j].numpy()*255.) 
                    M_pred = np.uint8((y_pred[j].numpy() > 0.5) *255.)
                    
                    # overlay nucleus segmentation
                    I = data_io.overlay_mask(I, M[:,:,nuc_idx], M_pred[:,:,nuc_idx], 
                                     true_color=None, pred_color=(0,255,255))
                    # overlay cell segmentation
                    I = data_io.overlay_mask(I, M[:,:,cell_idx], M_pred[:,:,cell_idx], 
                                     true_color=None, pred_color=(255,0,255))
                else:
                    I = data_io.overlay_mask(I, M[:,:,0], M_pred[:,:,0])
                
                fname = os.path.join(result_folder, '{}_{}.png'.format(i,j))
                cv2.imwrite(fname, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
                
        if i > max(idx_to_plot):
            print("Save results at:")
            print('  ', result_folder)
            break
