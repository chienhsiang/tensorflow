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


###############################################################################################
"""
Parameters
"""
###############################################################################################

"""
Common
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
root_path = r'/awlab/users/chsu/WorkSpace/tensorflow/segmentation'

MODE = 'TEST' # 'TRAIN', 'EVAL' or 'TEST'


"""-------------------------------------------------------------------------------------------------
For EVAL and TEST
"""
# For loading previously trained model using validation set
model_name = 'incucyte_nuc_weighted_bce_dice_20190424_16-09-32'
model_dir = os.path.join(root_path, 'models', model_name)
result_folder = os.path.join(root_path, 'results', model_name + '_test')

# for EVAL
n_test = 5

# for TEST
file_dir = os.path.join(root_path,'data','2019028023_PC9_A549_with_nuclear_marker','images')
file_type = '*.png'
filter_patter = None
test_read_cfg = {
    'channels': 1,
    'dtype': 'uint8', 
    'crop_bd_width': 0,
    'resize': [1024, 1408],
    'scale': 1/255.
}


"""-------------------------------------------------------------------------------------------------
For TRAIN
"""
# data IO
task = 'incucyte_nucleus'
test_size = 0.2
random_state = 423

match_pattern = '_[A-Z]([4]|10)_'

nuc_idx = 1
cell_idx = 0

# dataset config
resize = None
scale = 1/255.

train_cfg = {
    'crop_size': [512, 512],
    'to_flip': True
}

val_cfg = {
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

loss_fn_type = 'unweighted_bce_dice'
metrics = [u_net.dice_loss]
monitor = 'val_dice_loss'

# training model
model_tag = 'incucyte_nuc_' + loss_fn_type + '_'
epochs = 50


###############################################################################################
### Get file names
def get_train_val_dataset():
    data_cfg = dataset_configs.get_dataset_config(task)
    data_cfg['match_pattern'] = match_pattern

    x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
        data_io.get_data_filenames(**data_cfg, test_size=test_size, random_state=random_state)

    num_train_data = len(x_train_fnames)
    num_val_data = len(x_val_fnames)


    ### Configure training and validation dataset
    read_cfg = {
        'channels': data_cfg['n_channels'],
        'dtype': data_cfg['dtype'], 
        'crop_bd_width': data_cfg['crop_bd_width'],
        'resize': resize,
        'scale': scale
    }
    read_img_fn = functools.partial(data_io._get_image_from_path, **read_cfg)

    batch_size = data_cfg['batch_size']

    tr_preproc_fn = functools.partial(data_io._augment, **train_cfg)
    val_preproc_fn = functools.partial(data_io._augment, **val_cfg)

    train_ds = data_io.get_dataset(x_train_fnames, y_train_fnames, read_img_fn=read_img_fn,
                                   preproc_fn=tr_preproc_fn, shuffle=True, batch_size=batch_size)
    val_ds = data_io.get_dataset(x_val_fnames, y_val_fnames, read_img_fn=read_img_fn, 
                                 preproc_fn=val_preproc_fn, shuffle=False, batch_size=batch_size)

    return train_ds, val_ds, num_train_data, num_val_data, batch_size


### Build the model
def get_model():
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

    model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

    return model


def get_trained_model():
    # Load the trained model
    latest = tf.train.latest_checkpoint(model_dir)
    print()
    print('Loading model from:')
    print('  ', latest)

    model = get_model()
    model.load_weights(latest)

    # performance = model.evaluate(val_ds, steps=int(np.ceil(num_val_data / batch_size)))
    # print()
    # print("Performance: {}".format(performance))

    return model


if __name__ == '__main__':

    if MODE == 'TRAIN':
        train_ds, val_ds, num_train_data, num_val_data, batch_size = get_train_val_dataset()
        model = get_model()

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
        train_ds, val_ds, num_train_data, num_val_data, batch_size = get_train_val_dataset()
        model = get_trained_model()

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

    if MODE == 'TEST':
        # Get test dataset
        img_files = data_io.get_filenames(file_dir, file_type, filter_patter)
        read_img_fn = functools.partial(data_io._get_image_from_path, **test_read_cfg)
        test_ds = data_io.get_dataset(img_files, None, read_img_fn=read_img_fn,
                                      shuffle=False, repeat=False, batch_size=1)

        # Get trained model
        model = get_trained_model()
        y_pred = model.predict(test_ds, verbose=1)

        # Output the images
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        for i, x in enumerate(test_ds):
            print("{}/{}".format(i+1, len(img_files)), end='\r')
            I = np.uint8(x[0]*255.)
            M_pred = np.uint8((y_pred[i,...,0] > 0.5) * 255.)
            I = data_io.overlay_mask(I, [], M_pred, true_color=None, pred_color=(255,0,0))
            
            fname = os.path.join(result_folder, os.path.basename(img_files[i]))
            cv2.imwrite(fname, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
    
