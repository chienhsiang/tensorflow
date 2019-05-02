""" Script to train model"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.24'


import os
import functools
import yaml
from argparse import ArgumentParser

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
Commond
"""


CONFIGS_FOLDER = 'configs'
COMMON_YAML = 'common.yaml'


class Task:
    def __init__(self, task_yaml):
        self.cfg = self.get_cfg_from_yaml(task_yaml)


    def get_yaml_path_from_name(self, fname):
        return os.path.join(CONFIGS_FOLDER, fname)


    def get_cfg_from_yaml(self, yaml_name):
        # get common paths
        with open(self.get_yaml_path_from_name(COMMON_YAML)) as f:
            common_cfg = yaml.safe_load(f)

        with open(self.get_yaml_path_from_name(yaml_name)) as f:
            cfg = yaml.safe_load(f) 

        # merge these two configs
        cfg = {**common_cfg, **cfg}

        cfg['img_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfoler'],
                                      cfg['train_data'],  cfg['img_subfolder'])
        cfg['img_dir'] = os.path.join(cfg['root_folder'], cfg['data_subfoler'],
                                      cfg['train_data'],  cfg['mask_subfolder'])
        
        # conversion string to number or function handle
        if 'read_cfg' in cfg:
            cfg['read_cfg']['scale'] = eval(cfg['read_cfg']['scale'])
        
        if 'metrics' in cfg:
            cfg['metrics'] = [eval(s) for s in cfg['metrics']]
            
        return cfg  


    # For getting tf datasets
    def get_train_val_dataset(self):
        cfg = self.cfg


        x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
            data_io.get_data_filenames(**cfg)

        num_train_data = len(x_train_fnames)
        num_val_data = len(x_val_fnames)


        ### Configure training and validation dataset
        read_img_fn = functools.partial(data_io._get_image_from_path, **cfg['read_cfg'])

        batch_size = cfg['batch_size']

        tr_preproc_fn = functools.partial(data_io._augment, **cfg['train_cfg'])
        val_preproc_fn = functools.partial(data_io._augment, **cfg['val_cfg'])

        train_ds = data_io.get_dataset(x_train_fnames, y_train_fnames, read_img_fn=read_img_fn,
                                       preproc_fn=tr_preproc_fn, shuffle=True, batch_size=batch_size)
        val_ds = data_io.get_dataset(x_val_fnames, y_val_fnames, read_img_fn=read_img_fn, 
                                     preproc_fn=val_preproc_fn, shuffle=False, batch_size=batch_size)

        return train_ds, val_ds, num_train_data, num_val_data, batch_size


    # Build the model
    def get_model(self):
        num_filters_list = self.cfg['num_filters_list']
        n_classes = self.cfg['n_classes']
        metrics = self.cfg['metrics']

        model = u_net.Unet(num_filters_list, n_classes=n_classes, dynamic=True)

        # loss functions
        loss_fn = self.get_loss_fn_from_name(self.cfg['loss_fn_name'])

        model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

        return model


    def get_loss_fn_from_name(self, loss_fn_name):
        w_cfg = self.cfg['w_cfg']
        fn_map = {
            'weighted_bce': functools.partial(u_net.weighted_bce_loss, 
                                              w0=w_cfg['w0'], sigma=w_cfg['sigma']),
            'weighted_bce_dice': functools.partial(u_net.weighted_bce_dice_loss, 
                                                   w0=w_cfg['w0'], sigma=w_cfg['sigma']),
            'unweighted_bce': losses.binary_crossentropy,
            'unweighted_bce_dice': u_net.bce_dice_loss,
            'unweighted_dice': u_net.dice_loss
        }

        return fn_map[loss_fn_name]


    def get_trained_model(self):
        cfg = self.cfg

        # Load the trained model
        model_dir = os.path.join(cfg['root_folder'], cfg['model_subfolder'], cfg['model_name'])
        latest = tf.train.latest_checkpoint(model_dir)
        print()
        print('Loading model from:')
        print('  ', latest)

        model = self.get_model()
        model.load_weights(latest)

        # performance = model.evaluate(val_ds, steps=int(np.ceil(num_val_data / batch_size)))
        # print()
        # print("Performance: {}".format(performance))

        return model


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("task_yaml", help="yaml file of the task")
    parser.add_argument("-mode", dest="mode", help="TRAIN, EVAL or TEST", default="TRAIN")
    parser.add_argument("-gpu_id", dest="gpu", help="ID of GPU to use", default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    MODE = args.mode

    task = Task(args.task_yaml)

    for k in task.cfg:
        print(k, ':', task.cfg[k])

    # if MODE == 'TRAIN':
    #     train_ds, val_ds, num_train_data, num_val_data, batch_size = get_train_val_dataset()
    #     model = get_model()

    #     # callbacks
    #     timestamp = '{}'.format(datetime.datetime.now()).split('.')[0]
    #     timestamp = timestamp.replace('-','').replace(':','-').replace(' ','_')
    #     foler_name = model_tag + timestamp

    #     # model weights
    #     weights_path = os.path.join(root_path, 'models', foler_name, 'weights-{epoch:04d}.ckpt')
    #     weights_dir = os.path.dirname(weights_path)
    #     if not os.path.isdir(weights_dir):
    #         os.makedirs(weights_dir, exist_ok=True)
    #     cp = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, monitor=monitor, 
    #                                             save_best_only=True, save_weights_only=True, verbose=1)
    #     # tensorboard
    #     log_dir = os.path.join(root_path, 'logs', foler_name)
    #     if not os.path.isdir(log_dir):
    #         os.makedirs(log_dir, exist_ok=True)
    #     tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    #     # training

    #     history = model.fit(train_ds, epochs=epochs, 
    #                         steps_per_epoch=int(np.ceil(num_train_data / batch_size)),
    #                         validation_data=val_ds,
    #                         validation_steps=int(np.ceil(num_val_data / batch_size)),
    #                         callbacks=[cp, tb])

    # if MODE == 'EVAL':
    #     train_ds, val_ds, num_train_data, num_val_data, batch_size = get_train_val_dataset()
    #     model = get_trained_model()

    #     # Export predicted images
    #     if not os.path.isdir(result_folder):
    #         os.makedirs(result_folder)

    #     idx_to_plot = np.random.choice(num_val_data, n_test)
    #     for i, (img, mask) in enumerate(val_ds):
    #         if i in idx_to_plot:
    #             y_pred = model(img)        
    #             for j in range(batch_size-1):
    #                 I = np.uint8(img[j].numpy()*255.)
    #                 M = np.uint8(mask[j].numpy()*255.) 
    #                 M_pred = np.uint8((y_pred[j].numpy() > 0.5) *255.)
                    
    #                 if task == 'both_seg':
    #                     I = np.uint8(img[j].numpy()*255.)
    #                     M = np.uint8(mask[j].numpy()*255.) 
    #                     M_pred = np.uint8((y_pred[j].numpy() > 0.5) *255.)
                        
    #                     # overlay nucleus segmentation
    #                     I = data_io.overlay_mask(I, M[:,:,nuc_idx], M_pred[:,:,nuc_idx], 
    #                                      true_color=None, pred_color=(0,255,255))
    #                     # overlay cell segmentation
    #                     I = data_io.overlay_mask(I, M[:,:,cell_idx], M_pred[:,:,cell_idx], 
    #                                      true_color=None, pred_color=(255,0,255))
    #                 else:
    #                     I = data_io.overlay_mask(I, M[:,:,0], M_pred[:,:,0])
                    
    #                 fname = os.path.join(result_folder, '{}_{}.png'.format(i,j))
    #                 cv2.imwrite(fname, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
                    
    #         if i > max(idx_to_plot):
    #             print("Save results at:")
    #             print('  ', result_folder)
    #             break

    # if MODE == 'TEST':
    #     # Get test dataset and break into chunks (memory issue)
    #     img_files = data_io.get_filenames(file_dir, file_type, filter_patter)
    #     img_files = [img_files[i:i+chunk_size] for i in range(0, len(img_files), chunk_size)]
    #     if ans_dir is not None:
    #         ans_files = data_io.get_filenames(ans_dir, file_type, filter_patter)
    #         ans_files = [ans_files[i:i+chunk_size] for i in range(0, len(ans_files), chunk_size)]

    #     read_img_fn = functools.partial(data_io._get_image_from_path, **test_read_cfg)

    #     # Get trained model
    #     model = get_trained_model()

    #     # Output the images
    #     if not os.path.isdir(result_folder):
    #         os.makedirs(result_folder)

    #     for i, g in enumerate(img_files):
    #         print()
    #         print("Predicting chunck {}/{}...".format(i+1, len(img_files)))
    #         test_ds = data_io.get_dataset(g, None, read_img_fn=read_img_fn,
    #                                       shuffle=False, repeat=False, batch_size=1)
    #         y_pred = model.predict(test_ds, verbose=1)

    #         if ans_dir is not None:
    #             ans_ds = data_io.get_dataset(ans_files[i], None, read_img_fn=read_img_fn,
    #                                          shuffle=False, repeat=False, batch_size=1)
    #         else:
    #             holder = [[] for i in range(len(g))]
    #             ans_ds = tf.data.Dataset.from_tensor_slices(holder)

    #         test_ds = tf.data.Dataset.zip((test_ds, ans_ds))

    #         for j, (x, y) in enumerate(test_ds):
    #             print("Saving results {}/{}...".format(j+1, len(g)), end='\r')
    #             I = np.uint8(x[0]*255.)
    #             M_pred = np.uint8((y_pred[j,...,0] > 0.5) * 255.)

    #             if ans_dir is None:
    #                 M = []
    #                 true_color = None
    #             else:
    #                 M = np.uint8((y[0].numpy() > 0.5) * 255.)
    #                 true_color = (0,255,0)

    #             I = data_io.overlay_mask(I, M, M_pred, true_color=true_color, pred_color=(255,0,0))
                
    #             fname = os.path.join(result_folder, os.path.basename(g[j]))
    #             cv2.imwrite(fname, cv2.cvtColor(I, cv2.COLOR_RGB2BGR))
    #         print()
    
