"""Data sets"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.05'

import os


root_folder = r'/awlab/users/chsu/WorkSpace/tensorflow/segmentation/data'
img_subfolder = 'images'
mask_subfolder = 'masks'


def get_dataset_config(task):
    cfg = None

    if task == 'both_seg':
        cfg = {
            'data_folder': 'p2017017086_ki67_merge',
            'img_file_pattern': '*.png',
            'mask_file_pattern': '*.png', # merged masks, green: nucleus, red: cell
            'remove_pattern': '_[A-Z]2_',

            'crop_bd_width': 100, # pixel width to remove from boundary

            'n_channels': 3,
            'dtype': 'uint8',
            'batch_size': 5
        }
        

    if task == 'nuc_seg':
        cfg = {
            'data_folder': 'plate_2017017086_ki67',
            'img_file_pattern': '*-2.png',
            'mask_file_pattern': '*_nucleus.png',
            'remove_pattern': '_[A-Z]2_',

            'crop_bd_width': 100, # pixel width to remove from boundary

            'n_channels': 1,
            'dtype': 'uint16',
            'batch_size': 5
        }
        

    if task == 'cell_seg':
        cfg = {
            'data_folder': 'plate_2017017086_ki67',
            'img_file_pattern': '*-3.png',
            'mask_file_pattern': '*_cell.png',
            'remove_pattern': '_[A-Z]2_',

            'crop_bd_width': 100, # pixel width to remove from boundary

            'n_channels': 1,
            'dtype': 'uint16',
            'batch_size': 5
        }
        
    elif cfg is None:
        raise ValueError('Unknown task.')

    cfg['img_dir'] = os.path.join(root_folder, cfg['data_folder'], img_subfolder)
    cfg['mask_dir'] = os.path.join(root_folder, cfg['data_folder'], mask_subfolder)

    return cfg


