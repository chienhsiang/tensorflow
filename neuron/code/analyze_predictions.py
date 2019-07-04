""" Analyze prediction results to get total length and tip numbers. 


Chien-Hsiang Hsu, 2019.07.01
"""


CONFIGS_FOLDER = 'configs'
COMMON_YAML = 'common.yaml'

# Data frame 
COL_ORDER = ['file_name', 'mean_intensity', 'area_true', 'n_endpoints_true', 
             'dice_loss', 'area_pred', 'n_endpoints_pred']
TO_TAKE = ['file_name', 'dice_loss', 'area_pred', 'n_endpoints_pred']

PROB_THRESHOLD = 0.5
N_WORKERS = 200

FILE_TYPE = '*.png'
FILTER_PATTERN = None

DIL_TYPE = 1 # dilation type, 1: 4-connectivity, 2: 8-connectivity
MIN_AREA = 10 # minimal object area


#--------------------------------------------------------------------------------------
import sys, os, re
import glob
import yaml

import numpy as np
import pandas as pd
from functools import reduce

import skimage
from skimage.transform import resize
from skimage.morphology import skeletonize
from skimage.morphology import binary_dilation
from skimage.morphology import reconstruction
from skimage.morphology import square
from skimage.morphology import remove_small_objects
import skimage.filters.rank as rank

from multiprocessing import Pool
import time


def get_common_configs():
    yaml_path = os.path.join(CONFIGS_FOLDER, COMMON_YAML)
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join('..', CONFIGS_FOLDER, COMMON_YAML)
    with open(yaml_path) as f:
        common_cfg = yaml.safe_load(f)
    return common_cfg


def get_filenames(file_dir, file_type, filter_pattern):
    """ Get file names 
        file_type: e.g. '*.png'
        filter_pattern: regular expression of file name to keep

    Return:
        list of file full paths.
    """    
    fnames = sorted(glob.glob(os.path.join(file_dir, file_type))) 

    if filter_pattern is not None:
        pattern = re.compile(filter_pattern)
        fnames = [f for f in fnames if pattern.search(os.path.basename(f))]       
    
    return fnames


def find_endpoints(img):
    """Identidy endpoints.
    
    Inputs:
    ---------
    img: a boolean mask.
    
    Outputs
    ---------
    A boolean mask.
    """
    assert img.dtype == np.bool, "img must be boolean." 
    
    n_neighbors = rank.sum(img.astype(np.uint8), square(3))
    return (n_neighbors==2) & img


def find_junctions(img):
    """Identify junctions.
    
    Inputs:
    ---------
    img: a boolean mask.
    
    Outputs
    ---------
    A boolean mask.
    """
    assert img.dtype == np.bool, "img must be boolean."
    
    n_neighbors = rank.sum(img.astype(np.uint8), square(3))
    return (n_neighbors>3) & img


def clean_mask(img):
    """
    1. Remove small objects.
    2. Dilation the images.
    3. Skeletonize.
    
    Inputs:
    ---------
    img: a boolean mask.
    
    Outputs
    ---------
    A boolean mask.
    """
    assert img.dtype == np.bool, "img must be boolean."
    
    # 1. Remove small objects.
    img = remove_small_objects(img, min_size=MIN_AREA, connectivity=2)

    # 2. Dilation the images.
    if DIL_TYPE == 1:
        selem = None

    elif DIL_TYPE == 2:
        selem = square(3)

    else:
        raise ValueError('DIL_TYPE can only be 1 or 2.')

    img = binary_dilation(img, selem=selem)
    
    # 3. Skeletonize.
    sk_img = skeletonize(img)

    return sk_img

    # j_img = find_junctions(sk_img)
    # return reconstruction(j_img, sk_img) > 0


def get_img_mean_intensity(file_name):
    img = skimage.io.imread(file_name)
    return img[img>0].mean()


def dice_coeff(y_true, y_pred):
    intersect = np.sum((y_true > 0) & (y_pred > 0))
    d = 2. * intersect / (y_true.sum() + y_pred.sum())
    return d


def dice_loss(y_true, y_pred):
    return 1- dice_coeff(y_true, y_pred)


def get_dice_loss(path_pair):
    """path_pair: (y_true_path, y_pred_path)"""
    y_pred = skimage.io.imread(path_pair[1]) / 255.
    y_pred = np.float64(y_pred > PROB_THRESHOLD)
    
    y_true = skimage.io.imread(path_pair[0]) / 255.
    if y_true.shape != y_pred.shape:
        y_true = resize(y_true, (y_pred.shape[1], y_pred.shape[0]))
    
    return dice_loss(y_true, y_pred)


def get_endpoint_numbers(mask):
    return find_endpoints(mask).sum()


def get_results(path_pair):
    """path_pair: (y_true_path, y_pred_path)"""
    y_true_path = path_pair[0]
    y_pred_path = path_pair[1]

    fname = os.path.basename(y_pred_path)

    # Get path to the input image
    cfg = get_common_configs()
    dataset_dir = os.path.dirname(os.path.dirname(y_true_path))
    img_file = os.path.join(dataset_dir, cfg['img_subfolder'], fname)

    y_pred = skimage.io.imread(path_pair[1]) / 255.
    y_pred = y_pred > PROB_THRESHOLD
    
    y_true = skimage.io.imread(path_pair[0]) / 255.
    if y_true.shape != y_pred.shape:
        y_true = resize(y_true, (y_pred.shape[1], y_pred.shape[0]))
     
    DL = dice_loss(y_true, y_pred.astype(np.float64))

    # Clean up y_pred before computing area and endpoint numbers
    y_pred = clean_mask(y_pred)

    results = {'file_name': fname,
               'mean_intensity': get_img_mean_intensity(img_file),
               'dice_loss': DL, 
               'area_true': y_true.sum(),
               'area_pred': y_pred.sum(),
               'n_endpoints_true': get_endpoint_numbers(y_true > 0),
               'n_endpoints_pred': get_endpoint_numbers(y_pred)
               }
 
    return results


def get_model_results(dataset, model_name):
    print("Retrieving results of {}...".format(model_name))
    
    # Common paths
    cfg = get_common_configs()
    root_path = cfg['root_folder']
    data_root = os.path.join(root_path, cfg['data_subfoler'])
    result_root = os.path.join(root_path, cfg['result_subfolder'])
    img_dir = os.path.join(data_root, dataset, cfg['img_subfolder'])

    y_true_dir = os.path.join(data_root, dataset, cfg['mask_subfolder'])
    y_pred_dir = os.path.join(result_root, model_name, dataset, cfg['prediction_subfolder'])
    
    y_true_paths = get_filenames(y_true_dir, FILE_TYPE, FILTER_PATTERN)
    y_pred_paths = get_filenames(y_pred_dir, FILE_TYPE, FILTER_PATTERN)
    # print()
    
    # check y_true_paths and y_pred_paths have the same file names
    assert [os.path.basename(f) for f in y_true_paths] == \
           [os.path.basename(f) for f in y_pred_paths], 'y_true and y_pred must have same file names'
    
    path_pairs = zip(y_true_paths, y_pred_paths)
    n_tasks = len(y_pred_paths)
    
    # results = []
    # for i, r in enumerate(map(get_results, path_pairs), 1):
    #     results.append(r)
    #     print("  Done {}/{}".format(i, n_tasks), end='\r')
            
    with Pool(N_WORKERS) as p:
        results = []
        for i, r in enumerate(p.imap(get_results, path_pairs), 1):
            results.append(r)
            print("  Done {}/{}".format(i, n_tasks), end='\r')
    
    return results


def results_to_dataframe(results, tag='', **kwargs):   
    df = pd.DataFrame(results)
    df = df[COL_ORDER]

    # Add tag to column names
    if tag is not '':
        df = df.rename(columns={'dice_loss': 'dice_loss' + tag, 
                                'area_pred': 'area_pred' + tag,
                                'n_endpoints_pred': 'n_endpoints_pred' + tag})
    
    return df
    

def get_result_df(dataset, model_name, **kwargs):
    return results_to_dataframe(get_model_results(dataset, model_name), **kwargs)


def get_pred_part(df):
    """ Return columns related to predictions."""    
    pred_cols = [c for c in list(df) if any([t in c for t in TO_TAKE])]
    return df[pred_cols]


def main(dataset, model_names, tags=None, output_csv=None):    
    # Construct result dataframes fro each model
    start = time.time()

    if tags is None:
        tages = [''] * len(model_names)
    dfs = [get_result_df(dataset, m, tag=t) for (m, t) in zip(model_names, tags)]

    # Merge all dataframes
    df = reduce(lambda x, y: pd.merge(x, get_pred_part(y), on=['file_name'], 
                                      how='left', suffixes=(False, False)), dfs)
    del dfs

    print("  Done (Time elapsed: {}s)".format(int(time.time() - start)))

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


# if __name__ == '__main__':
#     main(model_names, dataset)