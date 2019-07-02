""" Analyze prediction results to get total length and tip numbers. 


Chien-Hsiang Hsu, 2019.07.01
"""



# input data
exp_name = 'adults_larvae'
model_names = ['A_uw', 'AL_uw', 'AL_uw_deep']
tags = ['_A', '_AL', '_deep']
# model_names = ['A_uw']
# tags = ['_A']

root_path = r'/awlab/users/chsu/WorkSpace/tensorflow/neuron'
# root_path = r'D:\USERS\Han-Hsuan\neuron')

PROB_THRESHOLD = 0.5
n_workers = 200

file_type = '*.png'
filter_pattern = None

#--------------------------------------------------------------------------------------
import sys, os, re
import glob

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


# Common paths
data_root = os.path.join(root_path, 'data')
result_root = os.path.join(root_path, 'results')
img_dir = os.path.join(data_root, exp_name, 'images')
y_true_dir = os.path.join(data_root, exp_name, 'masks')


col_order = ['file_name', 'mean_intensity', 'area_true', 'n_endpoints_true', 
             'dice_loss', 'area_pred', 'n_endpoints_pred']
to_take = ['file_name', 'dice_loss', 'area_pred', 'n_endpoints_pred']



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


def get_y_pred_path(model_name):
    y_pred_dir = os.path.join(result_root, model_name, exp_name, 'predictions')
    return get_filenames(y_pred_dir, file_type, filter_pattern)


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
    """Skeletonize and remove fragments without branches.
    
    Inputs:
    ---------
    img: a boolean mask.
    
    Outputs
    ---------
    A boolean mask.
    """
    assert img.dtype == np.bool, "img must be boolean."
    
    sk_img = skeletonize(img)
    j_img = find_junctions(sk_img)
    return reconstruction(j_img, sk_img) > 0


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

    fname = os.path.basename(path_pair[1])
    img_file = os.path.join(img_dir, fname) # path to the input image

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


def get_model_results(model_name):
    print("Retrieving results of {}...".format(model_name))
    
    y_pred_paths = get_y_pred_path(model_name)
    y_true_paths = get_filenames(y_true_dir, file_type, filter_pattern)
    print()
    
    # check y_true_paths and y_pred_paths have the same file names
    assert [os.path.basename(f) for f in y_true_paths] == \
           [os.path.basename(f) for f in y_pred_paths], 'y_true and y_pred must have same file names'
    
    path_pairs = zip(y_true_paths, y_pred_paths)
    n_tasks = len(y_pred_paths)
    
    # results = [get_results(p) for p in path_pairs]
    # results = []
    # for i, r in enumerate(map(get_results, path_pairs), 1):
    #     results.append(r)
    #     print("  Done {}/{}".format(i, n_tasks), end='\r')
            
    with Pool(n_workers) as p:
        results = []
        for i, r in enumerate(p.imap(get_results, path_pairs), 1):
            results.append(r)
            print("  Done {}/{}".format(i, n_tasks), end='\r')
    
    return results


def results_to_dataframe(results, tag='', **kwargs):   
    df = pd.DataFrame(results)
    df = df[col_order]

    # Add tag to column names
    if tag is not '':
        df = df.rename(columns={'dice_loss': 'dice_loss' + tag, 
                                'area_pred': 'area_pred' + tag,
                                'n_endpoints_pred': 'n_endpoints_pred' + tag})
    
    return df
    

def get_result_df(model_name, **kwargs):
    return results_to_dataframe(get_model_results(model_name), **kwargs)


def get_pred_part(df):
    """ Return columns related to predictions."""    
    pred_cols = [c for c in list(df) if any([t in c for t in to_take])]
    return df[pred_cols]


def main(save_csv=False):
    # Construct result dataframes fro each model
    start = time.time()

    dfs = [get_result_df(m, tag=t) for (m, t) in zip(model_names, tags)]

    # Merge all dataframes
    df = reduce(lambda x, y: pd.merge(x, get_pred_part(y), on=['file_name'], 
                                      how='left', suffixes=(False, False)), dfs)
    del dfs

    print("  Done (Time elapsed: {}s)".format(int(time.time() - start)))

    if save_csv:
        df.to_csv('results.csv', index=False)

    return df


if __name__ == '__main__': 
    main()