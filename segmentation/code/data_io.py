"""Module for data IO, preprocessing, and augmentation"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.05'


import functools
import tensorflow as tf


MAX_INTENSITY = 4095.


# Get image and mask from path name
def _get_image_from_path(img_path, mask_path):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=1, dtype=tf.uint16)
    img = tf.image.convert_image_dtype(tf.cast(img, tf.float32) / MAX_INTENSITY, tf.uint8)
    mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=1, dtype=tf.uint8)
    
    # Remove bounday 100 pixels since masks touching boundaries were removed
    w = 100
    img = img[w:-w,w:-w,:]
    mask = mask[w:-w,w:-w,:]
    
    return img, mask


"""
Data augmentation
"""
# Flip images
def flip_images(to_flip, img, mask):
    """Flip image and mask horizonally and vertically with prob = 0.5 (separately)"""
    if to_flip:
        flip_prob = tf.random.uniform([2], 0, 1) # [horizontal, vertical]
        
        # flip horizontally
        img, mask = tf.cond(tf.less(flip_prob[0], 0.5), 
                            lambda: (tf.image.flip_left_right(img), 
                                     tf.image.flip_left_right(mask)),
                            lambda: (img, mask))
        # flip vertically
        img, mask = tf.cond(tf.less(flip_prob[1], 0.5), 
                            lambda: (tf.image.flip_up_down(img), 
                                     tf.image.flip_up_down(mask)),
                            lambda: (img, mask))
    return img, mask


# Random crop
def random_crop(img, mask, size=[500, 700]):
    if size is not None:
        assert len(size) == 2, "size must have 2 elments"
        # Combine image and maks then crop
        comb = tf.concat([img, mask], axis=2)
        crop_size = comb.shape.as_list()
        crop_size[:2] = size
        comb = tf.image.random_crop(comb, size=crop_size)

        # Take out copped image and mask
        img_dim = img.shape[-1]
        img = comb[:,:,:img_dim]
        mask = comb[:,:,img_dim:]
    
    return img, mask


# Assembled augmentation function
def _augment(img, mask, resize=None, scale=1., crop_size=None, to_flip=False):
    if resize is not None:
        img = tf.image.resize(img, size=resize)
        mask = tf.image.resize(mask, size=resize)
    
    # Crop and flip
    img, mask = random_crop(img, mask, size=crop_size)
    img, mask = flip_images(to_flip, img, mask)
    
    # Scale the intensity
    img = tf.cast(img, tf.float32) * scale
    mask = tf.cast(mask, tf.float32) * scale
    
    return img, mask


"""
Input pipeline
"""
def get_dataset(img_paths, mask_paths, preproc_fn=functools.partial(_augment),
                shuffle=False, batch_size=1, threads=5):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(_get_image_from_path, num_parallel_calls=threads)    
    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
    
    if shuffle:
        n_samples = len(img_paths)
        dataset = dataset.shuffle(n_samples)
    
    dataset = dataset.repeat().batch(batch_size)
    
    return dataset  