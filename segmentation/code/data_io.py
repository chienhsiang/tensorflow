"""Module for data IO, preprocessing, and augmentation"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.05'


import functools
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


# Get image and mask from path name
def _get_image_from_path(img_path, mask_path, channels=1, dtype=tf.uint8):
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=channels, dtype=dtype)
    mask = tf.image.decode_png(tf.io.read_file(mask_path), channels=channels, dtype=dtype)
    
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

        # Combine image and mask then crop
        comb = tf.concat([img, mask], axis=2)
        crop_size = comb.shape.as_list()        
        crop_size[:2] = size
        # size.append(comb.shape[2])
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
def get_dataset(img_paths, mask_paths, read_img_fn=functools.partial(_get_image_from_path),
                preproc_fn=functools.partial(_augment),
                shuffle=False, batch_size=1, threads=AUTOTUNE):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(read_img_fn, num_parallel_calls=threads)
    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
    
    if shuffle:
        n_samples = len(img_paths)
        dataset = dataset.shuffle(n_samples)
    
    dataset = dataset.repeat().batch(batch_size)
    
    return dataset


# """
# Pixel weight
# """
# def balancing_weight_tf(mask):
#     """mask is a tensor"""
#     mask = tf.cast(mask, tf.bool)
#     n_ones = tf.math.count_nonzero(mask, dtype=tf.int32)
#     n_zeros = tf.size(mask, out_type=tf.int32) - n_ones
#     x = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_ones, tf.float32)
#     y = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_zeros, tf.float32)
#     wc = tf.where(mask, x, y)
#     wc = wc / tf.reduce_min(wc)
    
#     return wc


# def distance_weight(mask, w0=10, sigma=1):
#     """mask is a numpy array"""
    
#     # bw2label
#     n_objs, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    
#     # compute distance to each object for every pixel
#     H, W = mask.shape
#     D = np.zeros([H, W, n_objs])
    
#     for i in range(1, n_objs+1):
#         bw = np.uint8(lbl==i)
#         D[:,:,i-1] = cv2.distanceTransform(1-bw, cv2.DIST_L2, 3)
        
#     D.sort(axis=-1)
#     weight = w0 * np.exp(-0.5 * (np.sum(D[:,:,:2], axis=-1)**2) / (sigma**2))
    
#     return np.float32(weight)


# def get_pixel_weights(mask, **kwargs):
#     """mask is a tensor"""
    
#     mask = tf.squeeze(mask, axis=-1)
#     wc = balancing_weight_tf(mask)
#     # dw = wc
#     dw = tf.numpy_function(lambda x: distance_weight(x, **kwargs), [mask], tf.float32)

#     return tf.expand_dims(wc + dw, axis=-1)


# def concat_weight(img, mask, **kwargs):
#     mask = tf.concat([tf.cast(mask, tf.float32), 
#                       get_pixel_weights(mask, **kwargs)], axis=-1)
#     # mask = tf.map_fn(lambda x: tf.concat([tf.cast(x, tf.float32), 
#     #                                       get_pixel_weights(x, **kwargs)], axis=-1), 
#     #                  mask, dtype=tf.float32)
        
#     return img, mask

