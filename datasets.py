import random

import cv2
import numpy as np
import pandas as pd

from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils

from scipy.misc.pilutil import imread

from params import args
from sklearn.model_selection import train_test_split
import sklearn.utils
from random_transform_mask import ImageWithMaskFunction, pad_img
import os


def pad(image, padding_w, padding_h):
    batch_size, height, width, depth = image.shape
    # @TODO: Avoid creating new array
    new_image = np.zeros((batch_size, height + padding_h * 2, width + padding_w * 2, depth), dtype=image.dtype)
    new_image[:, padding_h:(height + padding_h), padding_w:(width + padding_w)] = image
    # @TODO: Fill padded zones
    # new_image[:, :padding_w] = image[:, :padding_w]
    # new_image[:padding_h, :] = image[:padding_h, :]
    # new_image[-padding_h:, :] = image[-padding_h:, :]

    return new_image

def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled

def generate_images(img):
    output = np.zeros((img.shape[0], img.shape[1], 3))
    exr = 1.4 * img[:, :, 0] - img[:, :, 1]
    exr[exr < 0] = 0
    exr = np.uint8(exr)
    output[:, :, 0] = img[:, :, 0]/255
    output[:, :, 1] = img[:, :, 1]/255
    output[:, :, 2] = exr/255
    # output[:, :, 3] = min_max(cv2.Canny(exr, 30, 60))
    # output[:, :, 4] = min_max(cv2.Laplacian(exr, cv2.CV_64F))
    # output[:, :, 5] = min_max(cv2.Sobel(exr, cv2.CV_64F, 1, 0, ksize=7))
    # output[:, :, 6] = min_max(cv2.Sobel(exr, cv2.CV_64F, 0, 1, ksize=7))
    # chan_list = [min_max(chan) for chan in [edges, laplacian, sobelx, sobely]]
    return output

def unpad(image, padding_w):
    return image[:, :, padding_w:(image.shape[1] - padding_w), :]


def generate_filenames(car_ids):
    return ['{}_{}.jpg'.format(id, str(angle + 1).zfill(2)) for angle in range(16) for id in car_ids]

def bootstrapped_split(car_ids, seed=args.seed):
    """
    # Arguments
        metadata: metadata.csv provided by Carvana (should include
        `train` column).

    # Returns
        A tuple (train_ids, test_ids)
    """
    all_ids = pd.Series(car_ids)
    train_ids, valid_ids = train_test_split(car_ids, test_size=args.test_size_float,
                                                     random_state=seed)

    np.random.seed(seed)
    bootstrapped_idx = np.random.random_integers(0, len(train_ids))
    bootstrapped_train_ids = train_ids[bootstrapped_idx]

    return generate_filenames(bootstrapped_train_ids.values), generate_filenames(valid_ids)


def build_batch_generator(filenames, img_man_dir=None, img_auto_dir=None, batch_size=None,
                          shuffle=False, transformations=None,
                          out_size=None, crop_size=None, mask_dir=None, aug=False):
    mask_function = ImageWithMaskFunction(out_size=out_size, crop_size=crop_size, mask_dir=mask_dir)

    while True:
        # @TODO: Should we fixate the seed here?
        if shuffle:
            filenames = sklearn.utils.shuffle(filenames)

        for start in range(0, len(filenames), batch_size):
            batch_x = []
            tangents = []
            nadirs = []
            weights = []
            end = min(start + batch_size, len(filenames))
            train_batch = filenames[start:end]

            for ind, filename in train_batch.iterrows():
                if filename['manual'] == 1:
                    img_path = os.path.join(img_man_dir, filename['folder'], 'jpegs')
                    # mask_path = os.path.join(img_man_dir)
                else:
                    img_path = os.path.join(img_auto_dir, filename['folder'], 'rgg_train_imgs')
                    # mask_path = os.path.join(img_auto_dir)
                # img = imread(os.path.join(img_dir, filename))
                # img = img_to_array(
                #     load_img(os.path.join(img_dir, filename), grayscale=False, target_size=(out_size[0], out_size[1])))
                img = img_to_array(
                    load_img(os.path.join(img_path, "8bit_" + filename['name'].replace('mask', filename['folder']).replace('.tif', '.jpg')), grayscale=False))
                tangent = np_utils.to_categorical(filename['tangent_label'], 3)
                nadir = np_utils.to_categorical(filename['nadir_cat_label'], 3)
                if img.shape[:2] != out_size:
                    img, mask_img = pad_img(img, None, out_size)
                if args.edges:
                    img = generate_images(img)
                stacked_channels = []
                for i in range(args.stacked_channels):
                    channel_path = os.path.join(args.stacked_channels_dir,
                                                str(i),
                                                filename.replace('.jpg', '.png'))
                    stacked_channel = imread(channel_path, mode='L')
                    stacked_channels.append(stacked_channel)
                stacked_img = np.dstack((img, *stacked_channels))
                batch_x.append(stacked_img)
                weights.append(filename['weight'])
                nadirs.append(nadir)
                tangents.append(tangent)
            batch_x = np.array(batch_x, np.float32)
            batch_x, masks = mask_function.mask_pred(batch_x, train_batch, range(batch_size), img_man_dir,img_auto_dir , aug)
            weights = np.array(weights)
            nadirs = np.array(nadirs)
            tangents = np.array(tangents)
            if crop_size is None:
                # @TODO: Remove hardcoded padding
                batch_x, masks = pad(batch_x, 1, 0), pad(masks, 1, 0)
            if args.edges:
                yield batch_x, masks, weights
            else:
                yield imagenet_utils.preprocess_input(batch_x, mode=args.preprocessing_function),\
                      [masks, nadirs, tangents]#, weights


def get_edges(image):
    img = np.uint8(image[:,:,0])
    out = np.zeros((args.img_height, args.img_width, 5))
    laplacian = min_max(cv2.Laplacian(img, cv2.CV_64F))
    sobelx = min_max(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    sobely = min_max(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))
    edges = min_max(cv2.Canny(img, 30, 200))
    for n, c in zip(range(5), [image[:, :, 0]/255, laplacian, sobelx, sobely, edges]):
        out[:, :, n] = c
    return out


def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled

