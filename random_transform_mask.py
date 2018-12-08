import random
import os

import keras.backend as K
import numpy as np
# from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis, \
#     load_img, img_to_array
from keras.preprocessing.image import random_channel_shift, load_img, img_to_array
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)
import cv2

class ImageWithMaskFunction:
    def __init__(self, out_size, mask_dir, mask_suffix=".png", crop_size=None):
        super().__init__()
        self.out_size = out_size
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.crop_size = crop_size

    def random_transform(self,
                         x,
                         mask,
                         rotation_range=None,
                         height_shift_range=None,
                         width_shift_range=None,
                         shear_range=None,
                         zoom_range=None,
                         channel_shift_range=None,
                         horizontal_flip=None, vertical_flip=None, fill_mode='reflect', cval=0):
        """Randomly augment a image tensor and mask.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2

        # use composition of homographies
        # to generate final transform that needs to be applied
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
        else:
            theta = 0

        if height_shift_range:
            uniform = np.random.uniform(-height_shift_range, height_shift_range)
            tx = uniform * x.shape[img_row_axis]
            tmx = uniform * mask.shape[img_row_axis]
        else:
            tx = 0
            tmx = 0

        if width_shift_range:
            random_uniform = np.random.uniform(-width_shift_range, width_shift_range)
            ty = random_uniform * x.shape[img_col_axis]
            tmy = random_uniform * mask.shape[img_col_axis]
        else:
            ty = 0
            tmy = 0

        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

        transform_matrix = None
        transform_matrix_mask = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
            transform_matrix_mask = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            shift_matrix_mask = np.array([[1, 0, tmx],
                                          [0, 1, tmy],
                                          [0, 0, 1]])

            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
            transform_matrix_mask = shift_matrix_mask if transform_matrix_mask is None else np.dot(
                transform_matrix_mask,
                shift_matrix_mask)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
            transform_matrix_mask = shear_matrix if transform_matrix_mask is None else np.dot(transform_matrix_mask,
                                                                                              shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
            transform_matrix_mask = zoom_matrix if transform_matrix_mask is None else np.dot(transform_matrix_mask,
                                                                                             zoom_matrix)
        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=fill_mode, cval=cval)

        if transform_matrix_mask is not None:
            h, w = mask.shape[img_row_axis], mask.shape[img_col_axis]
            transform_matrix_mask = transform_matrix_offset_center(transform_matrix_mask, h, w)
            mask[:, :, 0:1] = apply_transform(mask[:, :, 0:1], transform_matrix_mask, img_channel_axis,
                                              fill_mode='constant', cval=0.)
        if channel_shift_range != 0:
            x = random_channel_shift(x, channel_shift_range,
                                     img_channel_axis)
        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                mask = flip_axis(mask, img_col_axis)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                mask = flip_axis(mask, img_row_axis)

        return x, mask

    def mask_pred(self, batch_x, filenames, index_array, img_man_dir, img_auto_dir, aug=False):
        mask_pred = np.zeros((len(batch_x), self.out_size[0], self.out_size[1], 1), dtype=K.floatx())
        mask_pred[:, :, :, :] = 0.
        augmentation = strong_aug(p=0.9)
        for i, (ind, j) in enumerate(filenames.iterrows()):
            fname = j['name'].replace('rgg', 'rgg')
            if j['manual'] == 1:
                mask = os.path.join(img_man_dir, j['folder'], 'masks', fname.split('/')[-1].replace(".jpg", self.mask_suffix).replace("rgg", 'nrg'))
            else:
                mask = os.path.join(img_auto_dir, j['folder'], 'train_masks', fname.split('/')[-1].replace(".jpg", self.mask_suffix))
            # mask_img = load_img(mask, grayscale=True, target_size=(self.out_size[0], self.out_size[1]))
            mask_img = img_to_array(load_img(mask, grayscale=True))
            if mask_img.shape[:2] != self.out_size:
                _, mask_img = pad_img(None, mask_img, self.out_size)
            mask_pred[i, :, :, :] = mask_img
            #
            mask_pred = mask_pred > 0
            if aug:

                data = {"image": batch_x[i, :, :, :].astype(np.uint8), 'mask': mask_pred[i, :, :, :].astype(np.float32)}
                augmented = augmentation(**data)
                batch_x[i, :, :, :], mask_pred[i, :, :, :] = augmented["image"], augmented['mask'].reshape((augmented['mask'].shape[0],
                                                                                                            augmented['mask'].shape[1],
                                                                                                            1))
                # batch_x[i, :, :, :], mask_pred[i, :, :, :] = self.random_transform(x=batch_x[i, :, :, :],
                #                                                                    mask=mask_pred[i, :, :, :],
                #                                                                    height_shift_range=0.2,
                #                                                                    width_shift_range=0.2,
                #                                                                    shear_range=0.0,
                #                                                                    rotation_range=45,
                #                                                                    zoom_range=[0.7, 1.2],
                #                                                                    channel_shift_range=0.1,
                #                                                                    horizontal_flip=True,
                #                                                                    vertical_flip=True)

        if self.crop_size:
            height = self.crop_size[0]
            width = self.crop_size[1]
            ori_height = self.out_size[0]
            ori_width = self.out_size[1]
            if aug:
                h_start = random.randint(0, ori_height - height - 1)
                w_start = random.randint(0, ori_width - width - 1)
            else:
                # validate on center crops
                h_start = (ori_height - height) // 2
                w_start = (ori_width - width) // 2
            MASK_CROP = mask_pred[:, h_start:h_start + height, w_start:w_start + width, :]
            return batch_x[:, h_start:h_start + height, w_start:w_start + width, :], MASK_CROP
        else:
            return batch_x, mask_pred

    def mask_pred_train(self, batch_x, filenames, index_array, l):
        return self.mask_pred(batch_x, filenames, index_array, True)

    def mask_pred_val(self, batch_x, filenames, index_array, l):
        return self.mask_pred(batch_x, filenames, index_array, False)


def pad_img(img, mask, shape):
    # pad_shape = np.int8((np.array(shape) - np.array(mask.shape[:2]))/2)
    padded_img = None
    padded_mask = None
    # print(pad_shape)
    if isinstance(mask, np.ndarray):
        pad_shape = np.int8(np.ceil(((np.array(shape) - np.array(mask.shape[:2])) / 2)))
        if pad_shape.min() < 0:
            padded_mask = cv2.resize(mask, shape)
            padded_mask = np.expand_dims(padded_mask, axis=2)
        else:
            padded_mask = np.pad(mask, ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (0, 0)), 'reflect')
            padded_mask = padded_mask[:shape[0], :shape[1], :]
    if isinstance(img, np.ndarray):
        pad_shape = np.int16(np.ceil((np.array(shape) - np.array(img.shape[:2])) / 2))
        if pad_shape.min() < 0:
            padded_img = cv2.resize(img, shape)
        else:
            padded_img = np.zeros((img.shape[0] + 2 * pad_shape[0],
                                   img.shape[1] + 2 * pad_shape[1], 3), dtype=np.uint8)
            for i in range(3):
                padded_img[:, :, i] = np.pad(img[:, :, i], ((pad_shape[0],pad_shape[0]), (pad_shape[1],pad_shape[1])), 'reflect')
            padded_img = padded_img[:shape[0], :shape[1], :]
    # print(paded_mask.shape, paded_img.shape)
    return padded_img, padded_mask


def random_transform_two_masks(x,
                               mask1,
                               mask2,
                               rotation_range=None,
                               height_shift_range=None,
                               width_shift_range=None,
                               shear_range=None,
                               zoom_range=None,
                               channel_shift_range=None,
                               horizontal_flip=None, vertical_flip=None, fill_mode='constant', cval=0):
    """Randomly augment a image tensor and masks.

    # Arguments
        x: 3D tensor, single image.

    # Returns
        A randomly transformed version of the input (same shape).
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0

    if height_shift_range:
        uniform = np.random.uniform(-height_shift_range, height_shift_range)
        tx = uniform * x.shape[img_row_axis]
        tmx1 = uniform * mask1.shape[img_row_axis]
        tmx2 = uniform * mask2.shape[img_row_axis]
    else:
        tx = 0
        tmx1 = 0
        tmx2 = 0

    if width_shift_range:
        random_uniform = np.random.uniform(-width_shift_range, width_shift_range)
        ty = random_uniform * x.shape[img_col_axis]
        tmy1 = random_uniform * mask1.shape[img_col_axis]
        tmy2 = random_uniform * mask2.shape[img_col_axis]
    else:
        ty = 0
        tmy1 = 0
        tmy2 = 0

    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None
    transform_matrix_mask1 = None
    transform_matrix_mask2 = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix
        transform_matrix_mask1 = rotation_matrix
        transform_matrix_mask2 = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        shift_matrix_mask1 = np.array([[1, 0, tmx1],
                                       [0, 1, tmy1],
                                       [0, 0, 1]])
        shift_matrix_mask2 = np.array([[1, 0, tmx2],
                                       [0, 1, tmy2],
                                       [0, 0, 1]])

        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
        transform_matrix_mask1 = shift_matrix_mask1 if transform_matrix_mask1 is None else np.dot(
            transform_matrix_mask1,
            shift_matrix_mask1)
        transform_matrix_mask2 = shift_matrix_mask1 if transform_matrix_mask2 is None else np.dot(
            transform_matrix_mask2,
            shift_matrix_mask2)

    if shear != 0:
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
        transform_matrix_mask1 = shear_matrix if transform_matrix_mask1 is None else np.dot(transform_matrix_mask1,
                                                                                            shear_matrix)
        transform_matrix_mask2 = shear_matrix if transform_matrix_mask2 is None else np.dot(transform_matrix_mask2,
                                                                                            shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)
        transform_matrix_mask1 = zoom_matrix if transform_matrix_mask1 is None else np.dot(transform_matrix_mask1,
                                                                                           zoom_matrix)
        transform_matrix_mask2 = zoom_matrix if transform_matrix_mask2 is None else np.dot(transform_matrix_mask2,
                                                                                           zoom_matrix)
    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis,
                            fill_mode=fill_mode, cval=cval)

    if transform_matrix_mask1 is not None:
        h, w = mask1.shape[img_row_axis], mask1.shape[img_col_axis]
        transform_matrix_mask1 = transform_matrix_offset_center(transform_matrix_mask1, h, w)
        mask1[:, :, 0:1] = apply_transform(mask1[:, :, 0:1], transform_matrix_mask1, img_channel_axis,
                                           fill_mode='constant', cval=0.)
    if transform_matrix_mask2 is not None:
        h, w = mask2.shape[img_row_axis], mask2.shape[img_col_axis]
        transform_matrix_mask2 = transform_matrix_offset_center(transform_matrix_mask2, h, w)
        mask2[:, :, 0:1] = apply_transform(mask2[:, :, 0:1], transform_matrix_mask2, img_channel_axis,
                                           fill_mode='constant', cval=0.)
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range,
                                 img_channel_axis)
    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)
            mask1 = flip_axis(mask1, img_col_axis)
            mask2 = flip_axis(mask2, img_col_axis)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)
            mask1 = flip_axis(mask1, img_row_axis)
            mask2 = flip_axis(mask2, img_row_axis)

    return x, mask1, mask2



# height_shift_range=0.2,
# width_shift_range=0.2,
# shear_range=0.0,
# rotation_range=45,
# zoom_range=[0.7, 1.2],
# channel_shift_range=0.1,
# horizontal_flip=True,
# vertical_flip=True)
def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)