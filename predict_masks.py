import os
from time import clock
from skimage.measure import label

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img, flip_axis
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed
from models import make_model
from params import args
from evaluate import dice_coef
from datasets import generate_images
from random_transform_mask import pad, unpad, tiles_with_overlap, read_img_opencv # rgb2rgg
# from datasets import build_batch_generator_predict_folder
from CosmiQ_SN4_Baseline.cosmiq_sn4_baseline.metrics import f1_score


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

prediction_dir = args.pred_mask_dir


def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    else:
        return pred

# def predict():
#     output_dir = args.pred_mask_dir
#     model = make_model((None, None, 3))
#     model.load_weights(args.weights)
#     batch_size = args.pred_batch_size
#     nbr_test_samples = 100064
#
#     filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
#
#     start_time = clock()
#     for i in range(int(nbr_test_samples / batch_size) + 1):
#         x = []
#         for j in range(batch_size):
#             if i * batch_size + j < len(filenames):
#                 img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
#                 x.append(img_to_array(img))
#         x = np.array(x)
#         x = preprocess_input(x, args.preprocessing_function)
#         x = do_tta(x, args.pred_tta)
#         batch_x = np.zeros((x.shape[0], 1280, 1920, 3))
#         batch_x[:, :, 1:-1, :] = x
#         preds = model.predict_on_batch(batch_x)
#         preds = undo_tta(preds, args.pred_tta)
#         for j in range(batch_size):
#             filename = filenames[i * batch_size + j]
#             prediction = preds[j][:, 1:-1, :]
#             array_to_img(prediction * 255).save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
#         time_spent = clock() - start_time
#         print("predicted batch ", str(i))
#         print("Time spent: {:.2f}  seconds".format(time_spent))
#         print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
#         print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))


def predict():
    output_dir = args.pred_mask_dir
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = args.pred_batch_size
    test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    nbr_test_samples = len(test_df)
    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]

    start_time = clock()
    for i in range(int(nbr_test_samples / batch_size)):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                # img = imread(os.path.join(img_dir, filename))
                # img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
                img = Image.open(filenames[i * batch_size + j])
                img_size = img.size
                img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                img_sizes.append(img_size)
                if args.edges:
                    img = generate_images(img)

                x.append(img_to_array(img))
        x = np.array(x)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        # preds = undo_tta(preds, args.pred_tta)
        for j in range(batch_size):
            filename = filenames[i * batch_size + j]
            print(filename)
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            prediction = prediction > 0.325
            pred_im = array_to_img(prediction * 255).resize(img_sizes[j], Image.ANTIALIAS)
            try:
                assert pred_im.size == img_size
            except:
                print('bad')
            pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        time_spent = clock() - start_time
        print("predicted batch ", str(i))
        print("Time spent: {:.2f}  seconds".format(time_spent))
        print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
        print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))


def predict():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = args.pred_batch_size
    nbr_test_samples = len(os.listdir(args.test_data_dir))

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
    ss = pd.read_csv(args.pred_sample_csv)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                # img = imread(os.path.join(img_dir, filename))
                # img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
                img = Image.open(filenames[i * batch_size + j])
                img_size = img.size
                # img_padded, pads = pad(np.array(img))
                ## Attention pads are all the same!!!!!
                # img = cv2.resize(np.array(img_padded), (args.input_height, args.input_width))
                img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                img_sizes.append(img_size)
                x.append(np.array(img))
                # x.append(img)
        x = np.array(x, dtype=np.float32)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        x_tta = do_tta(x, args.pred_tta)
        batch_x = np.concatenate((x, x_tta), axis=0)

        # batch_x.
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        preds_tta = undo_tta(preds[batch_size:], args.pred_tta)
        preds = (preds[:batch_size] + preds_tta)/2
        for j in range(batch_size):
            filename = filenames[i * batch_size + j]
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            # prediction = prediction > 0.3125
            pred = cv2.resize(np.uint8(prediction * 255), (101, 101))
            pred_im = pred
            # pred_im = array_to_img(prediction * 255).resize(img_sizes[j], Image.ANTIALIAS)
            # pred_im = unpad(pred, pads)
            try :
                assert pred_im.shape == img_size
            except:
                print('bad')

            cv2.imwrite(os.path.join(output_dir, filename.split('/')[-1].split('.')[0] + ".png"), np.uint8(pred_im))

    ss.to_csv(args.submissions_dir + '/submission_resnet50_fold0_tta_127.csv', index=False)

def predict_folder():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    tile_size = 320

    overlap = 0.8
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))
    model = make_model((args.input_height, args.input_width, 3))
    model.load_weights(args.weights)
    batch_size = 1
    thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    scores = []
    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
    nbr_test_samples = len(filenames)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img_path = os.path.join(args.test_data_dir, filenames[i * batch_size + j])
                mask_path = img_path.replace("jpegs_fine", "masks").replace("8bit_Atlanta_nadir53_catid_1030010003193D00_",
                                                                       "mask_").replace(".jpg", ".tif")
                img_ = read_img_opencv(img_path)
                mask = read_img_opencv(mask_path, mask=True)
                img_size = img_.shape
                # img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)

                # img = rgb2rgg(img)
                img_sizes.append(img_size)
                # tls, rects = split_on_tiles(img, overlap=1)
                # tls, rects = split_mask(img)
                tls, rects = tiles_with_overlap(img_, tile_size, overlap)
                tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                for tile in tls:
                    #tile = cv2.resize(tile, (320, 320))
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        padded_tile, pad_ = pad(tile, tile_size)
                        padded_tiles.append(padded_tile)
                        pads.append(pad_)
                    else:
                        padded_tiles.append(tile)
                        pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                if tile_size != args.input_width:
                    tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in padded_tiles]
                else:
                    tls = [img_to_array(tile) for tile in padded_tiles]
                # tile_sizes = [tile.shape for tile in tls]
                # x.append(tls)
        x = np.array(tls)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        #preds = []
        preds = np.zeros((batch_x.shape[0], args.input_width, args.input_height, 1))
        steps = max(batch_x.shape[0]//6, 1)
        for batch_id in range(steps):
            step_start = batch_id * 6
            step_end = min((batch_id + 1) * 6, batch_x.shape[0])
            # if batch_id == 0:
            pred_ = model.predict_on_batch(batch_x[step_start:step_end])
            preds[step_start:step_end] = pred_[0]
        #preds = np.array(preds)
        pred = np.zeros((img_.shape[0], img_.shape[1]))
        for r, p, s, pad_ in zip(rects, preds, tile_sizes, pads):
            try:
                # res_pred = cv2.resize(p * 255, (s[1], s[0]))
                # res_pred = cv2.resize(p * 255, (s[1], s[0]))
                if tile_size != args.input_width:
                    res_pred = cv2.resize(p * 255, (tile_size, tile_size))
                else:
                    #tls = [img_to_array(tile) for tile in padded_tiles]
                    res_pred = p[:, :, 0] * 255#.astype(np.uint8)
                res_pred = unpad(res_pred, pad_)
                # stack_arr = np.dstack([res_pred, pred[r[1][0]:r[1][1], r[0][0]:r[0][1]]])
                stack_arr = np.dstack([res_pred, pred[r[2]:r[3], r[0]:r[1]]])
                stack_arr = nan_to_num(stack_arr)
                pred[r[2]:r[3], r[0]:r[1]] = stack_arr
                # pred[r[2]:r[3], r[0]:r[1]] = np.amax(stack_arr, axis=2)
                # pred[r[2]:r[3], r[0]:r[1]]
                # pred[r[1]:r[1] + s[0], r[0]:r[0] + s[1]] = np.mean(stack_arr, axis=2)
            except:
                print('hi')
        mask_img = pred
        threshold = 0.6

        img_copy = np.copy(mask_img)
        img_copy[mask_img <= threshold + 0.25] = 0
        img_copy[mask_img > threshold + 0.25] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, 100).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_objects(mask_img, 120).astype(np.uint8)

        prediction = my_watershed(mask_img, img_copy)
        #prediction = pred > thresh * 255
        filename = filenames[i * batch_size + j]
        pred_im = array_to_img(np.expand_dims(prediction*255, axis=2))
        try:
            assert pred_im.size == (img_size[1], img_size[0])
        except:
            print('bad')
        pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        score = dice_coef(mask[:, :, 0] > 0.2, prediction > 0.5)
        scores.append(score)
        print(np.mean(scores))
        write_debug(img_, mask, (prediction * 255).astype(np.uint8), filename, output_dir)


def predict_folder_resize():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    tile_size = 448
    overlap = 0.5
    rm_cutoff = 40
    scores = []
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    set_session(tf.Session(config=config))
    model = make_model((args.input_height, args.input_width, 3))
    model.load_weights(args.weights)
    batch_size = 1
    thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
    nbr_test_samples = len(filenames)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img_path = os.path.join(args.test_data_dir, filenames[i * batch_size + j])
                mask_path = img_path.replace("jpegs_fine", "masks").replace("8bit_Atlanta_nadir53_catid_1030010003193D00_", "mask_").replace(".jpg", ".tif")
                img_ = read_img_opencv(img_path)
                mask = read_img_opencv(mask_path, mask=True)
                img_size = img_.shape
                img_sizes.append(img_size)
                img = cv2.resize(img_, (args.input_width, args.input_height))
                # tls, rects = tiles_with_overlap(img, tile_size, overlap)
                # tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                # for tile in tls:
                if img.shape[0] != tile_size or img.shape[1] != tile_size:
                    padded_tile, pad_ = pad(img, tile_size)
                    padded_tiles.append(padded_tile)
                    pads.append(pad_)
                else:
                    padded_tiles.append(img)
                    pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                #tls = img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in padded_tiles]
                # tile_sizes = [tile.shape for tile in tls]
                x.append(img_to_array(padded_tiles[0]))
        x = np.array(x)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)

        batch_x = x
        pred = model.predict_on_batch(batch_x)
        res_pred = unpad(pred[0][0][:,:,0], pads[0])

        res_pred = cv2.resize(res_pred, (900, 900))
        mask_img = res_pred
        threshold = 0.3

        img_copy = np.copy(mask_img)
        img_copy[mask_img <= threshold + 0.5] = 0
        img_copy[mask_img > threshold + 0.5] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, 100).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_objects(mask_img, 120).astype(np.uint8)

        prediction = my_watershed(mask_img, img_copy)
        # prediction[prediction > 0.5] =1
        #prediction = res_pred > thresh * 255
        filename = filenames[i * batch_size + j]
        #print(prediction.max())
        pred_im = array_to_img(np.expand_dims(prediction * 255, axis=2))
        try:
            assert pred_im.size == (img_size[1], img_size[0])
        except:
            print('bad')
        pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        score = dice_coef(mask[:, :, 0] > 0.2, prediction> thresh)
        scores.append(score)
        print(np.mean(scores), np.argmax(pred[1]), np.argmax(pred[2]))
        write_debug(img_, mask, (prediction * 255).astype(np.uint8), filename, output_dir)

def predict_folder_centercrop():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    tile_size = 320
    overlap = 0.5
    rm_cutoff = 40
    scores = []
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    set_session(tf.Session(config=config))
    model = make_model((args.input_height, args.input_width, 3))
    model.load_weights(args.weights)
    batch_size = 1
    thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
    nbr_test_samples = len(filenames)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img_path = os.path.join(args.test_data_dir, filenames[i * batch_size + j])
                mask_path = img_path.replace("jpegs_fine", "masks").replace("8bit_Atlanta_nadir53_catid_1030010003193D00_", "mask_").replace(".jpg", ".tif")
                img_ = read_img_opencv(img_path)
                mask = read_img_opencv(mask_path, mask=True)
                img_size = img_.shape
                img_sizes.append(img_size)
                #img = cv2.resize(img, (args.input_width, args.input_height))
                height = tile_size
                width = tile_size
                ori_height = 900
                ori_width = 900
                h_start = (ori_height - height) // 2
                w_start = (ori_width - width) // 2
                img = img_[h_start:h_start + height, w_start:w_start + width, :]
                # tls, rects = tiles_with_overlap(img, tile_size, overlap)
                # tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                # for tile in tls:
                if img.shape[0] != tile_size or img.shape[1] != tile_size:
                    padded_tile, pad_ = pad(img, tile_size)
                    padded_tiles.append(padded_tile)
                    pads.append(pad_)
                else:
                    padded_tiles.append(img)
                    pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                #tls = img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in padded_tiles]
                # tile_sizes = [tile.shape for tile in tls]
                x.append(img_to_array(padded_tiles[0]))
        x = np.array(x)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)

        batch_x = x
        pred = model.predict_on_batch(batch_x)
        res_pred = unpad(pred[0][0][:,:,0], pads[0])
        predict = np.zeros((900, 900))
        predict[h_start:h_start + height, w_start:w_start + width] = res_pred
        res_pred = predict
        #res_pred = cv2.resize(res_pred, (900, 900))
        mask_img = res_pred
        threshold = 0.3

        img_copy = np.copy(mask_img)
        img_copy[mask_img <= threshold + 0.5] = 0
        img_copy[mask_img > threshold + 0.5] = 1
        img_copy = img_copy.astype(np.bool)
        img_copy = remove_small_objects(img_copy, 100).astype(np.uint8)

        mask_img[mask_img <= threshold] = 0
        mask_img[mask_img > threshold] = 1
        mask_img = mask_img.astype(np.bool)
        mask_img = remove_small_objects(mask_img, 120).astype(np.uint8)

        prediction = my_watershed(mask_img, img_copy)
        # prediction[prediction > 0.5] =1
        #prediction = res_pred > thresh * 255
        filename = filenames[i * batch_size + j]
        #print(prediction.max())
        pred_im = array_to_img(np.expand_dims(prediction * 255, axis=2))
        try:
            assert pred_im.size == (img_size[1], img_size[0])
        except:
            print('bad')
        pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        score = dice_coef(mask[:, :, 0] > 0.2, prediction> thresh)
        scores.append(score)
        print(np.mean(scores))
        write_debug(img_, mask, (prediction * 255).astype(np.uint8), filename, output_dir)
# def remove_small_objects():
#     if rm_cutoff:
#         labels = label(prediction)
#         labs, cts = np.unique(labels, return_counts=True)
#         labels[np.isin(labels, labs[cts < rm_cutoff])] = 0
#         prediction = labels > 0

def my_watershed(mask1, mask2):
    """
    watershed from mask1 with markers from mask2
    """
    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(mask1, markers, mask=mask1, watershed_line=True)
    return labels


def write_debug(img, mask, res_, filename, output_dir):
    out_dir = output_dir.split("/")[:-1]
    out_dir.append("debug_lat")
    out_dir = "/".join(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename.split('/')[-1][:-4] + ".png")
    res = np.zeros((900,900,3))
    res[:, :, 0] = res_
    res[:, :, 1] = res_
    res[:, :, 2] = res_
    res = res.astype(np.uint8)
    mask[:, :, 1][mask[:, :, 1] > 1] = 150
    res[:, :, 0][res[:, :, 0] > 1] = 100
    res[:, :, 1][res[:, :, 1] > 1] = 100
    res[:, :, 2][res[:, :, 2] > 1] = 30
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(img, alpha, mask, beta, 0.0)
    dst_2 = cv2.addWeighted(dst, alpha, res, beta, 0.0)
    cv2.imwrite(out_path, dst_2)


def nan_to_num(arr):
    # test_arr = np.array([[0.5, 0.6, 0],
    #                      [0.1, 0, 1]])
    #arr[arr < 0.1] = np.nan
    my_mean = np.mean(arr, axis=2)
    my_mean = np.nan_to_num(my_mean)
    return my_mean
if __name__ == '__main__':
    # predict()
    # predict_and_evaluate()
    predict_folder_resize()
    # predict_folder()
    # predict_folder_centercrop()