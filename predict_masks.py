import os
from time import clock

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img, flip_axis
from PIL import Image
import pandas as pd

from models import make_model
from params import args
from evaluate import dice_coef
from datasets import generate_images

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


def predict_and_evaluate():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    # test_mask_dir = args.test_mask_dir
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = args.pred_batch_size
    test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    nbr_test_samples = len(test_df)

    # filenames = [os.path.join(args.test_data_dir, f) for f in test_df['name']]
    # mask_filenames = [os.path.join(args.test_mask_dir, f).replace('.jpg', '.png') for f in test_df['name']]
    # filenames = [f for f in test_df['name']]
    # mask_filenames = [f.replace('.jpg', '.png') for f in test_df['name']]

    start_time = clock()
    dices = []
    for i in range(int(nbr_test_samples / batch_size)):
        img_sizes = []
        x = []
        masks = []
        for j in range(batch_size):
            if i * batch_size + j < len(test_df):
                row = test_df.iloc[i * batch_size + j]
                img_path = os.path.join(args.test_data_dir, row['folder'], 'rgg_labeled', row['name'].replace('.png', '.jpg').replace('nrg', 'rgg'))
                mask_path = os.path.join(args.test_data_dir, row['folder'], 'nrg_masks', row['name'])
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                img_size = img.size
                img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                mask = mask.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                # mask = img_to_array(mask) > 0
                img_sizes.append(img_size)
                if args.edges:
                    img = generate_images(img_to_array(img))
                else:
                    img = img_to_array(img)
                x.append(img)
                masks.append(img_to_array(mask)/255)
        x = np.array(x)
        masks = np.array(masks)
        # x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        # preds = undo_tta(preds, args.pred_tta)
        batch_dice = dice_coef(masks, preds)
        dices.append(batch_dice)
        for j in range(batch_size):
            row = test_df.iloc[i * batch_size + j]
            # filename = filenames[i * batch_size + j]
            filename = row['name']
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            prediction = prediction > 0.5
            pred_im = array_to_img(prediction * 255).resize(img_sizes[j], Image.ANTIALIAS)
            pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        time_spent = clock() - start_time
        print("predicted batch ", str(i))
        print("Time spent: {:.2f}  seconds".format(time_spent))
        print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
        print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))
        print("predicted batch dice {}".format(batch_dice))
    print(np.mean(dices))

if __name__ == '__main__':
    # predict()
    predict_and_evaluate()