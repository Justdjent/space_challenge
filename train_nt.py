import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

from CyclicLearningRate import CyclicLR
from datasets import build_batch_generator, generate_filenames
from losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border
from models import make_model
from params import args
from utils import freeze_model, ThreadsafeIter
from CosmiQ_SN4_Baseline.cosmiq_sn4_baseline.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def main():
    man_dir = args.manual_dataset_dir
    auto_dir = args.auto_dataset_dir
    # man_mask_dir = os.path.join(args.manual_dataset_dir, args.train_mask_dir_name)
    # man_val_mask_dir = os.path.join(args.manual_dataset_dir, args.val_mask_dir_name)
    # auto_mask_dir = os.path.join(args.manual_dataset_dir, args.train_mask_dir_name)
    # auto_val_mask_dir = os.path.join(args.manual_dataset_dir, args.val_mask_dir_name)
    #
    # man_train_data_dir = os.path.join(args.auto_dataset_dir, args.train_data_dir_name)
    # man_val_data_dir = os.path.join(args.auto_dataset_dir, args.val_data_dir_name)
    # auto_train_data_dir = os.path.join(args.auto_dataset_dir, args.train_data_dir_name)
    # auto_val_data_dir = os.path.join(args.auto_dataset_dir, args.val_data_dir_name)
    # man_mask
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    set_session(tf.Session(config=config))
    # mask_dir = 'data/train/masks_fail'
    # val_mask_dir = 'data/val/masks'
    #
    # train_data_dir = 'data/train/images_fail'
    # val_data_dir = 'data/val/images'

    if args.net_alias is not None:
        formatted_net_alias = '-{}-'.format(args.net_alias)

    best_model_file =\
        '{}/{}{}loss-{}-fold_{}-{}{:.6f}'.format(args.models_dir, args.network, formatted_net_alias, args.loss_function, args.fold, args.input_width, args.learning_rate) +\
        '-{epoch:d}-{val_prediction_loss:0.7f}-{val_prediction_dice_coef:0.7f}-{val_prediction_f1_score:0.7f}-{val_nadir_output_acc:0.7f}-{val_tangent_output_acc:0.7f}.h5'
    if args.edges:
        ch = 3
    else:
        ch = 3
    # model = make_model((None, None, args.stacked_channels + ch))
    model = make_model((args.input_height, args.input_width, args.stacked_channels + ch))

    freeze_model(model, args.freeze_till_layer)

    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()

    model.compile(loss=[make_loss(args.loss_function), 'categorical_crossentropy', 'categorical_crossentropy'],
                  optimizer=optimizer, loss_weights=[1, 0.5, 0.11],
                  metrics={'prediction': [dice_coef_border, dice_coef, binary_crossentropy, dice_coef_clipped, f1_score],
                           'nadir_output': 'accuracy',
                           'tangent_output': 'accuracy'})

    crop_size = None

    if args.use_crop:
        crop_size = (args.input_height, args.input_width)
        print('Using crops of shape ({}, {})'.format(args.input_height, args.input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')

    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)
    # folds_df = pd.read_csv(os.path.join(args.dataset_dir, args.folds_source))
    # train_ids = generate_filenames(folds_df[folds_df.fold != args.fold]['id'])
    # val_ids = generate_filenames(folds_df[folds_df.fold == args.fold]['id'])
    # train_ids = os.listdir(train_data_dir)
    # val_ids = os.listdir(val_data_dir)

    print('Training fold #{}, {} in train_ids, {} in val_ids'.format(args.fold, len(train_df), len(val_df)))

    train_generator = build_batch_generator(
        train_df,
        img_man_dir=man_dir,
        img_auto_dir=auto_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        # mask_dir=mask_dir,
        aug=True
    )

    val_generator = build_batch_generator(
        val_df,
        img_man_dir=man_dir,
        img_auto_dir=auto_dir,
        batch_size=args.batch_size,
        shuffle=False,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        # mask_dir=val_mask_dir,
        aug=False
    )

    best_model = ModelCheckpoint(best_model_file, monitor='val_prediction_dice_coef',
                                                  verbose=1,
                                                  save_best_only=False,
                                                  save_weights_only=True,
                                                  mode='max')

    callbacks = [best_model,
                 # EarlyStopping(patience=45, verbose=10),
                 TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
                 ]
                 # ReduceLROnPlateau(monitor='val_prediction_dice_coef', mode='max', factor=0.2, patience=5, min_lr=0.00001,
                 #                   verbose=1)]
    if args.clr is not None:
        clr_params = args.clr.split(',')
        base_lr = float(clr_params[0])
        max_lr = float(clr_params[1])
        step = int(clr_params[2])
        mode = clr_params[3]
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step, mode=mode)
        callbacks.append(clr)
    model.fit_generator(
        ThreadsafeIter(train_generator),
        steps_per_epoch=len(train_df) / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=ThreadsafeIter(val_generator),
        validation_steps=len(val_df) / args.batch_size + 1,
        callbacks=callbacks,
        max_queue_size=50,
        workers=4)

if __name__ == '__main__':
    main()
