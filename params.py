import argparse
import distutils.util

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--gpu', default="0")
arg('--fold', type=int, default=None)
arg('--n_folds', type=int, default=5)
arg('--folds_source')
# base_lr = float(clr_params[0])
#         max_lr = float(clr_params[1])
#         step = int(clr_params[2])
#         mode = clr_params[3]
arg('--clr')
arg('--seed', type=int, default=80)
arg('--test_size_float', type=float, default=0.1)
arg('--epochs', type=int, default=30)
arg('--img_height', type=int, default=868)
arg('--img_width', type=int, default=868)
arg('--out_height', type=int, default=868)
arg('--out_width', type=int, default=868)
arg('--input_width', type=int, default=800)
arg('--input_height', type=int, default=800)
arg('--use_crop', type=distutils.util.strtobool, default='true')
arg('--learning_rate', type=float, default=0.0001)
arg('--batch_size', type=int, default=1)
arg('--auto_dataset_dir', default='/media/user/5674E720138ECEDF/geo_data/polygons_for_train')
arg('--manual_dataset_dir', default='/mnt/storage_4tb/ymi/spacenet/data/train')
arg('--models_dir', default='models')
arg('--weights')
arg('--loss_function', default='bce_dice')
arg('--freeze_till_layer', default='input_1')
arg('--show_summary', default=False, action='store_true')
arg('--network', default='simple_unet')
arg('--net_alias', default='')
arg('--preprocessing_function', default='tf')
arg('--mask_suffix', default='.jpg')
arg('--train_df', default='/mnt/storage_4tb/ymi/spacenet/data/train/baseline_train.csv')
arg('--val_df', default='/mnt/storage_4tb/ymi/spacenet/data/train/baseline_val.csv')
arg('--test_df', default='/mnt/storage_4tb/ymi/spacenet/data/test.csv')
arg('--inp_list', default='input_list_rgb')
arg('--transformer_type', default="affine")

arg('--pred_mask_dir')
arg('--pred_tta')
arg('--pred_batch_size', default=8)
arg('--test_data_dir', default='/mnt/storage_4tb/ymi/spacenet/data/test/SpaceNet-Off-Nadir_Test_Public')
arg('--pred_threads', type=int, default=1)
arg('--submissions_dir', default='submissions')
arg('--pred_sample_csv', default='input/sample_submission.csv')
arg('--predict_on_val', type=bool, default=False)
arg('--stacked_channels', type=int, default=0)
arg('--stacked_channels_dir', default=None)
arg('--edges', action='store_true')
# Dir names
arg('--train_data_dir_name', default='jpegs')
arg('--val_data_dir_name', default='jpegs')
arg('--train_mask_dir_name', default='masks')
arg('--val_mask_dir_name', default='masks')
arg('--threshold', default=0.5, type=float)

arg('--test_mask_dir', default='/media/user/5674E720138ECEDF/geo_data/manual_labelling/images_for_labeling')

arg('--dirs_to_ensemble', nargs='+')
arg('--ensembling_strategy', default='average')
arg('--folds_dir')
arg('--ensembling_dir')
arg('--ensembling_cpu_threads', type=int, default=6)
arg('--output_csv')

arg('--output_dir')
arg(
    '--footprint_threshold', '-ft', type=int, default=0,
    help='Minimum footprint size in square pixels to save for output. ' +
         'All footprints smaller than the provided cutoff will be ' +
         'discarded.'
    )
arg(
    '--window_step', '-ws', type=int, default=64,
    help='Step size for sliding window during inference. Window will ' +
         'step this far in x and y directions, and all inferences will ' +
         'be averaged. Helps avoid edge effects. Defaults to 64 pxs. ' +
         'to have no overlap between windows, use the size of the ' +
         'window being tested (512 for ternausnetv1 and unet defaults)'
)
arg(
    '--simplification_threshold', '-s', type=int, default=0,
    help='Threshold for simplifying polygons using ' +
    'shapely.shape.simplify() in units of meters. By default, performs ' +
    'no simplification.')
arg("--verbose")
args = parser.parse_args()