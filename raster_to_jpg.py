# import os
# from spacenetutilities.labeltools import coreLabelTools as cLT
import cv2
import numpy as np
import rasterio
import pandas as pd
import os
from shapely.geometry import Polygon
import geopandas as gpd
from tqdm import tqdm

def create_jpg_from_pan_sharpen(raster_path, dest_path, threshold=3000):
    with rasterio.open(raster_path, "r") as dataset:
        img = np.floor_divide(dataset.read(),
                                       threshold/255).astype('uint8')
        img = raster_cv(img)
        cv2.imwrite(dest_path, img)


def raster_cv(res):
    if res.shape[0] > 1:
        cv_res = np.zeros((res.shape[1], res.shape[2], res.shape[0]))
        cv_res[:, :, 0] = res[0]
        cv_res[:, :, 1] = res[1]
        cv_res[:, :, 2] = res[2]
    else:
        cv_res = res
    return cv_res


def jpegs_from_rasters(im_src_dir, mask_dest_dir,
                        skip_existing=False, verbose=False):
    """Create mask images from geojsons.

    Arguments:
    ----------
    geojson_dir (str): Path to the directory containing geojsons.
    im_src_dir (str): Path to a directory containing geotiffs corresponding to
        each geojson. Because the georegistration information is identical
        across collects taken at different nadir angles, this can point to
        geotiffs from any collect, as long as one is present for each geojson.
    mask_dest_dir (str): Path to the destination directory.

    Creates a set of binary image tiff masks corresponding to each geojson
    within `mask_dest_dir`, required for creating the training dataset.

    """
    if not os.path.exists(im_src_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(im_src_dir))
    rasters = [f for f in os.listdir(im_src_dir) if f.endswith('.tif')]
    os.makedirs(mask_dest_dir, exist_ok=True)
    for raster in tqdm(rasters):
        chip_id = os.path.splitext('_'.join(raster.split('_')[1:]))[0]
        dest_path = os.path.join(mask_dest_dir, '8bit_' + chip_id + '.jpg')
        if os.path.exists(dest_path) and skip_existing:
            if verbose:
                print('{} already exists, skipping...'.format(dest_path))
            continue
        create_jpg_from_pan_sharpen(os.path.join(im_src_dir, raster), dest_path)


if __name__ == '__main__':
    root_fldr = "data/test/SpaceNet-Off-Nadir_Test_Public"
    fldrs = []
    for i in os.listdir(root_fldr):
        if i.startswith('Atlanta') and os.path.isdir(os.path.join(root_fldr, i)):
            fldrs.append(i)

    for folder in fldrs:
        # geojson_path = "data/train/geojson/spacenet-buildings"
        masks_path = os.path.join(root_fldr, folder, "jpegs")
        img_path = os.path.join(root_fldr, folder, "Pan-Sharpen")# "data/Atlanta_nadir7_catid_1030010003D22F00/Pan-Sharpen"
        # masks_from_geojsons(geojson_path, img_path, masks_path, verbose=True)
        # masks_path = "data/nadir_7_jpegs/"
        # img_path = "data/Atlanta_nadir7_catid_1030010003D22F00/Pan-Sharpen"
        jpegs_from_rasters(img_path, masks_path, verbose=True)
