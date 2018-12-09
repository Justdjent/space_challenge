# import os
# from spacenetutilities.labeltools import coreLabelTools as cLT
import cv2
import numpy as np
import rasterio
import pandas as pd
import os
from shapely.geometry import Polygon
import geopandas as gpd



def _polygons_to_image_view(window_cut_polygons, rect, dataset):
    col_offset = rect[1][0]
    row_offset = rect[0][0]
    img_polyg_array = []
    to_img_mat = ~dataset.meta['transform']
    for item in window_cut_polygons.iterrows():
        if item[1]['geometry'].type != 'Polygon':
            # polyg_spati = np.array(cascaded_union(item[1]['geometry']).exterior.coords)
            continue
        else:
            polyg_spati = np.array(item[1]['geometry'].exterior.coords)
        polyg_img = [tuple(pt) * to_img_mat for pt in polyg_spati]
        polyg_img = np.subtract(polyg_img, (col_offset, row_offset))
        polyg_img = Polygon(polyg_img)
        # polyg_img = polyg_img.convex_hull
        img_polyg_array.append(polyg_img)

    return img_polyg_array


def mask_for_polygons(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def create_raster_from_geojson(json_path, raster_path, dest_path):
    with rasterio.open(raster_path, "r") as dataset:
        try:
            grove_polygons = gpd.read_file(json_path)

            rect = [[0, 0], [0, 0]]
            segment = _polygons_to_image_view(grove_polygons, rect, dataset)
            mask = mask_for_polygons(segment, (dataset.meta['height'], dataset.meta['width'])) * 255
            cv2.imwrite(dest_path, mask)
        except:
            print(json_path)
            mask = np.zeros((dataset.meta['height'], dataset.meta['width']))
            cv2.imwrite(dest_path, mask)



def masks_from_geojsons(geojson_dir, im_src_dir, mask_dest_dir,
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
    print("hi")
    if not os.path.exists(geojson_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(geojson_dir))
    if not os.path.exists(im_src_dir):
        raise NotADirectoryError(
            "The directory {} does not exist".format(im_src_dir))
    geojsons = [f for f in os.listdir(geojson_dir) if f.endswith('json')]
    ims = [f for f in os.listdir(im_src_dir) if f.endswith('.tif')]
    os.makedirs(mask_dest_dir, exist_ok=True)
    for geojson in geojsons:
        chip_id = os.path.splitext('_'.join(geojson.split('_')[1:]))[0]
        dest_path = os.path.join(mask_dest_dir, 'mask_' + chip_id + '.tif')
        if os.path.exists(dest_path) and skip_existing:
            if verbose:
                print('{} already exists, skipping...'.format(dest_path))
            continue
        matching_im = [i for i in ims if chip_id in i][0]
        # assign output below so it's silent
        create_raster_from_geojson(os.path.join(geojson_dir, geojson),
                                        os.path.join(im_src_dir, matching_im),
                                        dest_path)
#         g = cLT.createRasterFromGeoJson(os.path.join(geojson_dir, geojson),
#                                         os.path.join(im_src_dir, matching_im),
#                                         dest_path)

if __name__ == '__main__':
    root_fldr = "data/train"
    fldrs = []
    for i in os.listdir(root_fldr):
        if i.startswith('Atlanta') and os.path.isdir(os.path.join(root_fldr, i)):
            fldrs.append(i)

    for folder in fldrs:
        geojson_path = "data/train/geojson/spacenet-buildings"
        masks_path = os.path.join(root_fldr, folder, "masks")
        img_path = os.path.join(root_fldr, folder, "Pan-Sharpen")# "data/Atlanta_nadir7_catid_1030010003D22F00/Pan-Sharpen"
        masks_from_geojsons(geojson_path, img_path, masks_path, verbose=True)
