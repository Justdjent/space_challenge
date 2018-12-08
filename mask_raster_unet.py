import sys
from tqdm import tqdm
tqdm.monitor_interval = 0
import shutil


import json
from osgeo import ogr
import rasterio.mask
from keras.applications import imagenet_utils
from shapely.geometry import shape, Point, mapping, MultiPolygon, Polygon, LineString
import logging
import os
import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

from models import make_model

from params import args

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt=' %I:%M:%S ', level="INFO")

import rasterio
import numpy as np


class Reader:
    def __init__(self, raster_list: dict):
        self.raster_array = None
        self.meta = None
        self.raster_list = raster_list

    def load_stack(self):
        self.raster_array = {}
        self.meta = {}
        for r_type, path in self.raster_list.items():
            with rasterio.open(path, 'r') as src:
                self.raster_array[r_type] = src.read()
                self.meta[r_type] = src.meta

    def create_nrg(self):
        # self.raster_list['nrg'] = self.raster_list['green'].replace('green', 'nrg')
        path = self.raster_list['green'].split('/')
        path[-1] = path[-1].replace('green', 'nrg')
        self.raster_list['nrg'] = "/".join(path)
        self.raster_array['nrg'] = np.zeros((3,
                                             self.raster_array['green'].shape[1],
                                             self.raster_array['green'].shape[2]),
                                            np.float32)
        self.raster_array['nrg'][0] = self.raster_array['nir'][0]
        self.raster_array['nrg'][1] = self.raster_array['red'][0]
        self.raster_array['nrg'][2] = self.raster_array['green'][0]
        self.meta['nrg'] = self.meta['green'].copy()
        self.meta['nrg']['count'] = 3
        self.meta['nrg']['dtype'] = 'float32'

    def create_rgg(self):
        if 'rgb' not in self.raster_list:
            path = self.raster_list['green'].split('/')
            path[-1] = path[-1].replace('green', 'rgg')
            self.raster_list['rgg'] = "/".join(path)
            self.raster_array['rgg'] = np.zeros(
                (3, self.raster_array['green'].shape[1], self.raster_array['green'].shape[2]),
                np.float32)
            self.raster_array['rgg'][0] = self.raster_array['red'][0]
            self.raster_array['rgg'][1] = self.raster_array['green'][0]
            self.raster_array['rgg'][2] = self.raster_array['green'][0]
            self.meta['rgg'] = self.meta['green'].copy()
            self.meta['rgg']['count'] = 3
            self.meta['rgg']['dtype'] = 'float32'
        else:
            path = self.raster_list['rgb'].split('/')
            path[-1] = path[-1].split('.')[0] + '_rgg.tif'
            self.raster_list['rgg'] = "/".join(path)
            self.meta = {}
            with rasterio.open(self.raster_list['rgb'], 'r') as src:
                meta = src.meta
                meta['count'] = 3
                self.meta['rgg'] = meta
                # print(save_path)
                with rasterio.open(self.raster_list['rgg'], 'w', **meta) as dst:
                    for i in range(1, meta['count'] + 1):
                        if i == 3:
                            dst.write(src.read(2), i)
                        else:
                            dst.write(src.read(i), i)
        print('rgg created')
            # with rasterio.open(self.raster_list['rgb'], 'r') as src:
            #     self.meta['rgg'] = src.meta.copy()
            #     # self.meta['rgg']['count'] = 3
            #     with rasterio.open(self.raster_list['rgg'], 'w', **self.meta['rgg']) as dst:
            #         # self.meta['rgg'] = self.meta['rgb'].copy()
            #         for i in range(1, self.meta['rgg']['count']+1):
            #             if i == 3:
            #                 m = 2
            #             else:
            #                 m = i
            #             print(m)
            #             chan = src.read(m)
            #             print(chan.max())
            #
            #             #src_array = self.raster_array[raster_type][i - 1]
            #             # if m < 4:
            #             # src_array = chan
            #             dst.write(chan, m)
                # self.raster_array[r_type] = src.read()
                # self.meta[r_type] = src.meta
            # self.raster_array['rgg'] = np.zeros(
            #     (3, self.raster_array['rgb'].shape[1], self.raster_array['rgb'].shape[2]))
            # self.raster_array['rgg'][0] = self.raster_array['rgb'][0]
            # self.raster_array['rgg'][1] = self.raster_array['rgb'][1]
            # self.raster_array['rgg'][2] = self.raster_array['rgb'][1]
            # self.meta['rgg']['count'] = 3
            # self.meta['rgg']['dtype'] = 'int8'

    def save_raster(self, raster_type, save_path=None):
        if not save_path:
            save_path = self.raster_list[raster_type]
        with rasterio.open(save_path, 'w', **self.meta[raster_type]) as dst:
            for i in range(1, self.meta[raster_type]['count'] + 1):
                src_array = self.raster_array[raster_type][i - 1]
                dst.write(src_array, i)

    def get_rgg(self):
        return self.raster_array['rgg'], self.meta['rgg']

    def get_nrg(self):
        return self.raster_array['nrg'], self.meta['nrg']


class SegmentatorNN:
    def __init__(self, inp_list):
        # self.raster_path = raster_path
        # self.raster_type = raster_type
        inp = read_json(inp_list)
        self.reader = Reader(inp)
        if 'rgb' in inp.keys():
            self.reader.create_rgg()
        else:
            self.reader.load_stack()
            self.reader.create_rgg()
            self.reader.save_raster('rgg')

        # logger.info('RGG saved')

    def mask_tiles(self, save_mask=True, window_size=30):
        # batch_size = 1
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))
        model = make_model((None, None, 3))
        model.load_weights(args.weights)
        max_values = [1, 1, 1]
        min_values = [0, 0, 0]
        with rasterio.open(self.reader.raster_list['rgg'], 'r') as dataset:
            raster_array = np.zeros((dataset.meta['height'], dataset.meta['width']), np.float32)
            xs = dataset.bounds.left
            window_size_meters = window_size
            window_size_pixels = window_size / (dataset.res[0])
            cnt = 0
            pbar = tqdm()
            while xs < dataset.bounds.right:
                ys = dataset.bounds.bottom
                while ys < dataset.bounds.top:
                    row, col = dataset.index(xs, ys)
                    pbar.set_postfix(Row='{}'.format(row), Col='{}'.format(col))
                    step_row = row - int(window_size_pixels)
                    step_col = col + int(window_size_pixels)
                    res = dataset.read(window=((max(0, step_row), row),
                                               (col, step_col)))
                    rect = [[max(0, step_row), row], [col, step_col]]
                    # print(res.max())
                    # if res.max() > 0:
                    #     print(res.max())
                    if res.dtype == 'float32':
                        if res.max() > 1 or res.max() < 0.02:
                            res[res < 0] = 0
                            res = self.min_max(res, min=min_values, max=max_values)
                        res = self.process_float(res)
                        res = res.astype(np.uint8)
                    img_size = tuple([res.shape[2], res.shape[1]])
                    if res.shape[0] > 1:
                        cv_res = np.zeros((res.shape[1], res.shape[2], res.shape[0]))
                        cv_res[:, :, 0] = res[0]/5.87
                        cv_res[:, :, 1] = res[1]/5.95
                        cv_res[:, :, 2] = res[2]/5.95
                    else:
                        cv_res = res[0]
                    cv_res = cv2.resize(cv_res, (args.input_width, args.input_width))
                    # cv_res = cv_res/5.88
                    # while batch == 1
                    cv_res = np.expand_dims(cv_res, axis=0)
                    x = imagenet_utils.preprocess_input(cv_res, mode=args.preprocessing_function)
                    pred = model.predict(x)
                    # pred = pred > 0.5
                    # pred = cv2.resize(np.uint8(pred[0] * 255), img_size)
                    pred = cv2.resize(pred[0], img_size)
                    # pred = np.uint8(pred * 255)
                    stack_arr = np.dstack([pred, raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]])

                    raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = np.amax(stack_arr, axis=2)
                    # raster_array[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]] = pred
                    pbar.update(1)

                    cnt += 1
                    ys = ys + 0.5 * window_size_meters

                xs = xs + 0.5 * window_size_meters
        raster_array = raster_array > 0.5
        raster_array = (raster_array * 255).astype(np.uint8)
        if save_mask:
            cv2.imwrite(self.reader.raster_list['rgg'].replace('.tif', '_mask.jpg'), raster_array)
        im2, contours, hierarchy = cv2.findContours(raster_array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = self.polygonize(contours)

        # meta
        # bin_meta = self.reader.meta['green']
        # bin_meta['dtype'] = 'uint8'
        # bin_meta['nodata'] = 0
        # bin_raster_path = self.reader.raster_list['rgg'].replace('rgg', 'rgg_bin')
        # save_single_raster(np.expand_dims(raster_array, axis=0), bin_meta, bin_raster_path)
        # poly_path = self.reader.raster_list['rgg'].replace('.tif', '.shp')
        poly_path = os.path.dirname(self.reader.raster_list['rgg'])
        poly_path = os.path.join(poly_path, 'polygons')
        if os.path.isdir(poly_path):
            shutil.rmtree(poly_path, ignore_errors=True)
        os.makedirs(poly_path, exist_ok=True)
        poly_path = os.path.join(poly_path, self.reader.raster_list['rgg'].split('/')[-1].replace('.tif', '.shp'))
        try:
            if len(polygons) != 0:
                save_polys_as_shp(polygons, poly_path)
            else:
                print('no_polygons detected')
        except:
            print('done before')
        del model
        K.clear_session()
        return raster_array
        # raster_array = raster_array.astype(np.uint8)
        # self.save_raster_test(path, self.meta['masked'], raster_array)
        # raster_array = None

    def polygonize(self, contours, transform=True):
        polygons = []
        for i in tqdm(range(len(contours))):
            c = contours[i]
            n_s = (c.shape[0], c.shape[2])
            if n_s[0] > 2:
                if transform:
                    polys = [tuple(i) * self.reader.meta['rgg']['transform'] for i in c.reshape(n_s)]
                else:
                    polys = [tuple(i) for i in c.reshape(n_s)]
                polygons.append(Polygon(polys))
        return polygons

    @staticmethod
    def process_float(array):
        array = array.copy()
        array[array < 0] = 0
        array_ = np.uint8(array * 255)
        return array_

    @staticmethod
    def min_max(X, min, max):
        X_scaled = np.zeros(X.shape)
        for i in range(X.shape[0]):
            X_std = (X[i] - min[i]) / (max[i] - min[i])
            X_scaled[i] = X_std * (1 - 0) + 0

        return X_scaled


def save_single_raster(raster_array, meta, save_path):
    with rasterio.open(save_path, 'w', **meta) as dst:
        for i in range(1, meta['count'] + 1):
            src_array = raster_array[i - 1]
            dst.write(src_array, i)

def save_polys_as_shp(polys, name):
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(name)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # If there are multiple geometries, put the "for" loop here
    for i in range(len(polys)):
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(polys[i].wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        # feat = geom = None  # destroy these

    # Save and close everything
    # ds = layer = feat = geom = None


def read_json(path):
    with open(path, 'r') as json_data:
        d = json.load(json_data)
    return d


def run():
    d = read_json(args.inp_list)
    for grove, inp in d.items():
        segmentator = SegmentatorNN(inp)
        segmentator.mask_tiles()


if __name__ == '__main__':
    # predict()
    run()