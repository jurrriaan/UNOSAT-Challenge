#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:02:03 2019

@author: jurriaan
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.plot import plotting_extent
from shapely.geometry import Polygon, mapping
from rasterio.mask import mask
from os.path import join
import descartes
import pyproj
import warnings
import numpy as np
import h5py
from os.path import join
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio as rio

from src import load_data as ld
from src import utils as utl


def preprocces_data(dict_paths, path_file_hfd5, quantile_clip_max, patch_size,
                    dtype, dir_temp_plots, apply_log,  a_l=1, b_l=0.1,
                    compression_opts=4):
    with h5py.File(path_file_hfd5, 'w') as f:
        for area in dict_paths:
            print(area)
            i = 0
            # g = f.create_group(area)
            dir_save_plots = utl.make_dir(join(dir_temp_plots, area))

            for season in ld.seasons_fixed_order:
                for pol in sorted(dict_paths[area]['tif'][season]):
                    print(' ', season, pol)
                    path_raster = dict_paths[area]['tif'][season][pol]
                    np_arr_processed = process_image(rio.open(path_raster).read()[0],
                                                        quantile_clip_max,
                                                        dtype=dtype,
                                                        apply_log=apply_log,
                                                        a_l=a_l,
                                                        b_l=b_l,
                                                        clip_min=0,
                                                        plotting=True,
                                                        dir_save_plots=dir_save_plots,
                                                        area=area, season=season,
                                                        pol=pol)

                    shape_x, shape_y = np_arr_processed.shape
                    dim_x_add = patch_size[0] - shape_x % patch_size[0]
                    dim_y_add = patch_size[1] - shape_y % patch_size[1]
                    stacked_shape = (shape_x + dim_x_add, shape_y + dim_y_add) + (8,)

                    if i == 0:
                        dset = f.create_dataset(area,
                                                stacked_shape,
                                                dtype=dtype,
                                                chunks=True,
                                                compression="gzip",
                                                compression_opts=compression_opts)
                        print(' shape init', dset.shape)

                    dset[:, :, i] = pad_zeros_xy(np_arr_processed, dim_x_add, dim_y_add)
                    i += 1

            print("shape dataset:", dset.shape)


def preprocess_shape(dict_paths, file_hfd5_labels, sub_sample_shape, dtype,
                     compression_opts=4):
    with h5py.File(file_hfd5_labels, 'w') as f:
        for area in dict_paths:
            print(area)

            dict_polygons = shp2polygons(dict_paths)

            # raster obj only required for shape and crs, this is season or pol independent
            path_raster = dict_paths[area]['tif']['spring']['vh']
            raster_obj = rio.open(path_raster)
            arr_shape = raster_obj.shape

            dim_x_add = sub_sample_shape[0] - arr_shape[0] % sub_sample_shape[0]
            dim_y_add = sub_sample_shape[1] - arr_shape[1] % sub_sample_shape[1]
            arr_new_shape = (arr_shape[0] + dim_x_add,  arr_shape[1] + dim_y_add)

            dset = f.create_dataset(area,
                                    arr_new_shape,
                                    dtype=dtype,
                                    chunks=True,
                                    compression="gzip",
                                    compression_opts=compression_opts)

            dset[:] = pad_zeros_xy(polygons_2_np_arr_mask(dict_polygons[area],
                                                                 raster_obj),
                                   dim_x_add,
                                   dim_y_add
                                  )
            print('')


def pad_zeros_xy(arr, dim_x_add, dim_y_add, both_sides=False):
    """

    :param arr:
    :param dim_x_add:
    :param dim_y_add:
    :param both_sides:
    :return:
    """
    arr_zeros_stack_x = np.zeros((dim_x_add, arr.shape[1]), dtype=np.uint8)
    arr = np.concatenate((arr, arr_zeros_stack_x), axis=0)

    if both_sides:
        arr = np.concatenate((arr_zeros_stack_x, arr), axis=0)

    arr_zeros_stack_y = np.zeros((arr.shape[0], dim_y_add), dtype=np.uint8)
    arr = np.concatenate((arr, arr_zeros_stack_y), axis=1)

    if both_sides:
        arr = np.concatenate((arr_zeros_stack_y, arr), axis=1)
    return arr


def get_label_data(dict_paths, dict_raster_layers):
    dict_polygons = shp2polygons(dict_paths)
    dict_labels = dict()
    
    for area in dict_raster_layers:
        raster_obj = dict_raster_layers[area][0][2]

        dict_labels[area] = polygons_2_np_arr_mask(dict_polygons[area], raster_obj)
    return dict_labels


def shp2polygons(dict_paths):
    dict_polygons = dict()   
    for area in dict_paths:
        path_shape = dict_paths[area]['shp']
        dict_polygons[area] = gpd.read_file(path_shape)
    return dict_polygons


def tif2raster(dict_paths):
    dict_raster_layers = dict()

    for area in dict_paths:
        dict_raster_layers[area] = []

        for season in ld.seasons_fixed_order:
            for pol in sorted(dict_paths[area]['tif'][season]):
                path_raster = dict_paths[area]['tif'][season][pol]
                raster_obj = rio.open(path_raster)
                dict_raster_layers[area].append((season, pol, raster_obj))
                
    return dict_raster_layers 


def exponent(x, a, b):
    return a*x**b


def log(x, a, b):
    return np.log(a*x + b)


# Normalize bands into 0.0 - 1.0 scale
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def process_image(np_arr, quantile_clip_max, clip_min, plotting, 
                  dir_save_plots, dtype, apply_log=False, **kwargs_plotting):
    # need to clip and normalize contrast of figure
    clip_max = np.quantile(np_arr, quantile_clip_max)
    arr_clip = np.clip(np_arr, clip_min, clip_max)

    if apply_log:
        np_arr_log  = log(arr_clip, kwargs_plotting['a_l'], kwargs_plotting['b_l'])
        np_arr_norm = normalize(np_arr_log)
    else:
        np_arr_norm = normalize(arr_clip)

    if plotting:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(np_arr_norm.flatten(), bins=100)
        title = ''.join([str(val) + '-' for val in kwargs_plotting.values()])
        ax.set_title(title)
        fig.savefig(join(dir_save_plots, title + '.png'))
        plt.close()

    return utl.convert2integers(np_arr_norm, dtype)




def polygons_2_np_arr_mask(polygons, raster_obj):
    polygons_resh = polygons.to_crs(raster_obj.crs)
    geoms = polygons_resh['geometry']

    polygon_mask = rio.features.geometry_mask(geometries=geoms,
                                       out_shape=(raster_obj.height, raster_obj.width),
                                       transform=raster_obj.transform,
                                       all_touched=False,
                                       invert=True)


    print('polygon mask shape:', polygon_mask.shape)
    
    polygon_mask_int = np.multiply(polygon_mask, 1)
    print('sum polygon_mask_int:', sum(sum(polygon_mask_int)))
    return polygon_mask_int


def save_stacked_arrays(dict_arrays, dir_save_arrays):
    print('stack and save arrays...')
    list_paths_arrays = []
    for area in dict_arrays:
        print(area)
        list_np_arrs_area = []
        for season, pol, arr in dict_arrays[area]:
            list_np_arrs_area.append(arr)
        
        dir_save = join(dir_save_arrays, 'whole')
        ld.make_dir(dir_save)
        
        path_file_save = join(dir_save, area)
        np.save(path_file_save, np.array(list_np_arrs_area))
        list_paths_arrays.append(path_file_save)

    del list_np_arrs_area
    return list_paths_arrays


def save_labels(dict_labels, dir_save_labels):
    print('save labels...')
    list_paths_labels=[]
    for area in dict_labels:
        dir_save = join(dir_save_labels, 'whole')
        ld.make_dir(dir_save)
        
        path_file_save = join(dir_save, area + '_label')
        np.save(path_file_save, dict_labels[area])
        list_paths_labels.append(path_file_save)

    del dict_labels
    return list_paths_labels
