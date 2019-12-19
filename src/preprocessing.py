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
from os.path import join
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio as rio

from src.load_data import seasons_fixed_order, make_dir


EPSG_CRS = 'EPSG:32638'


def preprocess_data(dict_paths, dir_save_plots,
                               quantile_clip_max):
    print('preprocessing...')
    dict_arrays = dict()
    dict_raster_layers = tif2raster(dict_paths)

    for area in dict_raster_layers:
        print(area)
        list_np_arrs_area = []
        for season, pol, raster_obj in dict_raster_layers[area]:
            print(' ', season, pol)
            np_arr = raster_obj.read()[0]
            np_arr_norm = process_image(np_arr, quantile_clip_max,  clip_min=0, 
                                        plotting=True, 
                                        dir_save_plots=dir_save_plots,
                                        area=area,  season=season, pol=pol)
            list_np_arrs_area.append((season, pol, np_arr_norm))

        dict_arrays[area] = list_np_arrs_area

    return dict_raster_layers, dict_arrays


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

        for season in seasons_fixed_order:
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
                  dir_save_plots, **kwargs_plotting):
    # need to clip and normalize contrast of figure
    clip_max = np.quantile(np_arr, quantile_clip_max)
    arr_clip = np.clip(np_arr, clip_min, clip_max)
    np_arr_norm = normalize(arr_clip)
    
    if plotting:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(np_arr_norm.flatten(), bins=100)
        title = ''.join([val + '-' for val in kwargs_plotting.values()]) 
        ax.set_title(title)
        fig.savefig(join(dir_save_plots, title + '.png'))
        plt.close()
    
    return np_arr_norm 


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
        make_dir(dir_save)
        
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
        make_dir(dir_save)
        
        path_file_save = join(dir_save, area + '_label')
        np.save(path_file_save, dict_labels[area])
        list_paths_labels.append(path_file_save)

    del dict_labels
    return list_paths_labels
