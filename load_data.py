#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:40:37 2019

@author: jurriaan
"""

import os
import numpy as np
import re
from os.path import join

import geopandas as gpd
import rasterio as rio


dict_seasons = {'winter':['01', '02', '03'] , 
                'spring':['04', '05', '06'],
                'summer':['07', '08', '09'], 
                'autumn':['10', '11', '12']}
list_polarizations = ['vh', 'vv']

seasons_fixed_order = ['winter', 'spring', 'summer', 'autumn']


def paths_in_dict(path_data_local):
    dict_paths = dict()
    
    list_areas_year = [dir_area for dir_area in os.listdir(path_data_local) 
                       if os.path.isdir(join(path_data_local, dir_area))]
    if list_areas_year:
        str_year = list_areas_year[0].split('_')[1]
        list_areas = [area_year.split('_')[0] for area_year in list_areas_year]
    else:
        return {}
            
    for area in list_areas:
        dict_paths[area] = dict()
            
        # .tif files
        dict_paths[area]['tif'] = dict()
        for season, list_months in dict_seasons.items():
            dict_paths[area]['tif'][season] = dict()

            for month in list_months:
                for pol in list_polarizations:
                    str_pol = "_" + pol + "_"
                    year_month = str_year + month

                    for root, dirs, files in os.walk(path_data_local):
                        for file in files:
                            path_file = join(root, file)
                            if re.search(area + ".*" + str_pol + ".*" + year_month + ".*.tif$", path_file):
                                if 'BorderMask' not in file:
                                    dict_paths[area]['tif'][season][pol] = path_file
          
        # .shp files
        for root, dirs, files in os.walk(path_data_local):
            for file in files:
                path_file = join(root, file)
                if re.search(area + ".*.shp$", file):
                    dict_paths[area]['shp'] = path_file
                       
    return dict_paths


def tif2raster(dict_paths):
    dict_raster_layers = dict()

    for area in dict_paths:
        dict_raster_layers[area] = []

        for season in seasons_fixed_order:
            for pol in sorted(dict_paths[area]['tif'][season]):
                path_raster = dict_paths[area]['tif'][season][pol]
                raster_obj = rio.open(path_raster)
                dict_raster_layers[area].append(raster_obj)
                
    return dict_raster_layers 


def shp2polygons(dict_paths):
    dict_polygons = dict()
    for area in dict_paths:
        for season in seasons_fixed_order:
            path_shape = dict_paths[area]['shp'][season]
            polygons = gpd.read_file(path_shape)
            dict_polygons[area] = polygons
    return dict_polygons


def load_np_arrays(dir_save_arrays):
    dict_data = dict()
    for file in os.listdir(dir_save_arrays):
        file_name, ext = os.path.splitext(file)
        if ext == '.npy':
            path_file = join(dir_save_arrays, file)
            dict_data[os.path.splitext(file)[0]] = np.load(path_file)
            
    return dict_data


def make_dir(path_dir):
   if not os.path.exists(path_dir):
       os.mkdir(path_dir)
   return path_dir


def load_entire_dataset(path_data):
    print("loading all data from {}".format(path_data))
    for i, file in enumerate(os.path.listdir(path_data)):
        np_arr = np.load(join(path_data, file))

        if i == 0:
            np_arr_all == np_arr
        else:
            np_arr_all = np.concatenate((np_arr, np_arr_all), axis=0)

    print('dim np_arr_all:', np_arr_all.shape)
    return np_arr_all

