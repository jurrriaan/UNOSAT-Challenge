#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:27:07 2019

@author: jurriaan
"""

import sys
print(sys.executable)

import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings

# see: https://github.com/karolzak/keras-unet
from keras.optimizers import Adam
from keras_unet.models import satellite_unet 

import load_data as ld
import preprocessing as pp
import prepare_for_network as pfn
import deep_learning as dl


warnings.simplefilter("ignore")

# PARAMETERS
test_run = False  # False
save_processed_data = True  # set first time running at True
prepare_for_network = True  # set first time running at True
deeplearning = True  # set at True if deep learning

#size_sub_sample = (256, 256)
quantile_clip_max = 0.999
size_sub_sample = (512, 512)
inv_stride = 2
EPOCHS = 1
INIT_LR = 1e-3
BS = 32

# local paths, set your own path here!
path_data_main = '/Volumes/other/datasets_and_ML/UNOSAT_Challenge'

# put your data in these dirs
path_data_local_train = join(path_data_main, 'Train_Dataset')
#path_data_local_val = join(path_data_main, 'Validation_Dataset')
path_data_local_test = join(path_data_main, 'Evaluation_Dataset')


# other dirs are made automatically
dir_temp_data = ld.make_dir(join(path_data_main, 'data_temp'))
dir_temp_data_train = ld.make_dir(join(dir_temp_data, 'train'))
dir_temp_data_test = ld.make_dir(join(dir_temp_data, 'test'))
dir_temp_plots = ld.make_dir(join(dir_temp_data, 'plots'))
dir_temp_labels = ld.make_dir(join(dir_temp_data, 'labels'))

# paths test-run data
file_X = join(path_data_main, 'data_temp', 'train', 'split', 'Mosul_split_.npy')
file_Y = join(path_data_main, 'data_temp', 'labels', 'split', 'Mosul_split_.npy')


# LOAD AND PROCESS DATA
if save_processed_data:
    dict_info_data_train = ld.paths_in_dict(path_data_local_train)

    dict_raster_layers_train, dict_data_train = pp.preprocess_data(dict_info_data_train, dir_temp_plots, quantile_clip_max)
    dict_labels = pp.get_label_data(dict_info_data_train, dict_raster_layers_train)
    list_paths_data_train = pp.save_stacked_arrays(dict_info_data_train, dir_temp_data_train)
    list_paths_labels = pp.save_labels(dict_labels, dir_temp_labels)


#_, dict_data_test = pp.preprocess_data(dict_info_data_train, dir_temp_plots, quantile_clip_max)
#list_paths_data_test = pp.save_stacked_arrays(dict_data_test, dir_temp_data_train)


# PREPARE FOR NETWORK
if prepare_for_network:
    dir_temp_data_train_whole = ld.make_dir(join(dir_temp_data_train, 'whole'))
    dir_temp_labels_whole = ld.make_dir(join(dir_temp_labels, 'whole'))

    list_dir_np = [f for f in os.listdir(dir_temp_data_train_whole) if os.path.splitext(f)[1] == '.npy']
    list_dir_np_label = [f for f in os.listdir(dir_temp_labels_whole) if os.path.splitext(f)[1] == '.npy']

    for area_data_file, area_label_file in zip(list_dir_np, list_dir_np_label):
        print(area_data_file, area_label_file)
        pfn.pipeline_prepare_for_nn(area_data_file, dir_temp_data_train, inv_stride, size_sub_sample)
        pfn.pipeline_prepare_for_nn_labels(area_label_file, dir_temp_labels, inv_stride, size_sub_sample)
    
if deeplearning:
    # DEEP LEARNING
    IMAGE_DIMS = size_sub_sample + (8,)
    model = satellite_unet(IMAGE_DIMS)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    print('load data...')
    if test_run:
        # testing with one area
        trainX = np.load(file_X)
        trainY = np.load(file_Y)
    else:
        # for real with all areas, big data
        trainX = ld.load_entire_dataset(join(dir_temp_data_train, 'split'))
        trainY = ld.load_entire_dataset(join(dir_temp_labels, 'split'))


    trainX, trainY = dl.reshape_data(trainX, trainY)
    model, H = dl.init_and_fit_model(trainX, trainY, model, BS, EPOCHS, opt)
    dl.plot_info(H, EPOCHS)
