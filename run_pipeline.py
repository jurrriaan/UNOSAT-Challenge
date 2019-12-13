#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:27:07 2019

@author: jurriaan
"""

import sys
sys.executable

import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings

# see: https://github.com/karolzak/keras-unet
from keras.optimizers import Adam
from keras_unet.models import satellite_unet 
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.utils import get_augmented
from keras.callbacks import ModelCheckpoint

import load_data as ld
import preprocessing as pp
import prepare_for_network as pfn

warnings.simplefilter("ignore")


# local paths, set your own path here
path_data_main = '/Volumes/other/datasets_and_ML/UNOSAT_Challenge'


# put your data in these dirs
path_data_local_train = join(path_data_main, 'Train_Dataset')
path_data_local_val = join(path_data_main, 'Validation_Dataset')
path_data_local_test = join(path_data_main, 'Evaluation_Dataset')

# other dirs are made automatically
dir_temp_data =  ld.make_dir(join(path_data_main, 'data_temp'))
dir_temp_data_train = ld.make_dir(join(dir_temp_data, 'train'))
dir_temp_data_test = ld.make_dir(join(dir_temp_data, 'test'))
dir_temp_plots = ld.make_dir(join(dir_temp_data, 'plots'))
dir_temp_labels = ld.make_dir(join(dir_temp_data, 'labels'))

# PARAMETERS
#size_sub_sample = (256, 256)
quantile_clip_max = 0.999
size_sub_sample = (512, 512)
inv_stride = 2
EPOCHS = 1
INIT_LR = 1e-3
BS = 32


# LOAD AND PREORCESS DATA 
dict_info_data_train  = ld.paths_in_dict(path_data_local_train)

dict_raster_layers_train, dict_data_train = pp.preprocess_data(dict_info_data_train, dir_temp_plots, quantile_clip_max)
dict_labels = pp.get_label_data(dict_info_data_train, dict_raster_layers_train, dir_temp_labels)
list_paths_data_train = pp.save_stacked_arrays(dict_info_data_train, dir_temp_data_train)
list_paths_labels = pp.save_labels(dict_labels, dir_temp_labels)


#_, dict_data_test = pp.preprocess_data(dict_info_data_train, dir_temp_plots, quantile_clip_max)
#list_paths_data_test = pp.save_stacked_arrays(dict_data_test, dir_temp_data_train)


# PREPARE FOR NETWORK 
dir_temp_data_train_whole = ld.make_dir(join(dir_temp_data_train, 'whole'))
dir_temp_labels_whole = ld.make_dir(join(dir_temp_labels, 'whole'))

list_dir_np = [f for f in os.listdir(dir_temp_data_train_whole) if os.path.splitext(f)[1] == '.npy']
list_dir_np_label = [f for f in os.listdir(dir_temp_labels_whole) if os.path.splitext(f)[1] == '.npy']

for area_data_file, area_label_file in zip(list_dir_np, list_dir_np_label):
    print(area_data_file, area_label_file)
    pfn.pipeline_prepare_for_nn(area_data_file, dir_temp_data_train, inv_stride, size_sub_sample)
    pfn.pipeline_prepare_for_nn_labels(area_label_file, dir_temp_labels, inv_stride, size_sub_sample)
    
    
# DEEP LEARNING
IMAGE_DIMS =  size_sub_sample + (8,)
model = satellite_unet(IMAGE_DIMS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# load data
trainX = np.array([])
trainY = np.array([])

# testing with one area
file_X = '/Volumes/other/datasets_and_ML/UNOSAT_Challenge/data_temp/train/split/Mosul_split_.npy'
fily_Y = '/Volumes/other/datasets_and_ML/UNOSAT_Challenge/data_temp/labels/split/Mosul_split_.npy'
trainX = np.load(file_X)
trainY = np.load(fily_Y)

# for real with all areas, big data


trainX = trainX.reshape(trainX.shape[0], trainX.shape[2], trainX.shape[3], trainX.shape[1])
trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], trainY.shape[2], 1)


train_datagen = get_augmented(
                                trainX, 
                                trainY, 
                                X_val=None,
                                Y_val=None,
                                batch_size=BS, 
                                seed=0, 
                                data_gen_args = dict(
                                    rotation_range=90,
                                    height_shift_range=0,
                                    shear_range=0,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant'
                                )
)
                                
                                
model.compile(loss="binary_crossentropy", 
              optimizer=opt, 
              #metrics=["accuracy"], 
              metrics=[iou, iou_thresholded])

# checkpoint
filepath="weights-improvement-{epoch:02d}-{iou:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='iou', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


H = model.fit_generator(generator=train_datagen,
                    steps_per_epoch=len(trainX) // BS,
                    epochs=EPOCHS,
                    callbacks=callbacks_list
    )

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_epochs.h5")

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["iou"], label="train_iou")

plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/iou")
plt.legend(loc="lower left")
plt.savefig("plot")

