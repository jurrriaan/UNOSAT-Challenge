#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:55:46 2019

@author: jurriaan
"""
import gc
import numpy as np
import matplotlib.pyplot as plt
import os

from os.path import join


import src.load_data as ld
import src.preprocessing as pp


def pipeline_prepare_for_nn(area_file, dir_data, inv_stride, size_sub_sample):
    """Creates np array data ready for direct network input
    """
    arr_area = np.load(join(dir_data, 'whole', area_file))
    arr_area = zero_padding_1(arr_area, size_sub_sample)

    for dim_str in ['', 'x', 'y', 'xy']:
        arr_area_2 = zero_padding_2(arr_area, dim_str, inv_stride, size_sub_sample)
        arr_area_2 = split_array_3D(arr_area_2, size_sub_sample[0], size_sub_sample[1])
        print(dim_str, 'shape after split:', arr_area_2.shape)
        save_array_in_new_sub_dir(arr_area_2, dir_data, 'split', area_file, dim_str)
        del arr_area_2
        gc.collect()
    del arr_area
    gc.collect()


def pipeline_prepare_for_nn_labels(area_file, dir_data, inv_stride, size_sub_sample):
    """Creates np array data ready for direct network input
    """
    arr_area = np.load(join(dir_data, 'whole', area_file))
    arr_area = zero_padding_1_labels(arr_area, size_sub_sample)

    for dim_str in ['', 'x', 'y', 'xy']:
        arr_area_2 = zero_padding_2_labels(arr_area, dim_str, inv_stride, size_sub_sample)
        arr_area_2 = split_array_2D(arr_area_2, size_sub_sample[0], size_sub_sample[1])
        print(dim_str, 'shape after split:', arr_area_2.shape)
        save_array_in_new_sub_dir(arr_area_2, dir_data, 'split', area_file, dim_str)
        del arr_area_2
        gc.collect()
    del arr_area
    gc.collect()


def zero_padding_1(arr, sub_sample_shape):
    """make array dimension a multiple of sub_sample_shape
    """
    print(arr.shape)
    dim_x = arr.shape[1]
    dim_x_add = sub_sample_shape[0] - dim_x % sub_sample_shape[0]
    arr_zeros_stack = np.zeros((arr.shape[0], dim_x_add, arr.shape[2]), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=1)

    dim_y = arr.shape[2]
    dim_y_add = sub_sample_shape[1] - dim_y % sub_sample_shape[1]
    arr_zeros_stack = np.zeros((arr.shape[0], arr.shape[1], dim_y_add), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=2)
    print("new size padding 1:", arr.shape)

    return arr


def zero_padding_1_labels(arr, sub_sample_shape):
    """make array dimension a multiple of sub_sample_shape
    """
    dim_x = arr.shape[0]
    dim_x_add = sub_sample_shape[0] - dim_x % sub_sample_shape[0]
    arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=0)

    dim_y = arr.shape[1]
    dim_y_add = sub_sample_shape[1] - dim_y % sub_sample_shape[1]
    arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=1)
    print("new size padding 1:", arr.shape)

    return arr


def zero_padding_2(arr, dim_str, inv_stride, size_sub_sample):
    """ padding array to fit stride
    """
    print('padding 2:')
    print(' dim_str:', dim_str)
    if dim_str:
        if 'x' in dim_str:
            print('x1:', arr.shape)
            dim_x_add = int(np.ceil(size_sub_sample[0] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], dim_x_add, arr.shape[2]),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=1)
            arr = np.concatenate((arr_zeros_stack, arr), axis=1)
            print('x2:', arr.shape)
        
        if 'y' in dim_str:
            dim_y_add = int(np.ceil(size_sub_sample[1] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], arr.shape[1], dim_y_add),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=2)
            arr = np.concatenate((arr_zeros_stack, arr), axis=2)

    print(" new size array:", arr.shape)
    return arr


def zero_padding_2_labels(arr, dim_str, inv_stride, size_sub_sample):
    """ padding array to fit stride
    """
    print('padding 2:')
    print(' dim_str:', dim_str)
    if dim_str:
        if 'x' in dim_str:
            print('x1:', arr.shape)
            dim_x_add = int(np.ceil(size_sub_sample[0] / inv_stride))
            arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]), dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=0)
            arr = np.concatenate((arr_zeros_stack, arr), axis=0)
            print('x2:', arr.shape)
        
        if 'y' in dim_str:
            dim_y_add = int(np.ceil(size_sub_sample[1] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add), dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=1)
            arr = np.concatenate((arr_zeros_stack, arr), axis=1)

    print(" new size array:", arr.shape)
    return arr


def save_array_in_new_sub_dir(arr, main_dir, name_new_dir, filename, dim_str):
    print('save_array_in_new_sub_dir')
    name_new_dir = 'split'
    dir_data_split = join(main_dir, name_new_dir) 
    ld.make_dir(dir_data_split)
    path_file_split = join(dir_data_split, os.path.splitext(filename)[0] + '_' + name_new_dir + '_' + dim_str)
    np.save(path_file_split, arr)
    del arr
    gc.collect()


def split_array(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    h = array.shape[-1]
    r = array.shape[-2]
    
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
    

def split_array_2D(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    w, r, h = array.shape
    
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
    

def split_array_3D(array, nrows, ncols):
    """Split a 3D array into sub-arrays."""
    
    w, r, h = array.shape
    list_3D_arrs = []
        
    for i in range(w):
        arr_2D = array[i, :, :]

        list_3D_arrs.append(arr_2D.reshape(h//nrows, nrows, -1, ncols)
                                     .swapaxes(1, 2)
                                     .reshape(-1, nrows, ncols))
            
    arr_4D_resh1 = np.array(list_3D_arrs)
    return arr_4D_resh1.reshape(arr_4D_resh1.shape[1], arr_4D_resh1.shape[0], 
                                arr_4D_resh1.shape[2], arr_4D_resh1.shape[3])


def zero_padding(dict_data_train, sub_sample_shape):
    #dict_data_train_padded = dict()
    
    for area in dict_data_train:
        np_arr_area = dict_data_train[area]
    
        dim_x = np_arr_area.shape[1]
        dim_x_add = sub_sample_shape[0] - dim_x % sub_sample_shape[0]
        arr_zeros_stack = np.zeros((np_arr_area.shape[0], 
                                    dim_x_add, np_arr_area.shape[2]))
        np_arr_area = np.concatenate((np_arr_area, arr_zeros_stack), axis=1)
        
        dim_y = np_arr_area.shape[2]
        dim_y_add = sub_sample_shape[1] - dim_y % sub_sample_shape[1]
        arr_zeros_stack = np.zeros((np_arr_area.shape[0], np_arr_area.shape[1], 
                                    dim_y_add))
        np_arr_area = np.concatenate((np_arr_area, arr_zeros_stack), axis=2)
        print(area, "new size:", np_arr_area.shape)
        dict_data_train[area] = np_arr_area
        del np_arr_area
        gc.collect()
        
    return dict_data_train
    

def visualize_sample(np_arr, poly_mask, figsize=(10,10), linewidth=1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(poly_mask, 1, colors='red', linewidth=linewidth)
    ax.imshow(np_arr)