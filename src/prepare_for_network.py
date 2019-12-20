#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:55:46 2019

@author: jurriaan
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import time



def prepare_data_for_nn_pipeline(file_hfd5, file_split_hfd5, size_sub_sample,
                                 stride_mode, name_dset, dtype, inv_stride,
                                 dataset_kind, depth_img=8):
    t0 = time.time()
    with h5py.File(file_split_hfd5, 'w') as f2:
        with h5py.File(file_hfd5, 'r') as f1:

            idx_i = 0
            meta_data = dict()

            for area in sorted(f1.keys()):
                print(area)
                dset = f1[area]
                print(dset.shape)

                for dim_str in list_stride(stride_mode):
                    print("dim_str:", dim_str)
                    np_arr_shape = list(dset.shape)
                    if 'x' in dim_str:
                        np_arr_shape[0] = np_arr_shape[0] + size_sub_sample[0]
                    if 'y' in dim_str:
                        np_arr_shape[1] = np_arr_shape[1] + size_sub_sample[1]

                    idx_j = int(np_arr_shape[0] * np_arr_shape[1] / (
                            size_sub_sample[0] * size_sub_sample[1])) + idx_i

                    if dataset_kind == 'labels':
                        np_arr_shape_split = (idx_j,) + size_sub_sample
                        maxshape = (None,) + size_sub_sample
                    elif dataset_kind == 'data':
                        np_arr_shape_split = (idx_j,) + size_sub_sample + (depth_img,)
                        maxshape = (None,) + size_sub_sample + (depth_img,)

                    if idx_i == 0:
                        dset2 = f2.create_dataset(name_dset,
                                                  np_arr_shape_split,
                                                  dtype=dtype,
                                                  maxshape=maxshape,
                                                  chunks=True)

                    if dataset_kind == 'labels':
                        dset2.resize((idx_j,) + size_sub_sample)
                        dset2[idx_i:idx_j, :, :] = split_array_2D(zero_padding_2(dset[:],
                                                                         dim_str,
                                                                         inv_stride,
                                                                         size_sub_sample),
                                                                  size_sub_sample[0],
                                                                  size_sub_sample[1])
                    elif dataset_kind == 'data':
                        for k in range(depth_img):
                            dset2.resize((idx_j,) + size_sub_sample + (depth_img,))
                            dset2[idx_i:idx_j, :, :, k] = split_array_2D(
                                zero_padding_2(dset[:, :, k],
                                               dim_str,
                                               inv_stride,
                                               size_sub_sample),
                                size_sub_sample[0],
                                size_sub_sample[1])
                    print("dset2.shape:", dset2.shape)
                    print('idx i,j:', idx_i, idx_j)
                    meta_data[area + ' ' + dim_str] = (idx_i, idx_j)
                    idx_i = idx_j

                f2.attrs.update(meta_data)

    print('takes {} mins'.format(round((time.time() - t0) / 60)))


def list_stride(mode):
    if mode == 1:
        return ['_']
    if mode == 2:
        return ['_', 'x', 'y', 'xy']




def zero_padding_1_labels(arr, sub_sample_shape):
    """make array dimension a multiple of sub_sample_shape
    """
    print('  padding 1')
    dim_x = arr.shape[0]
    dim_x_add = sub_sample_shape[0] - dim_x % sub_sample_shape[0]
    arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=0)

    dim_y = arr.shape[1]
    dim_y_add = sub_sample_shape[1] - dim_y % sub_sample_shape[1]
    arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add), dtype=np.uint16)
    arr = np.concatenate((arr, arr_zeros_stack), axis=1)
    print("   new size padding 1:", arr.shape)

    return arr


def zero_padding_2(arr, dim_str, inv_stride, size_sub_sample):
    """ padding array to fit stride
    """
    if dim_str:
        if 'x' in dim_str:
            dim_x_add = int(np.ceil(size_sub_sample[0] / inv_stride))
            arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=0)
            arr = np.concatenate((arr_zeros_stack, arr), axis=0)
        if 'y' in dim_str:
            dim_y_add = int(np.ceil(size_sub_sample[1] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=1)
            arr = np.concatenate((arr_zeros_stack, arr), axis=1)
    return arr


def zero_padding_2(arr, dim_str, inv_stride, size_sub_sample):
    """ padding array to fit stride
    """
    #print('padding 2:')
    #print(' dim_str:', dim_str)
    if dim_str:
        if 'x' in dim_str:
            dim_x_add = int(np.ceil(size_sub_sample[0] / inv_stride))
            arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=0)
            arr = np.concatenate((arr_zeros_stack, arr), axis=0)
        if 'y' in dim_str:
            dim_y_add = int(np.ceil(size_sub_sample[1] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add),
                                       dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=1)
            arr = np.concatenate((arr_zeros_stack, arr), axis=1)

    #print(" new size array:", arr.shape)
    return arr


def zero_padding_2_labels(arr, dim_str, inv_stride, size_sub_sample):
    """ padding array to fit stride
    """
    print('  padding 2:')
    print('    dim_str:', dim_str)
    if dim_str:
        if 'x' in dim_str:
            dim_x_add = int(np.ceil(size_sub_sample[0] / inv_stride))
            arr_zeros_stack = np.zeros((dim_x_add, arr.shape[1]), dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=0)
            arr = np.concatenate((arr_zeros_stack, arr), axis=0)

        if 'y' in dim_str:
            dim_y_add = int(np.ceil(size_sub_sample[1] / inv_stride))
            arr_zeros_stack = np.zeros((arr.shape[0], dim_y_add), dtype=np.uint16)
            arr = np.concatenate((arr, arr_zeros_stack), axis=1)
            arr = np.concatenate((arr_zeros_stack, arr), axis=1)

    print("   new size array:", arr.shape)
    return arr


def split_array_2D(array, nrows, ncols):
    """Split a matrix into sub-matrices of shape (nrows, ncols)"""
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


def visualize_sample(np_arr, poly_mask, figsize=(10, 10), linewidth=1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(poly_mask, 1, colors='red', linewidth=linewidth)
    ax.imshow(np_arr)
