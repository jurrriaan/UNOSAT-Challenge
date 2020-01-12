
import h5py
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D, Input, concatenate

import utils as utl


class My_Custom_Generator(Sequence):
    def __init__(self, file_hdf5_X, file_hdf5_Y, batch_size, name_dset_X, name_dset_Y,
                 dtype):
        self.file_hdf5_X = file_hdf5_X
        self.file_hdf5_Y = file_hdf5_Y
        self.batch_size = batch_size
        self.name_dset_X = name_dset_X
        self.name_dset_Y = name_dset_Y
        self.dtype = dtype

    def __len__(self):
        with h5py.File(self.file_hdf5_Y, 'r') as f1_Y:
            dataset_Y = f1_Y[self.name_dset_Y]
            len_dataset_Y = dataset_Y.shape[0]

        return (np.ceil(len_dataset_Y) / float(self.batch_size)).astype(np.int)

    def __getitem__(self, idx):
        with h5py.File(self.file_hdf5_X, 'r') as f1_X:
            dataset_X = f1_X[self.name_dset_X]
            batch_x = dataset_X[idx * self.batch_size: (idx + 1) * self.batch_size, :, :,
                      :] / utl.get_nmax_dtype(self.dtype)

        with h5py.File(self.file_hdf5_Y, 'r') as f1_Y:
            dataset_Y = f1_Y[self.name_dset_Y]
            batch_y = dataset_Y[idx * self.batch_size: (idx + 1) * self.batch_size, :, :]

        return (batch_x,
                batch_y.reshape(batch_y.shape[0], batch_y.shape[1], batch_y.shape[2], 1))


def get_init_epoch(filepath_chk):
    return int(filepath_chk.split('-')[2])


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


# https://cdn-sv1.deepsense.ai/wp-content/uploads/2017/04/architecture_details.png
# https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
def satellite_unet(
        input_shape,
        num_classes=1,
        output_activation='sigmoid',
        num_layers=4):
    inputs = Input(input_shape)

    filters = 64
    upconv_filters = 96

    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    conv2d_trans_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': (2, 2),
        'padding': padding,
        'output_padding': (1, 1)
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):
        x = concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1),
                     activation=output_activation, padding='valid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)


def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

