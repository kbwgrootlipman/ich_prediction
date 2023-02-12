# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:05:11 2019

@author: Kevin
"""

import numpy as np
import keras
import random
from skimage.transform import resize
from keras.utils import to_categorical
import SimpleITK as sitk
from scipy.ndimage import rotate


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=2, shape=(96, 160, 160), n_channels=1,
                 n_classes=2, shuffle=True, modality='de', location='server', augment=True, val='train'):
        'Initialization'
        self.shape = shape
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.modality = modality
        self.augment = augment
        self.val = val
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_augmentation(self, x, i, rot):
        if self.augment:
            if i % 2:
                x = rotate(x, rot, axes=(1, 2))
            if i % 3:
                x = np.flip(x, axis=(0, 2))
        max_dim = np.max((x.shape[1], x.shape[2]))

        zoom_z = [self.shape[0] if x.shape[0] > self.shape[0] else x.shape[0]][0]
        zoom_y = [int(np.round(x.shape[1] * self.shape[1] / max_dim)) if max_dim > self.shape[1] else x.shape[1]][0]
        zoom_x = [int(np.round(x.shape[2] * self.shape[2] / max_dim)) if max_dim > self.shape[2] else x.shape[2]][0]
        x = resize(x, (zoom_z, zoom_y, zoom_x), preserve_range=True)

        z_pad = self.shape[0] - x.shape[0]
        y_pad = self.shape[1] - x.shape[1]
        x_pad = self.shape[1] - x.shape[2]

        x = np.pad(x, ((int(np.floor(z_pad / 2)), int(np.ceil(z_pad / 2))),
                       (int(np.floor(y_pad / 2)), int(np.ceil(y_pad / 2))),
                       (int(np.floor(x_pad / 2)), int(np.ceil(x_pad / 2)))),
                   'constant', constant_values=0)
        return x

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.shape, self.n_channels))
        y = np.empty((self.batch_size, 3))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            x_iod = sitk.GetArrayFromImage(
                sitk.ReadImage('/../' + str(ID) + '_iod.nii.gz'))
            x_seg = sitk.GetArrayFromImage(
                sitk.ReadImage('/../' + str(ID) + '_label.nii.gz'))

            rot = random.choice(range(-20, 20))
            x = x_iod * x_seg
            x = x.astype(np.float32)
            x = self.data_augmentation(x, i, rot=rot)
            x += -np.min(x)
            x /= np.max(x)
            X[i, :, :, :, 0] = x

            y[i, 0] = self.labels[ID][0]
            y[i, 1] = self.labels[ID][1]
            y[i, 2] = self.labels[ID][2]
        return X, y
