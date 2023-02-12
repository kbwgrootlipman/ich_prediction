# -*- coding: utf-8 -*-
"""
@author: Kevin Groot Lipman
"""
import sys
sys.path.insert(0, '/../analysis')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from datagenerator import DataGenerator
import pickle
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from resnet3d import Resnet3DBuilder
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from analysis import classification_analysis
import numpy as np


def train_test(partition, labels, fold):
    tf.set_random_seed(42)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    K.set_session(session)

    shape = (96, 256, 256)
    params = {'shape': shape,
              'n_classes': 3,
              'n_channels': 1}

    filename = '/../models/resnet'

    model = Resnet3DBuilder.build_resnet_50([shape[0], shape[1], shape[2], params['n_channels']],
                                            params['n_classes'],
                                            reg_factor=1e-3)

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    print(model.summary())

    training_generator = DataGenerator(partition['train'], labels, **params, augment=True, batch_size=8, shuffle=True)
    validation_generator = DataGenerator(partition['val'], labels, **params, augment=False,
                                         batch_size=len(partition['val']), shuffle=False)

    checkpoint_acc = ModelCheckpoint(filename + '_acc_fold{}.h5'.format(fold), monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='auto')
    checkpoint_loss = ModelCheckpoint(filename + '_loss_fold{}.h5'.format(fold), monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='auto')

    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=30, min_delta=0.001,
                       restore_best_weights=False)
    callbacks_list = [checkpoint_acc, checkpoint_loss, es]

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  validation_steps=len(partition['val']) // validation_generator.batch_size,
                                  epochs=100,
                                  verbose=1,
                                  use_multiprocessing=False,
                                  steps_per_epoch=20,
                                  workers=training_generator.batch_size,
                                  # max_queue_size=15,
                                  callbacks=callbacks_list)

    model.save(filename + '_final_fold{}.h5'.format(fold))

    ## Testing
    del model
    model = load_model(filename + '_loss_fold{}.h5'.format(fold), compile=False)
    model.compile('adam', 'binary_crossentropy', ['accuracy'])

    test_generator = DataGenerator(partition['test'], labels, batch_size=1, **params)

    test_pred_1 = model.predict_generator(generator=test_generator, steps=len(partition['test']))
    labels_test = [np.round(labels[k]) for k in partition['test']]

    results_he = classification_analysis(y_true=np.asarray([item[0] for item in labels_test]),
                                         y_pred=np.asarray([item[0] for item in test_pred_1]), repeats=10000)
    results_mrs = classification_analysis(y_true=np.asarray([item[1] for item in labels_test]),
                                          y_pred=np.asarray([item[1] for item in test_pred_1]), repeats=10000)
    results_mor = classification_analysis(y_true=np.asarray([item[2] for item in labels_test]),
                                          y_pred=np.asarray([item[2] for item in test_pred_1]), repeats=10000)

    with open('/../results/results_fold{}.pkl'.format(fold), 'wb') as file:
        pickle.dump([test_pred_1, labels_test, history.history, results_he, results_mrs, results_mor], file)
    K.clear_session()


if __name__ == '__main__':
    with open('/../partition/partition.pkl', 'rb') as file:
        ids, labels = pickle.load(file)

    he = [item[0] for item in labels.values()]
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    splits = {}
    for i, (train_index, test_index) in enumerate(skf.split(ids, he)):
        splits[i] = test_index

    for i in range(n_splits):
        partition = {'val': [ids[index] for index in splits[(i + 1) % n_splits]],
                     'test': [ids[index] for index in splits[i % n_splits]]}
        partition['train'] = [item for item in ids if item not in partition['val'] and item not in partition['test']]
        print('Fold {}:\nTrain: {}\nValidation: {}\nTest: {}'.format(i, len(partition['train']), len(partition['val']), len(partition['test'])))
        train_test(partition=partition, labels=labels, fold=i)
