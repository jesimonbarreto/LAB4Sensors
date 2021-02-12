import numpy as np                  # for working with tensors outside the network
import pandas as pd                 # for data reading and writing
import matplotlib.pyplot as plt     # for data inspection
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math
import pickle
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm
import cv2, json
import tensorflow as tf
import keras
from keras.layers import Flatten,Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
from keras.layers.merge import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
import matplotlib.image as mpimg
from keras.models import model_from_json
import json
from scipy.interpolate import interp1d
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
K.set_image_data_format('channels_first')

import keras
import numpy as np

import tensorflow as tf

def channel_split(a, b):
    return lambda x: x[:, :, :, a:b]

def channel_shuffle(size):
    idx = np.arange(size)
    np.random.shuffle(idx)
    return lambda x: tf.gather(x, idx, axis=-1)

def ShuffleA(n):
    prefix = 'shufflev2_{:d}a'.format(n)
    def _Shufflev2A(inputs):
        end = inputs.shape.as_list()[-1]
        mid = end//2
        c0 = keras.layers.Lambda(channel_split(0, mid), name=prefix+'/split0')(inputs)
        c1 = keras.layers.Lambda(channel_split(mid, end), name=prefix+'/split1')(inputs)
        c1 = keras.layers.Convolution2D(mid, (1, 1), padding='same', name=prefix+'/conv')(c1)
        c1 = keras.layers.BatchNormalization(name=prefix+'/bn0')(c1)
        c1 = keras.layers.DepthwiseConv2D((3, 3), padding='same', name=prefix+'/dwise')(c1)
        c1 = keras.layers.BatchNormalization(name=prefix+'/bn1')(c1)
        c1 = keras.layers.Activation('relu', name=prefix+'/act')(c1)
        concat = keras.layers.Concatenate(name=prefix+'/concat')([c0, c1])
        return keras.layers.Lambda(channel_shuffle(end), name=prefix+'/shuffle')(concat)
    return _Shufflev2A

def ShuffleB(n):
    prefix = 'shufflev2_{:d}b'.format(n)
    def _Shufflev2B(inputs):
        mid = end = inputs.shape.as_list()[-1]
        # left branch
        c0 = inputs
        c0 = keras.layers.DepthwiseConv2D((3, 3), 2, padding='same', name=prefix+'/dwise0')(c0)
        c0 = keras.layers.BatchNormalization(name=prefix+'/bn0')(c0)
        c0 = keras.layers.Convolution2D(mid, (1, 1), padding='same', name=prefix+'/conv0')(c0)
        c0 = keras.layers.Activation('relu', name=prefix+'/act0')(c0)
        # right branch
        c1 = inputs
        c1 = keras.layers.Convolution2D(mid, (1, 1), padding='same', name=prefix+'/conv1')(c1)
        c1 = keras.layers.BatchNormalization(name=prefix+'/bn1')(c1)
        c1 = keras.layers.DepthwiseConv2D((3, 3), 2, padding='same', name=prefix+'/dwise1')(c1)
        c1 = keras.layers.BatchNormalization(name=prefix+'/bn2')(c1)
        c1 = keras.layers.Activation('relu', name=prefix+'/act1')(c1)
        concat = keras.layers.Concatenate(name=prefix+'/concat')([c0, c1])
        return keras.layers.Lambda(channel_shuffle(2*end), name=prefix+'/shuffle')(concat)
    return _Shufflev2B

def shufflenetv2(inputs, filters, n_classes):
    #inputs = keras.layers.Input(input_size)
    x = keras.layers.Conv2D(filters, (3, 3), name='conv0', activation='relu', padding='same')(inputs)

    x = ShuffleA(0)(x)

    x = ShuffleB(1)(x)
    x = ShuffleA(2)(x)
    x = ShuffleA(3)(x)

    x = ShuffleB(4)(x)
    #x = ShuffleA(5)(x)
    #x = ShuffleA(6)(x)
    #x = ShuffleA(7)(x)
    #x = ShuffleA(8)(x)

    #x = ShuffleB(9)(x)
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    out = Dense(200, activation='relu')(x)
    out = Dense(units=n_classes, activation='softmax')(out)
    
    return keras.models.Model(inputs=inputs, outputs=out)


def resize(X):
    nX = []
    for sample in X:
        resized = cv2.resize(sample, (img_rows, img_cols))
        nX.append(resized)
    return np.array(nX)



def generateSpec(data):
    X_spec = []
    for spl in data[0]:
        x_acc = spl[0,:,0]
        y_acc = spl[0,:,1]
        z_acc = spl[0,:,2]
        fx, tx, Sxx = signal.spectrogram(x_acc, 1)
        fy, ty, Sxy = signal.spectrogram(y_acc, 1)
        fz, tz, Sxz = signal.spectrogram(z_acc, 1)
        Sx = np.concatenate((Sxx,Sxy,Sxz),axis=-1)
        X_spec.append(Sx)
    #normaliz?
    return np.array(X_spec)



img_channel, img_rows, img_cols = 3, 64, 64
PATH = '/home/jesimonsantos/datasets/sensor_RP/'



if __name__ == '__main__':
    np.random.seed(12227)

    batch = 100
    samp_rate = 0
    tp = 'linear'
    amp_rate_2 = 0 
    data_input_file_2 = '/home/jesimonsantos/sensor/datasets/LOSO/USCHAD.npz'#'USCHAD.npz/data.pickle'
    basemodel = '/home/jesimonsantos/sensor/datasets/LOSO/USCHAD.npz'

    if len(sys.argv) > 2:
        data_input_file = sys.argv[1]
        data_input_file_2 = sys.argv[2]
        #batch = int(sys.argv[3])


    data_input_file_2 = PATH + data_input_file_2

    tmp = np.load(data_input_file, allow_pickle=True)
    print('\n\nLoaded {} with success and rate reduction {}\n'.format(data_input_file, samp_rate))
    print('Reduction rate: {}'.format(samp_rate))
    sys.stdout.flush()

    X_sig = tmp['X']
    y_sig = tmp['y']
    folds = tmp['folds']
    dataset_name = data_input_file.split('/')[-1]

    data_sig = []
    #read X hand crafted
    #X_hand_craf = X[:, 0, :, :]

    print('\nDataset {}\n\n'.format(data_input_file_2))
    
    if dataset_name == 'MHEALTH.npz':
        data_sig.append(X_sig[:, :, :, 5:8])

    elif dataset_name == 'PAMAP2P.npz':
        data_sig.append(X_sig[:, :, :, 4:7])

    elif dataset_name == 'UTD-MHAD1_1s.npz':
        data_sig.append(X_sig[:, :, :, 0:3]) # ACC right-lower-arm
        #data.append(X[:, :, :, 3:6]) # GYR right-lower-arm

    elif dataset_name == 'UTD-MHAD2_1s.npz':
        data_sig.append(X_sig[:, :, :, 0:3]) # ACC right-lower-arm
        #data.append(X[:, :, :, 3:6]) # GYR right-lower-arm

    elif dataset_name == 'WHARF.npz':
        data_sig.append(X_sig[:, :, :, 0:3]) # ACC right-lower-arm

    elif dataset_name == 'USCHAD.npz':
        #data.append(X[:, :, :, 0:3]) # ACC right-lower-arm
        data_sig.append(X_sig[:, :, :, 3:6]) # GYR right-lower-arm
                
    elif dataset_name == 'WISDM.npz':
        data_sig.append(X_sig[:, :, :, 3:6]) # ACC right-lower-arm

    n_classes = y_sig.shape[1]
    #y = np.argmax(y, axis=1)

    X = []
    y = []
    tmp = []
    folds = []
    dataset_name = ''

    with open(data_input_file_2, 'rb') as handle_2:
        dataset = pickle.load(handle_2)
    
    X = np.array(dataset['X'])
    y = dataset['y']
    folds = dataset['folds']
    X_comp = []
    for fig in X:
        resized = cv2.resize(fig, (64, 64))
        X_comp.append(resized)
    
    X = np.array(X_comp)
    #X = resize(X)

    #X = rgb2gray(X)
    X_shape = X.shape
    print('X_shape {}'.format(X_shape))
    X = X.reshape(X_shape[0],X_shape[-1], X_shape[1],X_shape[2])
    X_shape = X.shape

    print('X_shape {}'.format(X_shape))
    print('y {}'.format(X_shape))
    sys.stdout.flush()

    n_classes = y.shape[1]
    print('y {}'.format(n_classes))

    avg_acc = []
    avg_recall = []
    avg_f1 = []

    #y = np.argmax(y, axis=1)
    print('folds {}'.format(len(folds)))

    #X = generateSpec(data_sig)
    #shp = X.shape
    #X = X.reshape(shp[0],1, shp[-1], shp[1])
    #X_shape = X.shape

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        X_train_sig = data_sig[0][train_idx]
        X_test_sig = data_sig[0][test_idx]

        shape_x = X_train_sig.shape
        print(X_train.shape)
        print(X_test.shape)
        #img_channel, img_rows, img_cols = shape_x[1], shape_x[2], shape_x[3]
        input_tensor = Input((X_shape[1], X_shape[2], X_shape[3]))
        #input_tensor = Input((img_channel, img_rows, img_cols))
        #input_tensor_sig = Input((shape_x[1], shape_x[2], shape_x[3]))


        
        '''hidden = Conv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal',padding='same')(input_tensor)
        hidden = MaxPooling2D(pool_size=(2, 1))(hidden)
        hidden = Conv2D(filters=16, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 1))(hidden)
        hidden = Flatten()(hidden)'''

        '''hidden_sig = Conv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal',padding='same')(input_tensor_sig)
        hidden_sig = MaxPooling2D(pool_size=(2, 1))(hidden_sig)
        hidden_sig = Conv2D(filters=16, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden_sig)
        hidden_sig = MaxPooling2D(pool_size=(2, 1))(hidden_sig)
        hidden_sig = Flatten()(hidden_sig)'''

        '''out = Dense(200, activation='relu')(hidden)
        out = Dense(200, activation='relu')(out)
        out = Dense(units=n_classes, activation='softmax')(out)'''

        #out_sig = Dense(200, activation='relu')(hidden_sig)
        #out_sig = Dense(200, activation='relu')(out_sig)
        #out_sig = Dense(units=n_classes, activation='softmax')(out_sig)

        #concat = keras.layers.concatenate([out, out_sig], axis=-1)
        
        #new_out = Dense(200, activation='relu')(concat)
        #new_out = Dense(200, activation='relu')(new_out)
        #new_out = Dense(units=n_classes, activation='softmax')(new_out)

        #model_ft = Model(inputs=input_tensor, outputs=out)

        model_ft = shufflenetv2(input_tensor, 32, n_classes)

        model_ft.compile(loss='categorical_crossentropy',
            optimizer='RMSProp',
            metrics=['accuracy'])

        
        hst = model_ft.fit(X_train, y_train, batch, cm.n_ep, verbose=0,
                    callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                    validation_data=(X_train, y_train))


        y_pred = model_ft.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y[test_idx], axis=1)

        acc_fold = accuracy_score(y_true, y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y_true, y_pred, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y_true, y_pred, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        print('______________________________________________________')
        #del model_ft
    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))