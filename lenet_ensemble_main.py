import numpy as np                  # for working with tensors outside the network
#import pandas as pd                 # for data reading and writing
#import matplotlib.pyplot as plt     # for data inspection
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os  
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.pylab import rcParams
import math
import pickle
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm
import cv2, json
import tensorflow as tf
import keras
from keras.layers import Flatten,Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, SeparableConv2D
from keras.layers.merge import add
from keras.regularizers import l2
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
#import matplotlib.image as mpimg
from keras.models import model_from_json
import json
from scipy.interpolate import interp1d
from scipy import signal
from scipy.fft import fftshift
#import matplotlib.pyplot as plt
K.set_image_data_format('channels_first')
from sklearn.metrics import plot_confusion_matrix

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

def sampling_rate(data, rate_reduc):
    number_samp = int(data[0].shape[-2])
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = [data[0][:,:,samples_slct,:]]
    return new_data

def resize(X):
    nX = []
    for sample in X:
        resized = cv2.resize(sample, (img_rows, img_cols))
        nX.append(resized)
    return np.array(nX)


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

    X_spc = generateSpec(data_sig)
    shp = X_spc.shape
    X_spc = X_spc.reshape(shp[0],1, shp[-1], shp[1])
    shp = X_spc.shape

    if samp_rate != 0:
        data_sig = sampling_rate(data_sig, samp_rate)
    
    #y = np.argmax(y, axis=1)
    print('folds {}'.format(len(folds)))
    drop_ = 0.6
    y_pred_total = []
    y_true_total = []

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]

        X_train = X[train_idx]
        X_test = X[test_idx]
        X_train_spc = X_spc[train_idx]
        X_test_spc = X_spc[test_idx]
        y_train = y[train_idx]
        X_train_sig = data_sig[0][train_idx]
        X_test_sig = data_sig[0][test_idx]

        shape_x = X_train_sig.shape
        print(shape_x)

        #img_channel, img_rows, img_cols = shape_x[1], shape_x[2], shape_x[3]

        input_tensor = Input((img_channel, img_rows, img_cols))
        input_tensor_spc = Input((shp[1], shp[2], shp[3]))
        input_tensor_sig = Input((shape_x[1], shape_x[2], shape_x[3]))
        
        hidden = SeparableConv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal',padding='same')(input_tensor)
        hidden = MaxPooling2D(pool_size=(2, 1))(hidden)
        hidden = SeparableConv2D(filters=16, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden)
        hidden = MaxPooling2D(pool_size=(2, 1))(hidden)
        hidden = Flatten()(hidden)
        #hidden = GlobalAveragePooling2D()(hidden)
        out = Dense(200, activation='relu')(hidden)
        out = Dropout(drop_)(out)
        out = Dense(200, activation='relu')(out)
        out = Dropout(drop_)(out)
        #out = Dense(units=n_classes, activation='softmax')(hidden)

        hidden_spc = SeparableConv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal',padding='same')(input_tensor_spc)
        hidden_spc = MaxPooling2D(pool_size=(2, 1))(hidden_spc)
        hidden_spc = SeparableConv2D(filters=16, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden_spc)
        hidden_spc = Flatten()(hidden_spc)
        #hidden_spc = GlobalAveragePooling2D()(hidden_spc)
        out_spc = Dense(200, activation='relu')(hidden_spc)
        out_spc = Dropout(drop_) (out_spc)
        out_spc = Dense(200, activation='relu')(out_spc)
        out_spc = Dropout(drop_) (out_spc)
        #out_spc = Dense(units=n_classes, activation='softmax')(hidden_spc)

        hidden_sig = SeparableConv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal',padding='same')(input_tensor_sig)
        hidden_sig = MaxPooling2D(pool_size=(2, 1))(hidden_sig)
        hidden_sig = SeparableConv2D(filters=16, kernel_size=(2,2), activation='relu', kernel_initializer='glorot_normal', padding='same')(hidden_sig)
        hidden_sig = MaxPooling2D(pool_size=(2, 1))(hidden_sig)
        hidden_sig = Flatten()(hidden_sig)
        out_sig = Dense(200, activation='relu')(hidden_sig)
        out_sig = Dropout(drop_) (out_sig)
        out_sig = Dense(200, activation='relu')(out_sig)
        out_sig = Dropout(drop_) (out_sig)
        #out_sig = Dense(units=n_classes, activation='softmax')(out_sig)

        concat = keras.layers.concatenate([out,out_spc, out_sig], axis=-1)
        
        new_out = Dense(200, activation='relu')(concat)
        new_out = Dropout(drop_) (new_out)
        new_out = Dense(200, activation='relu')(new_out)
        new_out = Dropout(drop_) (new_out)
        new_out = Dense(units=n_classes, activation='softmax')(new_out)

        model_ft = Model(inputs=[input_tensor, input_tensor_spc, input_tensor_sig], outputs=new_out)
        #model_ft.summary()
        sys.stdout.flush()

        model_ft.compile(loss='categorical_crossentropy',
            optimizer='RMSProp',
            metrics=['accuracy'])

        
        hst = model_ft.fit([X_train, X_train_spc, X_train_sig], y_train, batch, cm.n_ep, verbose=0,
                    callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                    validation_data=([X_train,X_train_spc, X_train_sig], y_train))


        y_pred = model_ft.predict([X_test, X_test_spc, X_test_sig])


        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y[test_idx], axis=1)

        if i == 0:
            y_pred_total = y_pred
            y_true_total = y_true
        else:
            y_pred_total = np.concatenate((y_pred_total, y_pred))
            y_true_total = np.concatenate((y_true_total, y_true))

        acc_fold = accuracy_score(y_true, y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y_true, y_pred, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y_true, y_pred, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        print('______________________________________________________')
        sys.stdout.flush()
        #del model_ft
    
    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]\n\n'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
    print(y_true_total)
    print(y_pred_total)
    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(y_true_total, y_pred_total))

    