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
import cv2

import keras
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, MaxPool2D
from keras.layers.merge import add
from keras.activations import relu, softmax
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
K.set_image_data_format('channels_first')


PATH = '/home/jesimonsantos/datasets/sensor_RP/'
img_rows, img_cols, img_channel = 224, 224, 3


def rgb2gray(X):
    nX = []
    for j in X:
        nX.append(np.dot(j[...,:3], [0.2989, 0.5870, 0.1140]))

    return np.array(nX)
def resize(X):
    nX = []
    for sample in X:
        resized = cv2.resize(sample, (img_rows, img_cols))
        nX.append(resized)
    return np.array(nX)
    



def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not
    
    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        
        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            # identity
            f = x
        
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    
    return f

if __name__ == '__main__':
    np.random.seed(12227)
    batch = 50

    if len(sys.argv) > 2:
        data_input_file = sys.argv[1]
        img_rows = int(sys.argv[2])
        img_cols = int(sys.argv[3])
    else:
        data_input_file = PATH+'UTD2/utd.pickle'
        batch = 50

    data_input_file = PATH + data_input_file

    with open(data_input_file, 'rb') as handle:
        dataset = pickle.load(handle)
    
    X = np.array(dataset['X'])
    y = dataset['y']
    folds = dataset['folds']
    X = resize(X)
    X_shape = X.shape
    print('X_shape {}'.format(X_shape))
    X = X.reshape(X_shape[0],X_shape[-1], X_shape[1],X_shape[2])
    X_shape = X.shape

    print('X_shape {}'.format(X_shape))
    print('y {}'.format(y.shape))
    sys.stdout.flush()

    n_classes = y.shape[1]
    print('y {}'.format(n_classes))
    
    avg_acc = []
    avg_recall = []
    avg_f1 = []

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]

        # input tensor is the 49 grayscale image
        input_tensor = Input((img_channel,img_rows, img_cols))

        # first conv2d with post-activation to transform the input data to some reasonable form
        x = Conv2D(kernel_size=(3,3), filters=32, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        #224x224
        x = MaxPool2D()(x)
        #112x112
        x = block(32)(x)
        x = block(32)(x)
        x = MaxPool2D()(x)
        #56x56
        x = block(32)(x)
        x = MaxPool2D()(x)
        #28x28
        x = block(32)(x)
        x = MaxPool2D()(x)
        #14x14
        x = block(32)(x)
        x = block(32)(x)
        x = MaxPool2D()(x)
        #7x7
        # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
        # and we can't add together tensors of inconsistent sizes, so we use upscale=True
        # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_4
        # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
        # F_5
        # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

        # F_6
        # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
        # F_7
        # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

        # last activation of the entire network's output
        x = BatchNormalization()(x)
        x = Activation(relu)(x)
        # average pooling across the channels
        # 28x28x48 -> 1x48
        x = GlobalAveragePooling2D()(x)
        # dropout for more robust learning
        x = Dropout(0.2)(x)
        # last softmax layer
        x = Dense(units=n_classes, kernel_regularizer=regularizers.l2(0.01))(x)
        x = Activation(softmax)(x)

        model = Model(inputs=input_tensor, outputs=x)
        
        sys.stdout.flush()
        model.compile(loss='categorical_crossentropy',
            optimizer='RMSProp',
            metrics=['accuracy'])
        
        model.fit(X_train, y_train, batch, cm.n_ep, verbose=0,
                    callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                    validation_data=(X_train, y_train))
        print('Trained')
        sys.stdout.flush()
        # serialize model to JSON
        model_json = model.to_json()
        with open('./model_rpm.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('./model_rpm.h5')
        print("Saved model to disk")
        break

        y_pred = model.predict(X_test)
        
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
        sys.stdout.flush()

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))



