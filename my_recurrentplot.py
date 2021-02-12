import os  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pickle
import sys
import cv2


PATH = "/home/jesimonsantos/datasets/sensor_RP/"
NEWPATH = "/home/jesimonsantos/datasets/sensor_RP/"
SAVEPATH = "/home/jesimonsantos/datasets/sensor_RP/"



def Distance2dim(a,b):
    return pow(pow(a[1]-b[1],2)+pow(a[0]-b[0],2), 0.5)
def Cosin2vec(a,b):
    return (a[1]*b[1]+a[0]*b[0])/(pow(pow(a[1],2) + pow(a[0],2) , 0.5)*pow(pow(b[1],2) + pow(b[0],2) , 0.5)) 
def WeightAngle(a,b):
    return math.exp(2*(1.1 - Cosin2vec(a,b)))
def varRP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(150):
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(150):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(150):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
    return R

def RP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(150):
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(150):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(150):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = Distance2dim(s[i],s[j])
    return R
def RemoveZero(l):
    nonZeroL = []
    #nonZeroL = []
    for i in range(len(l)):
        if l[i] != 0.0:
            nonZeroL.append(l[i])
    return nonZeroL
#a = [0,-1,0.02,3]
#print RemoveZero(a)
def NormalizeMatrix(_r):
    dimR = _r.shape[0]
    h_max = []
    for i in range(dimR):
        h_max.append(max(_r[i]))
    _max =  max(h_max)
    h_min = []
    for i in range(dimR):
        #print _r[i]
        h_min.append(min(RemoveZero(_r[i])))
    
    _min =  min(h_min)
    _max_min = _max - _min
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = (_r[i][j]-_min)/_max_min
    return _normalizedRP
def RGBfromRPMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            _pixel.append(X[i][j])
            _pixel.append(Y[i][j])
            _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage
def SaveRP(x_array,y_array,z_array):
    _r = RP(x_array)
    _g = RP(y_array)
    _b = RP(z_array)
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
    plt.imshow(newImage)
    plt.savefig('D:\Datasets\ADL_Dataset\\'+action+'\\'+'RP\\''{}{}.png' .format(action, subject[15:]),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
def SaveRP_XYZ(x,action, normalized):
    _r = myRP(x,'x')
    _g = myRP(x,'y')
    _b = myRP(x,'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+str(action)+'{}.png'  .format(action),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(NEWPATH+str(action)+'{}.png'  .format(action),bbox_inches='tight',pad_inches = 0)
        plt.close('all')

def SavevarRP_XYZ(x,action, normalized):
    _r = varRP(x,'x')
    _g = varRP(x,'y')
    _b = varRP(x,'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(SAVEPATH+'train/'+action+'/'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        plt.imshow(newImage)
        plt.savefig(SAVEPATH+'train/'+action+'/'+'{}{}.png'  .format(action, i),bbox_inches='tight',pad_inches = 0)
        plt.close('all')

def myRP(data, dim):#dim:=x,y,z
    x = []
    if dim == 'x':
        x = data[:][0]
    elif dim == 'y':
        x = data[:][1]
    elif dim == 'z':
        x = data[:][2]

    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    dimR = len(x)-1
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = Distance2dim(s[i],s[j])
    return R

def myvarRP(inf, dim):
    x = []
    if dim == 'x':
        x = inf[:,0]
    elif dim == 'y':
        x = inf[:,1]
    elif dim == 'z':
        x = inf[:,2]
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)

    #print s
    dimR = len(x)-1
    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
    return R


def mySaveRP_XYZ(x, action, normalized, index,dataset):
    _r = myvarRP(x, 'x')
    _g = myvarRP(x, 'y')
    _b = myvarRP(x, 'z')
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        #plt.imshow(newImage)
        #plt.savefig(SAVEPATH+dataset+'/0.png'.format(index, action),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        #plt.imshow(newImage)
        #plt.savefig(SAVEPATH+dataset+'/0.png'.format(index, action),bbox_inches='tight',pad_inches = 0)
        plt.close('all')
    

    return newImage
    

def sampling_rate(data, rate_reduc):
    number_samp = int(data[0].shape[-2])
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = [data[0][:,:,samples_slct,:]]
    return new_data

samp_rate = 0

if len(sys.argv) > 1:
    data_input_file = sys.argv[1]
    samp_rate = int(sys.argv[2])

else:
    data_input_file = '/home/jesimonsantos/sensor/datasets/LOSO/UTD-MHAD2_1s.npz'

tmp = np.load(data_input_file, allow_pickle=True)
print('Loaded {} with success'.format(data_input_file))
sys.stdout.flush()

X = tmp['X']
y = tmp['y']
folds = tmp['folds']
dataset_name = data_input_file.split('/')[-1]

data = []
#read X hand crafted
#X_hand_craf = X[:, 0, :, :]

if dataset_name == 'MHEALTH.npz':
    data.append(X[:, :, :, 5:8]) # ACC right-lower-arm
    #data.append(X[:, :, :, 17:20]) # GYR right-lower-arm
    #data.append(X[:, :, :, 20:23]) # MAG right-lower-arm

elif dataset_name == 'PAMAP2P.npz':
    data.append(X[:, :, :, 4:7]) # ACC2 right-lower-arm
    #data.append(X[:, :, :, 7:10]) # GYR2 right-lower-arm
    #data.append(X[:, :, :, 10:13]) # MAG2 right-lower-arm

elif dataset_name == 'UTD-MHAD1_1s.npz':
    data.append(X[:, :, :, 0:3]) # ACC right-lower-arm
    #data.append(X[:, :, :, 3:6]) # GYR right-lower-arm

elif dataset_name == 'UTD-MHAD2_1s.npz':
    data.append(X[:, :, :, 0:3]) # ACC right-lower-arm
    #data.append(X[:, :, :, 3:6]) # GYR right-lower-arm

elif dataset_name == 'WHARF.npz':
    data.append(X[:, :, :, 0:3]) # ACC right-lower-arm

elif dataset_name == 'USCHAD.npz':
    data.append(X[:, :, :, 0:3]) # ACC right-lower-arm
    #data.append(X[:, :, :, 3:6]) # GYR right-lower-arm
            
elif dataset_name == 'WISDM.npz':
    data.append(X[:, :, :, 0:3]) # ACC right-lower-arm

n_class = y.shape[1]
#y = np.argmax(y, axis=1)
print(data[0].shape)
    
if samp_rate != 0:
    data = sampling_rate(data, samp_rate)

print(data[0].shape)

if not os.path.exists(SAVEPATH+dataset_name+'/'):
    os.makedirs(SAVEPATH+dataset_name+'/')

img_rows, img_cols, img_channel = 100, 100, 3

images = []
for n, sample in enumerate(data[0]):
    img = mySaveRP_XYZ(sample[0], y[n], 1, n, dataset_name)
    resized = cv2.resize(img, (img_rows, img_cols))
    images.append(resized)
    

print(img.shape)

a = {'X':images, 'y':y, 'folds':folds}

with open(PATH+dataset_name+'/'+str(samp_rate)+'_data.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Finish {}'.format(data_input_file))
sys.stdout.flush()