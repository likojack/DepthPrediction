caffe_root = '/home/nico/caffe/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
import numpy as np
from pylab import *
import glob
import pickle
import os
#path = "/home/nico/DepthPrediction/train_1/"

def normalise_x(x):
    return abs(x-320)/320.0
def normalise_y(y):
    return y/480.0

with open("/home/nico/DepthPrediction/xy_extension/images_train.txt") as F1:
    images = F1.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/label_train.txt") as F2:
    label = F2.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/x_train.txt") as F3:
    x = F3.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/y_train.txt") as F4:
    y = F4.read().splitlines()

#images = pickle.load(open("/home/nico/DepthPrediction/img.txt"))
#label = pickle.load(open("/home/nico/DepthPrediction/label.txt"))

caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver("/home/nico/DepthPrediction/xy_extension/models/solver_cord_extension.prototxt")
solver.net.copy_from("/home/nico/DepthPrediction/models/unary_depth_regressor/places205CNN_iter_300000_upgraded.caffemodel")

transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('/home/nico/DepthPrediction/mean.npy')) # mean pixel
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

num_dbims = len(images)
it = 100000
train_loss = np.zeros(it)
for i in range(it):
    ims = np.zeros((128,3,227,227),dtype = np.float32)
    bat_label = np.zeros((128,1,1,1),dtype = np.float32)
    xy = np.zeros((128,1,6,6),dtype = np.float32)
    
    for j in range(128):
        randseed = (num_dbims-1)
        marker = randint(0,randseed)
        img_tmp = imread(images[marker])
        ims[j] = transformer.preprocess('data', img_tmp)
        bat_label[j] = float(label[marker])
        xy[j][0][0][0] = normalise_x(x[marker])
        xy[j][0][0][1] = normalise_y(y[marker])    
    
    #solver.net.blobs['data'].data[...] = ims.transpose(0,3,1,2)
    solver.net.blobs['data'].data[...] = ims
    solver.net.blobs['label'].data[...] = bat_label
    solver.net.blobs['xy'].data[...] = xy
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data
    
    if(i%500 == 0):
        f = open('/home/nico/DepthPrediction/xy_extension/cord_loss_record/loss_'+str(i)+'.npy','a')
        np.save(f,train_loss)
        f.close()