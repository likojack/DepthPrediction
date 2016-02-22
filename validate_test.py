caffe_root = '/home/nico/caffe-build_3_apr_2015/'
import sys
sys.path.insert(0,caffe_root+'python')
import caffe
import numpy as np
from pylab import *
import glob
import pickle
import os

def normalise_x(x):
    return np.abs(x-320.0)/320.0
def normalise_y(y):
    return y/480.0

with open("/home/nico/DepthPrediction/xy_extension/images_test.txt") as F5:
    images_validation = F5.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/label_test.txt") as F6:
    label_validation = F6.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/x_test.txt") as F7:
    x_validation = F7.read().splitlines()
with open("/home/nico/DepthPrediction/xy_extension/y_test.txt") as F8:
    y_validation = F8.read().splitlines()

validate_num_dbims = len(images_validation)
validate_loss = np.zeros(195)
for i in range(1,196):
    caffe.set_device(1)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver("/home/nico/DepthPrediction/xy_extension/models/solver_cord_extension.prototxt")
    solver.net.copy_from("/home/nico/DepthPrediction/xy_extension/models/snapshot_cord/_iter_"+str(i*500)+".caffemodel")
    transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load('/home/nico/DepthPrediction/mean.npy')) # mean pixel
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    bat_loss = np.zeros(200)
    for j in range(200):
        validate_img = np.zeros((128,3,227,227),dtype=np.float32)
        validate_bat_label = np.zeros((128,1,1,1),dtype=np.float32)
        validate_xy = np.zeros((128,1,6,6),dtype = np.float32)

        for k in range(128):
            validate_randseed = (validate_num_dbims - 1)
            validate_marker = randint(0,validate_randseed)
            img_tmp = imread(images_validation[validate_marker])
            validate_img[k] = transformer.preprocess('data',img_tmp)
            validate_bat_label[k] = float(label_validation[validate_marker])
            validate_xy[k][0][0][0] = normalise_x(float(x_validation[validate_marker]))
            validate_xy[k][0][0][1] = normalise_x(float(y_validation[validate_marker]))

        solver.net.blobs['data'].data[...] = validate_img
        solver.net.blobs['label'].data[...] = validate_bat_label
        solver.net.blobs['xy'].data[...] = validate_xy
        out = solver.net.forward()
        bat_loss[j] = out['loss']
    validate_loss[i-1] = mean(bat_loss)
    print "processing " + str(j)
    
    f = open('/home/nico/DepthPrediction/xy_extension/validate_loss_record/loss_'+str(i*500)+'.npy','a')
    np.save(f,validate_loss)
    f.close()