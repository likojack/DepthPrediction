caffe_root = '/home/nico/caffe-build_3_apr_2015/'
import sys
sys.path.insert(0,caffe_root+'python')
sys.path.insert(0,'/home/nico/DepthPrediction/xy_extension')
import caffe
import numpy as np
from pylab import *
from image_handling_with_xy import load_dataset
from image_handling_with_xy import preprocess_image
from image_handling_with_xy import apply_depths
from image_handling_with_xy import real_world_values
import glob
import pickle
import os
import scipy.io

def normalise_x(x):
    return np.abs(x-320.0)/320.0
def normalise_y(y):
    return y/480.0

[image_set, depths] = load_dataset('/home/nico/data/nyu_depth_v2_labeled.mat')
splits_path = '/home/nico/data/splits.mat'
splits = scipy.io.loadmat(splits_path)
train_inds = splits['trainNdxs']
train_slices = np.array_split(train_inds - 1,10)

pretrained = "/home/nico/DepthPrediction/xy_extension/models/snapshot_cord/_iter_18000.caffemodel"
net = "/home/nico/DepthPrediction/xy_extension/models/cord_extension_deploy.prototxt"
CNN = caffe.Net(net, pretrained, caffe.TEST)

no_superpixels = 400

for s in train_slices:
    images = tuple(s)
    for img_ind in images:
        print "processing image: "+str(img_ind[0])
        [image_segments, mask, segment_depths, centroids] = preprocess_image(image_set[img_ind[0]],true_depth=depths[img_ind[0]],
                                                                             no_superpixels=no_superpixels, x_window_size=113,y_window_size=113,
                                                                             depth_bins=32,depth_min=0.7,depth_max=10,depth_type=0);
        transformer = caffe.io.Transformer({'data': CNN.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        # transformer.set_raw_scale('data',225)
        transformer.set_mean('data', np.load('/home/nico/DepthPrediction/mean.npy')) # mean pixel
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        output_ = np.zeros((len(image_segments)),dtype = np.float32)
        for i in range(len(image_segments)):
            CNN.blobs['data'].data[...] = transformer.preprocess('data',image_segments[i].transpose(2,1,0))
            xy_channel = np.zeros((1,1,6,6),dtype = np.float32)
            xy_channel[0][0][0][0] = normalise_x(centroids[0,i])
            xy_channel[0][0][0][1] = normalise_y(centroids[1,i])
            CNN.blobs['xy'].data[...] = xy_channel
            out = CNN.forward()
            out = CNN.blobs['output_neuron'].data
            output_[i] = out[0][0]
            for i in range(len(output_)):
                if np.isnan(output_[i]) == True:
                    if i == 0:
                        output_[i] = output_[i+1]
                    output_[i] = output_[i-1]
        predicted = apply_depths(output_, mask)
        groundtruth = real_world_values(segment_depths,0.7,10,32)
        predicted_real = real_world_values(output_,0.7,10,32)
        loss = sqrt(mean((groundtruth - predicted_real)**2))
        f = open('/home/nico/DepthPrediction/xy_extension/extension_trainingloss_compare_18000.npy','a')
        f.write(str(img_ind[0]) + '\n')
        f.write(str(loss) + '\n')



