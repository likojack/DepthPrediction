# Creates dataset directories (jpegs with an index file)

import numpy as np
import sys
import matplotlib.pyplot as plt
from image_handling_with_xy import load_dataset
from image_handling_with_xy import segment_image

# Generate full datasets, using splits file: 
import scipy.io

# Load splits
splits_path = '/home/nico/data/splits.mat'
splits = scipy.io.loadmat(splits_path)
train_inds = splits['trainNdxs']
test_inds = splits['testNdxs']

from image_handling_with_xy import create_segments_dataset
from image_handling_with_xy import create_segments_directory

min_depth = 0.7;
max_depth = 10;
depth_bins = 32;

window_size = 113; # 167x167~NYUv2 paper # 227x227, imagenet standard
n_superpixels = 400;

output_images=False
average_type=0     # 0: center val, 1:superpixel avg, 2:patch avg
# Generate training set

# matlab file
input_file = '/home/nico/data/nyu_depth_v2_labeled.mat'
# Directory of images
img_filepath_train = '/home/nico/DepthPrediction/train_xy'
img_filepath_test = '/home/nico/DepthPrediction/test_xy'

train_slices = np.array_split(train_inds - 1,10)
test_slices = np.array_split(test_inds - 1, 10)

## THESE LINES FOR BUILDING A FRESH SET
print 'Building training set...'
for s in train_slices:
    images = tuple(s)
    data_file = create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_train,
                    no_superpixels=n_superpixels, x_window_size=window_size, y_window_size=window_size, images=images,
                    depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth,test=False)

print 'Building validation set...'
for s in test_slices:
    images = tuple(s)
    data_file = create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_test, 
                     no_superpixels=n_superpixels, x_window_size=113, y_window_size=113, images=images,
                     depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth,test=True)

print "nBins = ",depth_bins
print "Avg Type = ",average_type

#print 'Building training set...'
#for s in train_slices:
#	images = tuple(s)
#   	create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_train,
#                   no_superpixels=n_superpixels, x_window_size=window_size, y_window_size=window_size, images=images,
#                   depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth, depth_type=average_type,
#				   output_images=output_images, index_name='index_' + str(depth_bins) + '_' + str(average_type) + '.txt')

#print 'Building validation set...'
#for s in test_slices:
#	images = tuple(s)
#	create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_test,
#                   no_superpixels=n_superpixels, x_window_size=window_size, y_window_size=window_size, images=images,
#                   depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth, depth_type=average_type,
#				   output_images=output_images, index_name='index_' + str(depth_bins) + '_' + str(average_type) + '.txt')

    
