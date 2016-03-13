import h5py
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
import os
import scipy
import matplotlib.pyplot as plt
import pickle

def normalise_x(x):
    return np.abs(x-320.0)/320.0
def normalise_y(y):
    return y/480.0

def load_dataset(filename=None):
	""" 
	Load in the hdf5 dataset from the specified filename with the labels
	'images' and 'depths'.  Return these two hdf5 objects as a tuple.
	"""
	nyu_set = h5py.File(filename, 'r')
	images = nyu_set['images']
	depths = nyu_set['depths']
	return [images, depths]


def lin_pixelate_values(array, min_val, max_val, bins):
    cuts = np.linspace(min_val,max_val,num=(bins+1))
    array_vals = np.array(array[:])
    val = np.reshape(np.digitize(array_vals.flatten(), cuts) - 1, array_vals.shape)
    return val.astype(int)

def real_lin_values(array, min_val, max_val, bins):    
    return min_val + (abs(max_val-min_val)/bins)*array


def segment_image(image=None, no_segments=500):
	"""
	Break the images into no_segments parts using the SLIC algorithm.  The
	algorithm does not guarantee no_segments will be found.
	"""
	if len(image.shape) != 3:
		raise ValueError('Dimension of image is not 3')
	if image.shape[0] != 3:
		raise ValueError('First dimension of image is not 3')
	mask = slic(np.transpose(image, (1, 2, 0)), n_segments=no_segments, compactness=15, sigma=1)
	return mask

def calculate_sp_centroids(mask=None):
	"""
	Generates an average position of each superpixel in x and y.  Return a
	2 x n array with x in (0, :) and y in (1, :)
	"""
	pixel_ids = np.unique(mask)

	centroids = np.zeros((2, len(pixel_ids)))
	weight_x = np.array(range(0, mask.shape[0]))
	weight_y = np.array(range(0, mask.shape[1]))

	for pixel in pixel_ids:
		total = np.sum(mask == pixel, axis=None, dtype=float)
		centroids[0, pixel] = np.sum(weight_x * np.sum(mask == pixel, axis=1, dtype=float) / total)
		centroids[1, pixel] = np.sum(weight_y * np.sum(mask == pixel, axis=0, dtype=float) / total)

	return centroids

def gather_regions(image=None, centroids=None, x_window_size=10, y_window_size=10, zero_pad=True):
	"""
	Generate an np.ndarray containing a box around all of the superpixels
	within the image.  The box is 1 + 2 * window_size in the x and y
	directions.  If zero_pad is True (default), the images whos boxes overhang
	the edges of the image are zero padded.  If zero_pad is False, images whos
	boxes would hang over the edge of the image are moved up to the boundary
	edge, so that the centroid sits off-center closer to the edge of the image.
	"""
	x_width = 2 * x_window_size + 1
	y_width = 2 * y_window_size + 1
	regions = np.zeros((centroids.shape[1], 3, x_width,  y_width), dtype=image.dtype)
	center_pixels = np.array(centroids, dtype=int)
	for pixel in range(0, center_pixels.shape[1]):
		x = center_pixels[0, pixel]
		y = center_pixels[1, pixel]
		x_left = x - x_window_size - 1
		x_right = x + x_window_size
		y_left = y - y_window_size - 1
		y_right = y + y_window_size

		if zero_pad:
			# Pull out the region from the image, but make sure it doesn't
			# extend beyond the boundaries of the image.
			image_region = image[:, max(x_left,0):min(x_right, image.shape[1]-1),
								    max(y_left,0):min(y_right, image.shape[2]-1)]

			# Fill in with zeros on the appropriate side if necessary.
			# Deal with the x size first and then handle any adjustments
			# necessary in y
			if x_left < 0:
				image_region = np.concatenate((np.zeros((image_region.shape[0], x_width - image_region.shape[1], image_region.shape[2])), image_region), axis=1)
			elif x_right >= image.shape[1]:
				image_region = np.concatenate((image_region, np.zeros((image_region.shape[0], x_width - image_region.shape[1], image_region.shape[2]))), axis=1)

			if y_left < 0:
				image_region = np.concatenate((np.zeros((image_region.shape[0], image_region.shape[1], y_width - image_region.shape[2])), image_region), axis=2)
			elif y_right >= image.shape[2]:
				image_region = np.concatenate((image_region, np.zeros((image_region.shape[0], image_region.shape[1], y_width - image_region.shape[2]))), axis=2)
			# Save into our return array
			regions[pixel, ...] = image_region

		else:
			# Old Shifting implementation
			if x_left < 0:
				x_left = 0
				x_right = 2 * x_window_size + 1
				x_left_hit = True
			elif x_right >= image.shape[1]:
				x_left = image.shape[1] - 2 * x_window_size - 2
				x_right = image.shape[1] - 1
				x_right_hit = True

			if y_left < 0:
				y_left = 0
				y_right = 2 * y_window_size + 1
				y_left_hit = True
			elif y_right >= image.shape[2]:
				y_left = image.shape[2] - 2 * y_window_size - 2
				y_right = image.shape[2] - 1
				y_right_hit = True

			regions[pixel, ...] = image[:, x_left:x_right, y_left:y_right]
	
	return regions

def gather_depths(depths, centroids=None,
				  depth_bins=None, depth_min=None, depth_max=None,
				  mask=None,
				  x_window_size=None, y_window_size=None,
				  depth_type=0):
	"""
	Pulls out the needed depths from a given depth map.  If given a superpixel
	mask, it will average the depths over each superpixel.  Otherwise, if given
	a list of centroids, it will gather the depths at that those points or
	average across a window of a given size if x_window_size and y_window_size
	are both specified.  All arguments can be provided if depth_type is
	specified.  For depth_type = 0, the depth at the centroid is returned.  For
	depth_type = 1, the superpixel average is returned, and if depth_type = 2,
	the window average is returned.  If depth_bins, depth_min, and depth_max
	are provided, the log pixelated depth values are returned.
	"""

	if (depth_type is not 0) and (depth_type is not 1) and (depth_type is not 2):
		raise ValueError('Invalid depth_type value of %d' % depth_type)

	# if depth_type == 0:
	# 	mask = None
	# 	x_window_size = None
	# 	y_window_size = None
	# elif depth_type == 1:
	# 	centroids = None
	# 	x_window_size = None
	# 	y_window_size = None
	# elif depth_type == 2:
	# 	mask = None
	# else:
	# 	raise ValueError('Invalid depth_type value of %d' % depth_type)
	# print "depth type: ", depth_type
	#if mask is not None:
	if depth_type==1:
		mask_vals = np.unique(mask)
		no_segments = len(mask_vals)
		mask_flat = mask.ravel()
		depths_flat = depths.ravel()
		if not np.all(mask_vals == range(0, no_segments)):
			raise ValueError('Mask does not contain values between 0 and %d' % (no_segments-1))
	#elif centroids is not None:
	elif (depth_type==0) or (depth_type==2):
		no_segments = centroids.shape[1]
		center_pixels = np.array(centroids, dtype=int)

	#else:
	#	raise ValueError('Neither mask nor centroids provided')

	# window_average = False
	# if (x_window_size is not None) and (y_window_size is not None):
	# 	window_average = True

	# preallocate space for the depth values
	segment_depths = np.zeros((no_segments, 1),dtype=np.float32)

	for depth_idx in range(0, no_segments):
		#if mask is not None:
		if depth_type==1: 	# Superpixel average
			segment_depths[depth_idx] = np.average(depths_flat[mask_flat == depth_idx])
		elif depth_type==2: # Window average
			segment_depths[depth_idx] = np.average(depths[
				max(center_pixels[0, depth_idx] - x_window_size, 0):
				min(center_pixels[0, depth_idx] + x_window_size, depths.shape[0] - 1),
				max(center_pixels[1, depth_idx] - y_window_size, 0):
				min(center_pixels[1, depth_idx] + y_window_size, depths.shape[1] - 1)])
		else: # Center-value (no average)
			segment_depths[depth_idx] = depths[center_pixels[0, depth_idx],
		   					    			   center_pixels[1, depth_idx]]
		
	# Convert depths to quantized logspace:
	if (depth_bins is not None) and (depth_min is not None) and (depth_max is not None):
		segment_depths = \
		lin_pixelate_values(segment_depths, bins=depth_bins, min_val=depth_min, max_val=depth_max)
        
	return segment_depths


def load_dataset_segments(
	filename=None,
	no_superpixels=500,
	images=None,
	x_window_size=10,
	y_window_size=10,
	depth_bins=None, depth_min = None, depth_max=None,
	depth_type=0):
	"""
	Combines all of the above to load images segments and their associated
	depths from a dataset and and return them as a tuple of ndarrays.
	"""
	if images == None:
		images = range(0, images.shape[0])
	if ((type(images) is not tuple) and (type(images) is not list)):
		images = range(0, images)
	
	[image_set, depths] = load_dataset(filename)
	no_segments = no_superpixels * len(images)
	segment_depths = np.zeros((no_segments, 1))
	image_segments = np.ndarray((no_segments,
								 image_set.shape[1],
								 2 * x_window_size + 1,
								 2 * y_window_size + 1))
	masks = np.zeros((len(images), depths.shape[1], depths.shape[2]))

	current_image = 0
	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		masks[current_image, ...] = mask
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)

		end_index = current_segment + centroids.shape[1]
		if end_index >= image_segments.shape[0]:
			image_segments.resize((end_index + 1,) + image_segments.shape[1:])
		if end_index >= segment_depths.shape[0]:
			segment_depths.resize((end_index + 1,) + segment_depths.shape[1:])

		image_segments[current_segment:(current_segment+centroids.shape[1]), ...] = \
			gather_regions(image, centroids, x_window_size=x_window_size, y_window_size=y_window_size)

		segment_depths[current_segment:(current_segment+centroids.shape[1]), ...] = \
			gather_depths(depths[image_idx, ...],
						  centroids=centroids,
						  mask=mask,
						  x_window_size=x_window_size,
						  y_window_size=y_window_size,
						  depth_type=depth_type,
						  depth_bins=depth_bins,
						  depth_min=depth_min,
						  depth_max=depth_max)

		current_segment = current_segment + centroids.shape[1]
		current_image = current_image + 1

	return image_segments[0:current_segment, ...], segment_depths[0:current_segment, ...], masks



def create_segments_dataset(
	input_filename=None,
	output_filename=None,
	no_superpixels=500,
	x_window_size=10,
	y_window_size=10,
	images=None,
	image_output_filepath=None,
	depth_bins=None, depth_min = None, depth_max=None,
	depth_type=0):
	"""
	Combines all of the above to load images segments and their associated
	depths from a dataset and and return them as a tuple of ndarrays.

	-To output a directory of images w/ index file, provide image_output_filepath.
	-To quantize delivered depths into bins, provide depth_bins, depth_min, depth_max

	See gather_depths for depth_type behavior.
	"""

	if images == None:
		images = range(0, image_set.shape[0])
	if type(images) is not tuple:
		images = range(0, images)
	
	[image_set, depths] = load_dataset(input_filename)
	no_segments = no_superpixels * len(images)

	# Check whether to output individual images
	indiv_output = (image_output_filepath is not None)
	if indiv_output:
		if not os.path.exists(image_output_filepath):
			os.makedirs(image_output_filepath)
		out_log = open(image_output_filepath + '/index.txt','w+')

	# Check if exporting an hdf5 file
	hdf5_output = (output_filename is not None)
	if hdf5_output:
		output_file = h5py.File(output_filename, 'w')
		image_segments = output_file.create_dataset("data",
			(no_segments, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1),
			chunks=(1, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1))


		segment_depths = output_file.create_dataset("label", (no_segments, 1), chunks=True)
		segment_image_index = output_file.create_dataset("image", (no_segments, 1), chunks=True)
		segment_superpixel_index = output_file.create_dataset("pixel", (no_segments, 1), chunks=True)
	else:
		segment_image_index = []
		segment_superpixel_index = []
		image_segments = []

	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)

		# Resize the arrays if they ended up being too small.
		# Will probably only be called on the last image if at all.
		end_index = current_segment+centroids.shape[1]

		if (hdf5_output):
			if end_index >= image_segments.shape[0]:
				image_segments.resize((end_index + 1,) + image_segments.shape[1:])
				segment_depths.resize((end_index + 1,) + segment_depths.shape[1:])
				segment_image_index.resize((end_index + 1,) + segment_image_index.shape[1:])
				segment_superpixel_index.resize((end_index + 1,) + segment_superpixel_index.shape[1:])

		# Pull out sections around the centroid of the superpixel
		image_segments[current_segment:end_index, ...] = \
				gather_regions(image, centroids,
						x_window_size=x_window_size,
						y_window_size=y_window_size)

		segment_depths[current_segment:(current_segment+centroids.shape[1]), ...] = \
			gather_depths(depths[image_idx, ...],
						  centroids=centroids,
						  mask=mask,
						  x_window_size=x_window_size,
						  y_window_size=y_window_size,
						  depth_type=depth_type,
						  depth_bins=depth_bins,
						  depth_min=depth_min,
						  depth_max=depth_max)

 		if indiv_output:
			for i in range(current_segment,end_index):
				name = image_output_filepath + '/' + str(image_idx) + '_' + str(i) + '.jpg'
				# write image
				scipy.misc.imsave(name,np.transpose(image_segments[i, ...],(0,2,1)))
				# append to log
				out_log.write(name + ' ' + str(int(segment_depths[i][0])) + '\n')


		current_segment = current_segment + centroids.shape[1]
	# If the number of superpixels was smaller than we expected, resize the
	# arrays before returning them
	if current_segment != image_segments.shape[0]:
		image_segments.resize((current_segment,) + image_segments.shape[1:])
		segment_depths.resize((current_segment,)  + segment_depths.shape[1:])
		segment_image_index.resize((current_segment,) + segment_image_index.shape[1:])
		segment_superpixel_index.resize((current_segment,) + segment_superpixel_index.shape[1:])
	return output_file


def apply_depths(segment_depths, mask):
	depth_image = np.zeros(mask.shape, dtype=segment_depths.dtype)
	for depth_index in range(0, len(segment_depths)):
		depth_image += segment_depths[depth_index] * (mask == depth_index)
	return depth_image

def create_segments_directory(
	mean = None, 
	input_filename=None,
	image_output_filepath=None,
	no_superpixels=200,
	x_window_size=10,
	y_window_size=10,
	images=None,
	depth_bins=None, depth_min = None, depth_max=None, depth_type=0,
	output_images=False, index_name='index.txt', test=False):
	"""
	outputs a directory of image segments, with index file.
	"""
	# Select which images to work with
	if images == None:
		print "input is not image!!!"
		images = range(0, image_set.shape[0])
	if type(images) is not tuple:
		images = range(0, images)
	
	[image_set, depths] = load_dataset(input_filename)
	# no_segments = no_superpixels * len(images)

	print "image_set shape: ", image_set.shape

	if(test==False):
		out_img = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/images_train.txt",'a')
		out_label = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/label_train.txt",'a')
		out_x = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/x_train.txt",'a')
		out_y = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/y_train.txt",'a')
	if(test==True):
		out_img = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/images_test.txt",'a')
		out_label = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/label_test.txt",'a')
		out_x = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/x_test.txt",'a')
		out_y = open("/home/nico/DepthPrediction/xy_extension/residual_learning/quantized_data/y_test.txt",'a')        

	for image_idx in images:

		print 'processing image', image_idx
		
		# Preprocess image 
		residual_depth = depths[image_idx[0]] - mean
		[image_segments, mask,segment_depths,centroids] = preprocess_image(image_set[image_idx[0]], true_depth=residual_depth,
           no_superpixels=400, x_window_size=113, y_window_size=113, depth_bins=32,depth_min=-3.2,depth_max=7.6,depth_type=0)
		for i in range(image_segments.shape[0]):
			name = str(image_idx) + '_' + str(i) + '.jpg'
            
			# write image
			if output_images:
				if not os.path.exists(image_output_filepath):
					os.makedirs(image_output_filepath)
				out_log = open(image_output_filepath + '/' + index_name,'a')
				scipy.misc.imsave(image_output_filepath + '/' + name,np.transpose(image_segments[i, ...],(0,2,1)))
            
			out_img.write(image_output_filepath + '/' + name + '\n')            
			out_label.write(str(segment_depths[i][0]) + '\n')
			out_x.write(str(int(centroids[0,i])) + '\n')
			out_y.write(str(int(centroids[1,i])) + '\n')
        


def preprocess_image(
	image, true_depth=None,
	no_superpixels=200,
	x_window_size=10,
	y_window_size=10,
	depth_bins=None, depth_min = None, depth_max=None, depth_type=0):
	"""
	Returns image segments, etc.
	Depth_type:
		0:	Center value only
		1:	Average over superpixel
		2:	Average over patch
	"""
	masks = segment_image(image, no_segments=no_superpixels)
	centroids = calculate_sp_centroids(masks)
	center_pixels = np.array(centroids, dtype=int)
	no_segments = center_pixels.shape[1]

	image_segments = np.ndarray([no_segments,3,2*x_window_size+1, 2*y_window_size+1])
	segment_depths = np.ndarray(no_segments)
	# Pull out sections around the centroid of the superpixel
	image_segments = \
			gather_regions(image, centroids,
					x_window_size=x_window_size,
					y_window_size=y_window_size)

	# # If provided depth maps, quantize and return those too
	if true_depth is not None:
		segment_depths = \
			gather_depths(true_depth,
					centroids=centroids,depth_bins=depth_bins,depth_min=depth_min,depth_max=depth_max,
					mask=masks,
					x_window_size=x_window_size,
					y_window_size=y_window_size,
					depth_type=depth_type)
	if true_depth is not None:
 		return image_segments, masks, segment_depths, centroids
 	else:
 		return image_segments, masks, centroids