import scipy.io
import numpy as np
from image_handling_with_xy import load_dataset
splits_path = '/home/nico/data/splits.mat'
splits = scipy.io.loadmat(splits_path)
train_inds = splits['trainNdxs']
test_inds = splits['testNdxs']

[image_set, depths] = load_dataset('/home/nico/data/nyu_depth_v2_labeled.mat')
img = np.zeros((3,640,480))
for i in range(len(train_inds)):
    img = image_set[train_inds[i][0]] + img
img = img/float(len(train_inds))
np.save('/home/nico/DepthPrediction/xy_extension/mean.npy',np.mean(np.mean(img,axis=1),axis=1))