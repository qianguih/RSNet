import numpy as np
import h5py
import glob
import math
import cv2
import random
import os

#-- slice processing utils
def gen_slice_idx(data, resolution, axis=2):
    indices = np.zeros((  data.shape[0], data.shape[2] ))
    for n in range(data.shape[0]):
        indices[n] = gen_slice_idx_routine( data[n], resolution, axis )
    #
    return indices

def gen_slice_idx_routine(data, resolution, axis):
    if axis == 2:
        z_min, z_max = Z_MIN, Z_MAX
    else:
        z_min, z_max = data[:,:,axis].min(), data[:,:,axis].max()

    #gap = (z_max - z_min + 0.001) / numSlices
    gap = resolution
    indices = np.ones( ( data.shape[1], 1 ) ) * float('inf')
    for i in range( data.shape[1]  ):
        z = data[0,i,axis]
        idx = int( (z - z_min) / gap )
        indices[i, 0] = idx
    return indices[:, 0]


#-- utils for loading data, from Pointnet
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

#-- load data here
#- dataset setting, update when neccessay
block_size = 1.0
stride = 0.5
area = 'Area_5'
DATA_DIR = './data/'

TRAIN_DIR = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_{}_{}m_{}s_train/'.format(area, block_size, stride))
TEST_DIR = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_{}_{}m_{}s_test/'.format( area, block_size, block_size ))

print("loading raw data...")
train_files = glob.glob(TRAIN_DIR + '*.h5')
test_files = glob.glob(TEST_DIR + '*.h5')

assert len(train_files) != 0, "dataset not processed correctly"
assert len(test_files) != 0, "dataset not processed correctly"

train_data, train_label = [], []
for filename in train_files:
    data_batch, label_batch = loadDataFile(filename)
    train_data.append(data_batch)
    train_label.append(label_batch)

train_data = np.concatenate(train_data, 0)
train_label = np.concatenate(train_label, 0)

test_data, test_label = [], []
for filename in test_files:
    data_batch, label_batch = loadDataFile(filename)
    test_data.append(data_batch)
    test_label.append(label_batch)

test_data = np.concatenate(test_data, 0)
test_label = np.concatenate(test_label, 0)

print "training set: ", (train_data.shape, train_label.shape)
print "testing set: ", (test_data.shape, test_label.shape)

Z_MIN, Z_MAX = min( train_data[:,:,2].min(), test_data[:,:,2].min() ), max( train_data[:,:,2].max(), test_data[:,:,2].max() )

def iterate_data(batchsize, resolution, train_flag = True, require_ori_data=False, block_size=1.0):
    if train_flag:
        data_all = train_data
        label_all = train_label
        indices = range(data_all.shape[0])
        np.random.shuffle(indices)
    else:
        data_all = test_data
        label_all = test_label
        indices = range(data_all.shape[0])

    file_size = data_all.shape[0]
    num_batches = int(math.floor( file_size / float(batchsize) ))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batchsize
        excerpt = indices[start_idx:start_idx + batchsize]
        
        inputs = data_all[excerpt].astype('float32')
        
        if require_ori_data:
            ori_inputs = inputs.copy()
        
        for b in range(inputs.shape[0]):
            minx = min(inputs[b, :, 0])
            miny = min(inputs[b, :, 1])
            inputs[b, :, 0] -= (minx+block_size/2)
            inputs[b, :, 1] -= (miny+block_size/2)
        
        inputs = np.expand_dims(inputs,3).astype('float32')
        inputs = inputs.transpose(0,3,1,2)

        seg_target = label_all[excerpt].astype('int64') # num_batch, num_points

        if len(resolution) == 1:
            resolution_x = resolution_y = resolution_z = resolution
        else:
            resolution_x, resolution_y, resolution_z = resolution

        x_slices_indices = gen_slice_idx(inputs, resolution_x, 0).astype('int32')
        y_slices_indices = gen_slice_idx(inputs, resolution_y, 1).astype('int32')
        z_slices_indices = gen_slice_idx(inputs, resolution_z, 2).astype('int32')

        if not require_ori_data:
            yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target
        else:
            yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target, ori_inputs















