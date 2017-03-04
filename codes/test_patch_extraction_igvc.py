import numpy as np
import lutorpy as lua
# import os
from sklearn.feature_extraction import image
# import scipy.ndimage.imread as imread
from skimage.io import imread
from random import shuffle
# require('xlua')

path = "../data/test_selected/"
out_root = "../data/patches/"
Image = []
Truth = []
# Folder = []

print ''
print 'Loading data..'
for i in range(1,4):
    print i
    Image.append(path+'image-00'+str(i)+'.jpeg')
    Truth.append(path+'image-00'+str(i)+'_gt.jpg')

print ''
print 'Extracting patches..'
image_iter = len(Image)
# print(image_iter)

for i in range(3):
    # xlua.progress(i+1,image_iter)
    # print(Folder[i])
    print str(i+1)+'/'+str(image_iter)
    
    try:
        truth_data = imread(Truth[i])
    except:
        print ''
        print 'No truth file found'
        print ''
        continue    
    print(truth_data.shape)

    image_data = imread(Image[i])
    print(image_data.shape)
    image_data = image_data[:,:,2]
    print(image_data.shape)
    
    patch_size_x = 31
    patch_size_y = 31

    pixel_offset = 1

    Image_patch = image.extract_patches(image_data,[patch_size_x,patch_size_y],extraction_step=pixel_offset)
    Image_patch = Image_patch.reshape(Image_patch.shape[0]*Image_patch.shape[1], patch_size_x, patch_size_y)

    T_patch = image.extract_patches(truth_data,[patch_size_x,patch_size_y],extraction_step=pixel_offset)
    T_patch = T_patch.reshape(T_patch.shape[0]*T_patch.shape[1], patch_size_x, patch_size_y)
    T_patch = T_patch[:,(patch_size_x-1)/2,(patch_size_y-1)/2]
    print(np.unique(T_patch))
    truth_patches = np.zeros(T_patch.shape)
    truth_patches[T_patch>200] = 1
    truth_patches = truth_patches.astype(int)
    print(np.unique(truth_patches))
    print(truth_patches.shape)
    # print T_patch.shape

    # grass_index = np.where(truth_patches == 0)
    # lane_index = np.where(truth_patches == 1)
    # print 'length : ', len(indexx[0])

    # indexx = indexx[0]
    # print(len(indexx[0]))
    # print(indexx)

    # print(Image_patch.shape)
    # grass = Image_patch[grass_index,:]
    # lane = Image_patch[lane_index,:]

    if i == 0:
        data = Image_patch
        label = truth_patches
    else:
        data = np.append(data,Image_patch,axis=0)
        label = np.append(label,truth_patches,axis=0)

data = torch.fromNumpyArray(data)
label = torch.fromNumpyArray(label)

torch.save('/home/ipcv16/igvc/lane_detection/data/patches/test_data.t7', data)
torch.save('/home/ipcv16/igvc/lane_detection/data/patches/test_label.t7', label)