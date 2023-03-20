from os import listdir
from os.path import join
from torch.utils.data import Dataset
from newTransform import compose, normalize, toTensor, randomCrop, randomHorizontalFlip, randomVerticalFlip, randomRotation, Resize_train, Resize_test, randomRoll 

import scipy.io as scio
import numpy as np
import random  
import pdb
import torch 
from torch.autograd import Variable 
import os
from imresize import downscale, anisotropic_gaussian_kernel, isotropic_gaussian_kernel


    
def calculate_valid_crop_size(upscale_factor, patchSize):
    return  upscale_factor * patchSize
      
def train_hr_transform(crop_size):

    return compose([
      randomCrop(crop_size),
      randomHorizontalFlip(),
      randomVerticalFlip(),
      randomRoll(),
      randomRotation('90'),
      randomRotation('-90'),
      toTensor()     
    ])
 
def train_lr_transform(upscale_factor, interpolation):
    return compose([ 
        Resize_train(upscale_factor, interpolation)
    ])
        
class TrainsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir,  upscale_factor, interpolation = 'Bicubic', patchSize = 28, crop_num = 48):
        super(TrainsetFromFolder, self).__init__()
        
        image_names = listdir(dataset_dir) 
        
        self.total_names = []
        for i in range(crop_num):
            for j in range(len(image_names)):              
                self.total_names.append(dataset_dir + image_names[j])
          
        random.shuffle(self.total_names)
                     
        crop_size = calculate_valid_crop_size(upscale_factor, patchSize)

        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(upscale_factor, interpolation)
        
        csr = scio.loadmat('P_N_V2.mat')['P'].astype(np.float32)
        self.csr = Variable(torch.from_numpy(csr))
        
    def __getitem__(self, index):
   
        mat = scio.loadmat(self.total_names[index], verify_compressed_data_integrity=False) 
        
        hsi = mat['hsi'].astype(np.float32)
        rgb = mat['rgb'].astype(np.float32)
        cat = np.concatenate((hsi,rgb),axis=2)
        #print(cat.shape)

        crop = self.hr_transform(cat) 
       
        crop_hsi = crop[:128,:,:]
        crop_hr_rgb = crop[128:131,:,:]
               
        crop_lr_hsi = self.lr_transform(crop_hsi)      
       
        return crop_hsi, crop_lr_hsi, crop_hr_rgb

    def __len__(self):
        return len(self.total_names) 
        
        
def test_lr_hsi_transform(upscale_factor, interpolation):
    return compose([ 
        Resize_test(upscale_factor, interpolation)
    ])
    
            
def test_hr_rgb_transform():
    return compose([ 
        toTensor()
    ])
    
def mrop(scale, width, height):
    W = width//scale
    H = height//scale
    
    return int(W*scale), int(H*scale)
    
class ValsetFromFolder(Dataset):
    def __init__(self, dataset, dataset_dir, upscale_factor, interpolation):
        super(ValsetFromFolder, self).__init__()
        
        image_names = listdir(dataset_dir) 

        self.total_names = []
        for j in range(len(image_names)):              
            self.total_names.append(dataset_dir + image_names[j])         
                     
        self.lr_hsi_transform = test_lr_hsi_transform(upscale_factor, interpolation)
        self.hr_rgb_transform = test_hr_rgb_transform()
        
        csr = scio.loadmat('P_N_V2.mat')['P'].astype(np.float32)
        self.csr = Variable(torch.from_numpy(csr))
        self.scale = upscale_factor
        self.names = image_names

    def __getitem__(self, index):
        mat = scio.loadmat(self.total_names[index], verify_compressed_data_integrity=False) 
        hsi = mat['hsi'].astype(np.float32).transpose(2,0,1)
        rgb = mat['rgb'].astype(np.float32).transpose(2,0,1)
        W, H = mrop(self.scale, hsi.shape[1], hsi.shape[2]) 
        
        hsi = torch.from_numpy(hsi)[:,:W,:H]
        rgb = torch.from_numpy(rgb)[:,:W,:H]
        
                
        #out_path = '/home/opt/liqiang/data/CK/test_mat_CK/' + str(self.scale) + '/downsample/'  

        #if not os.path.exists(out_path):
        #   os.makedirs(out_path)

        lr_hsi = self.lr_hsi_transform(hsi)  
        hr_rgb = self.hr_rgb_transform(rgb)     
        
        #scio.savemat(out_path + self.names[index], {'LR': lr_hsi.numpy().transpose(1,2,0),'RGB': hr_rgb.numpy().transpose(1,2,0), 'HR': hsi.numpy().transpose(1,2,0)}) 

        return hsi, lr_hsi, hr_rgb, self.names[index]

    def __len__(self):
        return len(self.total_names)