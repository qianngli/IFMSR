import torch
import torch.nn as nn
import numpy as np
from imresize import anisotropic_gaussian_kernel
import pdb

class spectralAgg(nn.Module):
    def __init__(self):
        super(spectralAgg, self).__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
                        
    def forward(self, x):
        y = x
        
        count = 8
        step = 0        
        batchsize, C, height, width = x.size()

        stride = int(height/count)
        
        for m in range(count):
            for n in range(count):

                patch = x[:,:,m*stride:(m+1)*stride,n*stride:(n+1)*stride] 
                	  
                reshapePatch = patch.contiguous().view(batchsize, C, -1)   
               	
                #only several channels are selected
                energy = torch.bmm(reshapePatch, reshapePatch.permute(0,2,1)) # compute correlation between channels
                energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  
                
                attention = self.softmax(energy_new)
                
                out = torch.bmm(attention, reshapePatch)
                
                out = out.view(batchsize, C, stride, stride)
                
                x[:,:,m*stride:(m+1)*stride,n*stride:(n+1)*stride] = out                 
                	
        out = self.gamma * x + y
        
        return out
#
#class RGBchannel(nn.Module):
#    def __init__(self):
#        super(RGBchannel, self).__init__()
#        self.conv2 = nn.Conv2d(31, 3, 3, 1, 1)        
#             
#    def forward(self, x):
#        x = self.conv2(x)       
#        return x 
#
#
                
class HSIchannel(nn.Module):
    def __init__(self):
        super(HSIchannel, self).__init__()
                       
        factor = 32
        KS = 21
        kernel = torch.rand(1, 1, KS, KS)
        kernel[0, 0, :, :] = torch.from_numpy(anisotropic_gaussian_kernel( np.array([KS, KS]), np.array([factor, factor]), 0.175*factor, 2.5*factor, np.random.rand() * np.pi))
        Conv = nn.Conv2d(1, 1, KS, factor)
        Conv.weight = nn.Parameter(kernel)
        
        if  KS % 2 == 1: 
            pad = int((KS - 1) / 2.)
        else:
            pad = int((KS - factor) / 2.)
                        
        self.dow = nn.Sequential(nn.ReplicationPad2d(pad), Conv)        
               
    def forward(self, x):

        inputs = []
        for i in range(x.shape[1]):
            inputs.append(self.dow(x[0,i,:,:].unsqueeze(0).unsqueeze(0)))
        return torch.cat(inputs, 1)
        
        return x
         
def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    eps = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=eps) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=eps)
                  
    denominator = denominator.clamp(min=eps).squeeze()
    
    denominator[denominator==0] = eps
    denominator[denominator<0] = eps

    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum        

                    
class FineNet(nn.Module):

    def __init__(self):
        super(FineNet, self).__init__()
        self.ReLU = nn.ReLU(inplace=True) 
                
        self.Conv1 = nn.Conv2d(34, 192, 3, 1, 1)      
        self.Conv2 = nn.Conv2d(192, 192, 3, 1, 1)        
        self.Conv3 = nn.Conv2d(192, 31, 3, 1, 1)   
#        self.Conv4 = nn.Conv2d(64, 31, 3, 1, 1)
        
        self.hsi = HSIchannel()

    def forward(self, x, y, z):

        out = torch.cat([x,y], 1)
        out = self.Conv1(out) 
        out = self.Conv2(self.ReLU(out))           
        out = self.Conv3(self.ReLU(out))  
#        out = self.Conv4(self.ReLU(out))                 
        out = out + x 
                  
        return  out, self.hsi(out) 