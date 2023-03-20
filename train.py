#coding:utf-8
import os
from option import  opt
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


from data_utils import TrainsetFromFolder, ValsetFromFolder
from model import CoarseNet
from eval import PSNR, SSIM, SAM
import numpy as np
import pdb

from torch.nn.functional import interpolate
from patch import *

def main():
    bestPSNR = 0.
       

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True
    
    # Loading datasets
    train_set = TrainsetFromFolder(opt.datasetName, '/data/liqiang/HSI-Fusion/'+ opt.datasetName + '/train/', 
                                  opt.upscale_factor, opt.interpolation, opt.patchSize, opt.crop_num)
                                  
    val_set = ValsetFromFolder(opt.datasetName,'/data/liqiang/HSI-Fusion/'+ opt.datasetName + '/test/', 
                               opt.upscale_factor, opt.interpolation)
 
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)    
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)  
          
      # Buliding model       
     
    model = CoarseNet(opt)
    criterion = nn.L1Loss() #coarse_Loss()         

    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu()   
    print('# parameters:', sum(param.numel() for param in model.parameters())) 
                   
    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(),  lr=opt.lr,  betas=(0.9, 0.999), eps=1e-08)    
        
    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)         
            opt.start_epoch = checkpoint['epoch'] + 1 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
           
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))       
        
    # Training 
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):  
        
        lr = adjust_learning_rate(epoch-1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"])) 
        train(train_loader, optimizer, model, criterion, epoch)
        if epoch > 120:
            bestPSNR = val(val_loader, model, epoch, optimizer, bestPSNR) 
        if epoch%2 == 0:                  
            save_checkpoint(epoch, model, optimizer, "checkpoint/")


def train(train_loader, optimizer, model, criterion, epoch):  

    for iteration, batch in enumerate(train_loader, 1):
        hr_hsi, lr_hsi, hr_rgb = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
       # pdb.set_trace()    
        index, values = patchIndex(hr_rgb.cuda(), opt.upscale_factor*opt.stride, opt.upscale_factor*opt.patchsize)


        if opt.cuda:
            lr_hsi = lr_hsi.cuda()
            hr_rgb = hr_rgb.cuda() 
            hr_hsi = hr_hsi.cuda()
            
        for i in range(lr_hsi.shape[1]):
            neigbor = []       	    
            if i==0:
                                
                neigbor.append(lr_hsi[:,1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,2,:,:].data.unsqueeze(1))
                	                 		                	                
            elif i==lr_hsi.shape[1]-1:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i-2,:,:].data.unsqueeze(1))
               	
            else:
                neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                neigbor.append(lr_hsi[:,i+1,:,:].data.unsqueeze(1))
                	
            single = lr_hsi[:,i,:,:]
            neigbor =  Variable(torch.cat(neigbor, 1))
                   
            SR = model(single, neigbor, hr_rgb, [index, values])  
            #pdb.set_trace()
            loss = criterion(SR.squeeze(1), hr_hsi[:,i,:,:]) 
                         	             
            optimizer.zero_grad()                
            loss.backward()                   
            optimizer.step()  
                                                       
        if iteration % 20 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))


def val(val_loader, model, epoch, optimizer, bestPSNR):
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    with torch.no_grad():   
        
        for iteration, batch in enumerate(val_loader, 1):
            hsi, lr_hsi, hr_rgb, names = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3]
            data = np.zeros((hsi.shape[1], hsi.shape[2], hsi.shape[3])).astype(np.float32) 

     	         
            index, values = patchIndex(hr_rgb.cuda(), opt.upscale_factor*opt.stride, opt.upscale_factor*opt.patchsize)


            if opt.cuda:
                lr_hsi = lr_hsi.cuda()
                hr_rgb = hr_rgb.cuda() 

            for i in range(lr_hsi.shape[1]):
                neigbor = []       	    
                if i==0:                                
                    neigbor.append(lr_hsi[:,1,:,:].data.unsqueeze(1))
                    neigbor.append(lr_hsi[:,2,:,:].data.unsqueeze(1))
                	                 		                	                
                elif i==lr_hsi.shape[1]-1:
                    neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                    neigbor.append(lr_hsi[:,i-2,:,:].data.unsqueeze(1))               	
                else:
                    neigbor.append(lr_hsi[:,i-1,:,:].data.unsqueeze(1))
                    neigbor.append(lr_hsi[:,i+1,:,:].data.unsqueeze(1))
                	
                single = lr_hsi[:,i,:,:]
                neigbor =  Variable(torch.cat(neigbor, 1))
             
                SR = model(single, neigbor, hr_rgb, [index, values])  

                SR[SR>1] = 1
                SR[SR<0] = 0                
                       
                data[i,:,:] = SR.detach().cpu().numpy()

            hsi = hsi.detach().cpu().numpy()[0]
       
            val_psnr += PSNR(data, hsi) 
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        average_psnr = val_psnr / len(val_loader)
            
        if average_psnr > bestPSNR:   
            bestPSNR = average_psnr  
            model_out_path =  "result/" + "{}_model_{}_{}.pth".format(opt.datasetName, opt.upscale_factor, opt.interpolation)
            state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
            if not os.path.exists('result'):
                os.makedirs('result')     	
            torch.save(state, model_out_path)        	   
           
    
        print("PSNR = {:.3f}   SSIM = {:.4F}    SAM = {:.3f}".format(val_psnr / len(val_loader), val_ssim / len(val_loader), val_sam / len(val_loader)))              
    
    return bestPSNR         	             

def adjust_learning_rate(epoch):

    lr = opt.lr /(2**(epoch // opt.step))
    return lr
        
def save_checkpoint(epoch, model, optimizer, path):
    model_out_path = path + "/{}_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch , "model": model.state_dict(), "optimizer":optimizer.state_dict()}
    if not os.path.exists(path):
        os.makedirs(path)     	
    torch.save(state, model_out_path)
 
if __name__ == "__main__":
    main()

