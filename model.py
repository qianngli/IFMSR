import torch
import torch.nn as nn
import pdb
from patch import calc_padding
import torch.nn.functional as F                                        

        
class Unit(nn.Module):
    def __init__(self, wn, n_feats, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU(inplace=True)):
        super(Unit, self).__init__()        
        
        m = []
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))) 
        m.append(act)
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))) 
   
        self.m = nn.Sequential(*m)
    
    def forward(self, x):
        
        out = self.m(x) + x 
        #out = self.m(out) + x   

        return out 

class Unit1(nn.Module):
    def __init__(self, wn, n_feats, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU(inplace=True)):
        super(Unit1, self).__init__()        
        
        m = []
        m.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))) 
   
        self.m = nn.Sequential(*m)
    
    def forward(self, x):
        
        x = self.m(x)
        return x 

class rgb_Aggregate(nn.Module):
    def __init__(self, wn, n_feats, patchsize = 5, stride = 5):
        super(rgb_Aggregate, self).__init__()
        
        self.stride = stride
        self.patchsize = patchsize
        
        self.rgb_Fusion = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=3, padding=1, bias=True))
        self.hsi_Fusion = wn(nn.Conv2d(n_feats*4, n_feats, kernel_size=3, padding=1, bias=True)) 

        self.rgb_gamma = nn.Parameter(torch.ones(4))   
        self.hsi_gamma = nn.Parameter(torch.ones(4))   
        self.n_feats = n_feats


    def forward(self, x_rgb, y_hsi, corr, flag=False):
        
        n_feats = int(self.n_feats//2)
        hsi = torch.cat([x_rgb[:,0:n_feats,:,:], y_hsi[:,0:n_feats,:,:]], 1)
        rgb = torch.cat([x_rgb[:,n_feats:self.n_feats,:,:], y_hsi[:,n_feats:self.n_feats,:,:]], 1)
       
        rgb_index = corr[0]
        

        padtop, padbottom, padleft, padright = calc_padding(rgb.shape[2:], self.patchsize, self.stride, padding=None) 

        pad_rgb = F.pad(rgb, pad=(padleft, padright, padtop, padbottom)) 
        pad_hsi = F.pad(hsi, pad=(padleft, padright, padtop, padbottom)) 

        N, C, W, H = pad_rgb.size()

        fold_rgb = F.unfold(pad_rgb, kernel_size=self.patchsize, stride=self.stride)
        fold_hsi = F.unfold(pad_hsi, kernel_size=self.patchsize, stride=self.stride)          

        N, _, L = fold_rgb.size()
        fold_rgb = fold_rgb.permute(0, 2, 1)
        fold_rgb = fold_rgb.view(N, L, -1, self.patchsize, self.patchsize)  

        fold_hsi = fold_hsi.permute(0, 2, 1)
        fold_hsi = fold_hsi.view(N, L, -1, self.patchsize, self.patchsize)          


        N, B, I = rgb_index.size()

        patch_rgb = torch.zeros(N, fold_rgb.shape[1], C*4, self.patchsize, self.patchsize).cuda()
        patch_hsi = torch.zeros(N, fold_hsi.shape[1], C*4, self.patchsize, self.patchsize).cuda()

        for i in range(N):
            for j in range(B):
                patch_rgb[i,j,:,:,:] = torch.cat([self.rgb_gamma[0]*fold_rgb[i,j,:], self.rgb_gamma[1]*fold_rgb[i,rgb_index[i,j,1],:,:], self.rgb_gamma[2]*fold_rgb[i,rgb_index[i,j,2],:,:], 
                         self.rgb_gamma[3]*fold_rgb[i,rgb_index[i,j,3],:,:]], 0)  

                patch_hsi[i,j,:,:,:] = torch.cat([self.hsi_gamma[0]*fold_hsi[i,j,:], self.hsi_gamma[1]*fold_hsi[i,rgb_index[i,j,1],:,:], self.hsi_gamma[2]*fold_hsi[i,rgb_index[i,j,2],:,:], 
                         self.hsi_gamma[3]*fold_hsi[i,rgb_index[i,j,3],:,:]], 0)  
                        
        
        patch_rgb = patch_rgb.view(N, L, -1)
        patch_rgb = patch_rgb.permute(0, 2, 1)

        patch_rgb = F.fold(patch_rgb, (W, H), kernel_size=self.patchsize, stride=self.stride)
        patch_rgb = patch_rgb[:,:,padtop:W-padbottom, padleft:H-padright]


        patch_hsi = patch_hsi.view(N, L, -1)
        patch_hsi = patch_hsi.permute(0, 2, 1)

        patch_hsi = F.fold(patch_hsi, (W, H), kernel_size=self.patchsize, stride=self.stride)
        patch_hsi = patch_hsi[:,:,padtop:W-padbottom, padleft:H-padright]

        patch_rgb = self.rgb_Fusion(patch_rgb)  
        patch_hsi = self.hsi_Fusion(patch_hsi)  

        return patch_rgb + rgb,  patch_hsi + hsi       

class FASplit(nn.Module):
    def __init__(self, wn, n_feats, scale, stride, patchsize, kernel_size=3, padding=1, bias=True, act=nn.ReLU(inplace=True)): 
        super(FASplit, self).__init__()
               
        self.aggregate = rgb_Aggregate(wn, n_feats, stride, patchsize)
                
    def forward(self, rgb, hsi, corr):

        out_rgb, out_hsi = self.aggregate(rgb, hsi, corr)

        return out_rgb , out_hsi
    	
class DepthDC3x3_1(nn.Module):
    def __init__(self, wn, in_xC, in_yC, out_C):
        super(DepthDC3x3_1, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            #Unit1(wn, in_yC),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_3(nn.Module):
    def __init__(self, wn, in_xC, in_yC, out_C):
        super(DepthDC3x3_3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            #Unit1(wn, in_yC),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DDPM(nn.Module):
    def __init__(self, wn, in_xC=64, in_yC=64, out_C=64, kernel_size=3):

        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C // 2
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC3x3_1(wn, self.mid_c, in_yC, self.mid_c)
        self.branch_3 = DepthDC3x3_3(wn, self.mid_c, in_yC, self.mid_c)
        self.fuse = nn.Conv2d(96, 64, 3, 1, 1) 

    def forward(self, x, y):

        x = self.down_input(x)
        result_1 = self.branch_1(x, y)
        result_3 = self.branch_3(x, y)
        return self.fuse(torch.cat((x, result_1, result_3), dim=1))

        
class Head(nn.Sequential):
    def __init__(self, wn, input_feats, output_feats, kernel_size, padding=1, bias=True):

        m = []
        m.append(wn(nn.Conv2d(input_feats, output_feats, kernel_size=kernel_size, padding=1, bias=True)))
                    
        super(Head, self).__init__(*m)  


class FilterLayer(nn.Module):
   def __init__(self, in_planes, out_planes, reduction=16):
       super(FilterLayer, self).__init__()
       self.out_planes = out_planes
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.fc = nn.Sequential(
           nn.Linear(in_planes, out_planes // reduction),
           nn.ReLU(inplace=True),
           nn.Linear(out_planes // reduction, out_planes),
           nn.Sigmoid()
       )
       
   def forward(self, x):
       b, c, _, _ = x.size()
       y = self.avg_pool(x).view(b, c)
       y = self.fc(y).view(b, self.out_planes, 1, 1)
       return y
'''
Feature Separation Part
'''
class FSP(nn.Module):
   def __init__(self, in_planes, out_planes, reduction=16):
       super(FSP, self).__init__()
       self.filter = FilterLayer(2*in_planes, out_planes, reduction)
   def forward(self, guidePath, mainPath):
   
       combined = torch.cat([guidePath, mainPath], dim=1)
       channel_weight = self.filter(combined)
       out = mainPath + channel_weight * guidePath

       return out

class inter_module(nn.Module):
    def __init__(self, wn, n_feats, scale,  stride, patchsize, kernel_size = 3, padding = 1, bias = True, act=nn.ReLU()):
        super(inter_module, self).__init__()
        self.act = act

        self.unit_hsi_1 = Unit(wn, n_feats)
        self.unit_rgb_1 = Unit(wn, n_feats)

        self.unit_hsi_2 = Unit(wn, n_feats)
        self.unit_rgb_2 = Unit(wn, n_feats)

        self.FASplit = FASplit(wn, n_feats, scale, stride, patchsize)  	

        self.ddpm = DDPM(wn) 



    def forward(self, data):

        out_rgb = self.act(self.unit_hsi_1(data[0]))                       
        out_hsi = self.act(self.unit_rgb_1(data[1]))
  	
        #out_rgb, out_hsi = self.FASplit(out_rgb, out_hsi, data[2])  #MPCA
        
        out_rgb = torch.cat([out_rgb[:,0:32,:,:], out_hsi[:,0:32,:,:]], 1)
        out_hsi = torch.cat([out_rgb[:,32:64,:,:], out_hsi[:,32:64,:,:]], 1)

        out_rgb = self.unit_hsi_2(out_rgb)
        out_hsi = self.unit_rgb_2(out_hsi)

        out_rgb = out_rgb + data[0]
        out_hsi = out_hsi + data[1]        

        #out_mid = self.ddpm(out_hsi, out_rgb)   # RDE

 
        return out_rgb, out_hsi   #out_rgb+out_mid, out_hsi+out_mid  


class casde(nn.Module):
   def __init__(self, wn, n_module, n_feats=64, bias=True):
        super(casde, self).__init__()
        
        self.reduce1 = wn(nn.Conv2d(n_feats*n_module, n_feats, kernel_size=1, bias=bias)) 
        self.reduce2 = wn(nn.Conv2d(n_feats*n_module, n_feats, kernel_size=1, bias=bias)) 

        self.unit_hsi1 = Unit(wn, n_feats)
        self.unit_hsi2 = Unit(wn, n_feats)

        self.act = nn.ReLU()

        #self.reduce = wn(nn.Conv2d(n_feats*2, n_feats, kernel_size=1, bias=bias)) 

   def forward(self, rgb, hsi):
        
        ##################################   DCMF module  
        rgb = self.reduce1(torch.cat(rgb, 1))
        hsi = self.reduce2(torch.cat(hsi, 1))
        #hsi_cat =hsi  

        rgb_r = self.unit_hsi1(rgb) 
        hsi =  torch.mul(rgb_r, hsi) 
        hsi =  self.unit_hsi2(rgb) + hsi  
        #pdb.set_trace()
        #scio.savemat('lr_hsi.mat', {'hsi_cat' : hsi_cat.squeeze(0).cpu().numpy().transpose(1,2,0), 'hsi' : hsi.squeeze(0).cpu().numpy().transpose(1,2,0)})         
        ################################   single bottleneck
        #hsi = self.reduce(torch.cat([hsi, rgb], 1))
                                                        
        ##############################  bottleneck
        #hsi = self.reduce(torch.cat([hsi[1], rgb[1]], 1))  
        #pdb.set_trace()
        return hsi


class CoarseNet(nn.Module):
    def __init__(self, args):
        super(CoarseNet, self).__init__()
        
        scale = args.upscale_factor
        n_feats = args.n_feats          
        kernel_size = 3
        self.n_module = 2
        stride = args.stride
        patchsize = args.patchsize
        
        wn = lambda x: torch.nn.utils.weight_norm(x) 
        	                                    
        self.hsi_head = nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True) 
        self.rgb_head = nn.Conv2d(3, n_feats, kernel_size=kernel_size, padding=1, bias=True) 

        inter_body = [
                      inter_module(wn, n_feats, scale, scale*stride, scale*patchsize
                  ) for _ in range(self.n_module)
        ]
        self.inter_body =  nn.Sequential(*inter_body)                              
     
        #self.casde = casde(wn, self.n_module)
        self.nearest = nn.Upsample(scale_factor=scale, mode='nearest')
        self.hsi_end = Head(wn, n_feats, 1, kernel_size)   

    def forward(self,  hsi, neigbor, rgb, corr):
    
        hsi = torch.cat([neigbor[:,0,:,:].unsqueeze(1), hsi.unsqueeze(1), neigbor[:,1,:,:].unsqueeze(1)], 1)
        hsi = self.nearest(hsi)
     
        hsi =  self.hsi_head(hsi) 

        rgb = self.rgb_head(rgb) 
        
        skipHSI = hsi 
        
        mid_rgb = []
        mid_hsi = []
               
        for i in range(self.n_module):        
            rgb, hsi = self.inter_body[i]([rgb, hsi, corr]) 
            mid_rgb.append(rgb)
            mid_hsi.append(hsi)

        hsi = self.casde(mid_rgb, mid_hsi)  #DCMF
        hsi = hsi + skipHSI
        hsi = self.hsi_end(hsi)

        return hsi
