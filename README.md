<div align="justify">
  <div align="center">
    
  # [RGB-Guided Feature Modulation Network for Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10129137 "RGB-Guided Feature Modulation Network for Hyperspectral Image Super-Resolution")  
 
  </div>

## Update
**[2023-03-20]** IFMSR v0.1 is modified.  

## Abstract 
![Image text](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/architecture.png)  
Super-resolution (SR) is one of the powerful techniques to improve image quality for low-resolution (LR) hyperspectral image (HSI) with insufficient detail and noise. Traditional methods typically perform simple cascade or addition during the fusion of the auxiliary high-resolution (HR) RGB and LR HSI. As a result, the abundant HR RGB details are not utilized as a priori information to enhance the HSI feature representation, leaving room for further improvements. To address this issue, we propose an RGB-induced feature modulation network for HSI SR (IFMSR). Considering that similar patterns are common in images, a multi-corresponding patch aggregation is designed to globally assemble this contextual information, which is beneficial for feature learning. Besides, to adequately exploit plentiful HR RGB details, an RGB-induced detail enhancement (RDE) module and a deep cross-modality feature modulation (CFM) module are proposed to transfer the supplementary materials from RGB to HSI. These modules can provide a more direct and instructive representation, leading to further edge recovery. Experiments on several datasets demonstrate that our approach achieves comparable performance under more realistic degradation condition.  

## Motivation  
  - The image quality is degraded in the process of obtaining HSI due to the influence of sensor system and external conditions. As a result, the degraded image evidently loses high-frequency details of the real scene, which is not conducive to the effective discrimination of the objects.
  - Since the feature representation ability by handcrafted is limited, these methods are not able to recover the more details for realistic LR images, which makes the traditional methods less robust.
  - Although the deep learning-based methods sufficiently mine the internal information of their respective modality, they do not skillfully borrow the RGB image with full appearances to transfer the detailed contents to HSI modality for feature learning.
Therefore, how to leverage the intrinsic advantages of RGB image to assist the HSI branch to capture more discriminative features requires further research.

## Modules


### MCPA Module
When the spectral imager generates HSI, it commonly obtains the corresponding RGB image. How to address RGB image with abundant color and texture to guide HSI SR is extremely challenging dilemma. To fully utilize raw RGB image, we deal with this image to generate more meaningful materials. Generally, some patches in the whole image appear similar patterns. These patches can benefit detail restoration, which has been verified in several previous low-level tasks. Although Li et al. [10] consider contextual information in image, it only focuses on local content, and ignores other similar patterns in global perspective. To achieve this end, a MCPA module is developed, as shown in Fig. 2.

<div align="center">
      
  ![Image text](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/MCPA.png)  

</div>

*Fig. 3. Visual results in terms of spatial domain with existing SR methods on CAVE dataset. The results of balloons image are evaluated for scale factor × 4. The first line denotes SR results of 10-th band, and the second line denotes SR results of 20-th band.*  
  

### RDE Module
Among the low-level encoder features, RGB features contain more detailed information (i.e., texture and color), which can provide a more direct and instructive representation than HSI features. This facilitates the feature learning during encoding. Nevertheless, previous works only simply fuses the features of each modality, and do not exploit this remarkable behavior to induce the model that encourages the feature exploration of HSI modality. To tackle this issue, an RDE module is proposed to generate region-aware dynamic filter to guide encoding in HSI modality. The structure of RDE module is illustrated in Fig. 3.

<div align="center">
      
  ![Image text](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/RDE.png)   

</div>

*Fig. 3. Visual results in terms of spatial domain with existing SR methods on CAVE dataset. The results of balloons image are evaluated for scale factor × 4. The first line denotes SR results of 10-th band, and the second line denotes SR results of 20-th band.*  

### Deep Cross-Modality Feature Modulation Module
To produce super-resolved HSI, the two modality must be into a unified form during deep feature extraction. In this process, the RGB features are the auxiliary signals relative to the HSI features. For this reason, we propose a DCFM module, as illustrated in Fig. 1. This module modulates the RGB features and obtains the affine transformation coefficient to further enforce deep features in HSI modality.

<div align="center">
      
  ![Image text](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Deep_Cross-Modality_Feature_Modulation.png)   

</div>


## Dependencies  
**PyTorch, NVIDIA GeForce GTX 1080 GPU.**
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`


## Dataset Preparation 
Four public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](https://dataverse.harvard.edu/ "Harvard"), [Chikusei]() and []() are employed to verify the effectiveness of the proposed IFMSR.  

- In our work, we randomly select **80%** of the data as the training set and the rest for testing.  
- We augment the given training data by choosing **24** patches. With respect to each patch, its size is scaled **1**, **0.75**, and **0.5** times, respectively. We rotate these patches **90°** and flip them horizontally. Through various blur kernels, we then subsample these patches into LR hyperspectral images with the size of **L × 32 × 32**.  

## Implementation

### Pretrained model
Clone this repository:
 
        git clone https://github.com/qianngli/IFMSR.git
        cd IFMSR



### Main parameter settings


### Train & Test


## Result  

### Quantitative Evaluation

<div align="center">

  ![TABLE_VI](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/TABLE_VI.png)  
  ![TABLE_VII](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/TABLE_VII.png)  

</div>

### Qualitative Evaluation

<div align="center">
      
  ![Fig8](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig8.png)

</div>
  
  
<div align="center">
   
  ![Fig9](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig9.png)  
    
</div>

  
<div align="center">

  ![Fig10](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig10.png)  
    
</div>
 

## Citation 

[1] 

--------
If you has any questions, please send e-mail to liqmges@gmail.com.

</div>
