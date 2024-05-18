<div align="justify">
  <div align="center">
    
  # [RGB-Guided Feature Modulation Network for Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10129137 "RGB-Guided Feature Modulation Network for Hyperspectral Image Super-Resolution")  
 
  </div>

## Update
**[2023-03-20]** IFMSR v0.1 is modified.  

## Abstract 
![Architecture](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/architecture.png)  
Super-resolution (SR) is one of the powerful techniques to improve image quality for low-resolution (LR) hyperspectral image (HSI) with insufficient detail and noise. Traditional methods typically perform simple cascade or addition during the fusion of the auxiliary high-resolution (HR) RGB and LR HSI. As a result, the abundant HR RGB details are not utilized as a priori information to enhance the HSI feature representation, leaving room for further improvements. To address this issue, we propose an RGB-induced feature modulation network for HSI SR (IFMSR). Considering that similar patterns are common in images, a multi-corresponding patch aggregation is designed to globally assemble this contextual information, which is beneficial for feature learning. Besides, to adequately exploit plentiful HR RGB details, an RGB-induced detail enhancement (RDE) module and a deep cross-modality feature modulation (CFM) module are proposed to transfer the supplementary materials from RGB to HSI. These modules can provide a more direct and instructive representation, leading to further edge recovery. Experiments on several datasets demonstrate that our approach achieves comparable performance under more realistic degradation condition.  

## Motivation  
  - The image quality is degraded in the process of obtaining HSI due to the influence of sensor system and external conditions. As a result, the degraded image evidently loses high-frequency details of the real scene, which is not conducive to the effective discrimination of the objects.
  - Since the feature representation ability by handcrafted is limited, these methods are not able to recover the more details for realistic LR images, which makes the traditional methods less robust.
  - Although the deep learning-based methods sufficiently mine the internal information of their respective modality, they do not skillfully borrow the RGB image with full appearances to transfer the detailed contents to HSI modality for feature learning.
Therefore, how to leverage the intrinsic advantages of RGB image to assist the HSI branch to capture more discriminative features requires further research.

## Modules
### MCPA Module
When the spectral imager generates HSI, it commonly obtains the corresponding RGB image. How to address RGB image with abundant color and texture to guide HSI SR is extremely challenging dilemma. To fully utilize raw RGB image, we deal with this image to generate more meaningful materials. Generally, some patches in the whole image appear similar patterns. These patches can benefit detail restoration, which has been verified in several previous low-level tasks. Although Li et al. [3] consider contextual information in image, it only focuses on local content, and ignores other similar patterns in global perspective. To achieve this end, a MCPA module is developed, as shown in Fig. 1.

<div align="center">
      
  ![MCPA](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/MCPA.png)  

</div>

*Fig. 1. Illustration of MCPA module. For clear description, the processing of patch aggregation only in RGB modality is shown.*  
  

### RDE Module
Among the low-level encoder features, RGB features contain more detailed information (i.e., texture and color), which can provide a more direct and instructive representation than HSI features. This facilitates the feature learning during encoding. Nevertheless, previous works only simply fuses the features of each modality, and do not exploit this remarkable behavior to induce the model that encourages the feature exploration of HSI modality. To tackle this issue, an RDE module is proposed to generate region-aware dynamic filter to guide encoding in HSI modality. The structure of RDE module is illustrated in Fig. 2.

<div align="center">
      
  ![RDE](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/RDE.png)   

</div>

*Fig. 2. The architecture of RGB-induced detail enhancement (RDE) module.*  

### Deep Cross-Modality Feature Modulation Module
To produce super-resolved HSI, the two modality must be into a unified form during deep feature extraction. In this process, the RGB features are the auxiliary signals relative to the HSI features. For this reason, we propose a DCFM module, as illustrated in Fig. 3. This module modulates the RGB features and obtains the affine transformation coefficient to further enforce deep features in HSI modality.

<div align="center">
      
  ![Deep_Cross-Modality_Feature_Modulation](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Deep_Cross-Modality_Feature_Modulation.png)   

</div>

*Fig. 3. The architecture of Cross-Modality Feature Modulation module.*

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
Four public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](https://dataverse.harvard.edu/ "Harvard"), [Chikusei](https://naotoyokoya.com/Download.html "Chikusei") and [Sample of Roman Colosseum](https://earth.esa.int/eogateway/missions/worldview-2 "Sample of Roman Colosseum") are employed to verify the effectiveness of the proposed IFMSR.  

- **CAVE and Harvard:** 80% images are randomly selected as training set and the rest as test set.
- **Chikusei:** We crop the top left of the HSI (2000 × 2335 × 128) as the training set, and other content as the test set. To obtain extra samples, the training set is divided into non-overlapping images with size 200 × 194 × 128.
- **Sample of Roman Colosseum** To evaluate the performance in this dataset, the top left of the HSI (209 × 658 × 8) and HR RGB image (836 × 2632 × 3) are cropped to train model, and other part is selected to test.

- Since the spectral response function is known on the CAVE and Harvard datasets, the corresponding RGB images are generated using it. Other datasets with unknown spectral response function, Chikusei and Sample of Roman Collosseum, obtain corresponding RGB images via the position of pixels. Then, we random crop 64 patches with the size 12r × 12r from the each image in the training set, where r is upscale factor. All patches are augmented by random flip, rotation, and roll. Then, these patches are downsampled by the above strategy in this method to yield LR HSIs.

    > Existing works normally adopt Gaussian to degrade HR image during constructing label pairs. In real-world scenarios, image degradation is complicated by noise, blur, and other factors. The performance of traditional SR models using only a single type of kernel is significantly degraded for realistic degraded images. SR under unknown degradation is more challenging than traditional SR under simple degradation. Anisotropic and isotropic Gaussian are employed to randomly generate different kernels with five sizes. Here, the range of rotation angle is set to $[0, π]$ for anisotropic Gaussian kernel, and the range of kernel width is fixed at $[0.2, r]$. As for isotropic Gaussian kernel, the range of kernel width is set to $[0.2, r]$. Then, HR HSI $X′$ is downsampled by directly convolution, or is convoluted and interpolated. Gaussian noises with various levels are attached to the downsampled image. Finally, the label pair ${X, X′}$ is obtained.

- In the test stage, we adopt anisotropic Gaussian to generate kernel, so as to blur the HR HSI images. Each kernel is determined by a covariance matrix α , which is defined as

$$\begin{align*} \alpha = \left [{ \begin{array}{cc} \cos \left ({\theta }\right)&\quad {-}\sin \left ({\theta }\right)\\  
\sin \left ({\theta }\right)&\quad \cos \left ({\theta }\right) \end{array} }\right]\left [{ \begin{array}{cc} {\lambda _{1}}&\quad 0\\  
0&\quad {\lambda _{2}} \end{array} }\right]\left [{ \begin{array}{cc} \cos \left ({\theta }\right)&\quad \sin \left ({\theta }\right)\\  
{-} \sin \left ({\theta }\right)&\quad \cos \left ({\theta }\right) \end{array}}\right] \end{align*}$$  

## Implementation
### Pretrained model
Clone this repository:
 
        git clone https://github.com/qianngli/IFMSR.git
        cd IFMSR

1. Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org).  
1. You could download the [pre-trained model](https://github.com/qianngli/MulSR/blob/master/pre-train%20model.txt) from [Google Drive](https://drive.google.com/drive/folders/1LuXDv5__KDdC3EeJZU5DOMmbs0L4bE7I?usp=sharing).  
1. Remember to change the following path to yours：
   - `IFMSR/train.py` line 33, 36.
   - `IFMSR/fine.py` line 69, 70.

### Main parameter settings
- With respect to experimental setup, we select the size of convolution kernels to be **3 × 3**, except for the kernels mentioned above. Moreover, the number of these kernels is set to **64**.

        parser.add_argument('--kernel_size', type=int, default=3, help='number of module')
        parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')

- Following previous works, we fix the learning rate at **10^(−4)**, and its value is halved every **30** epoch.

        parser.add_argument("--lr", type=int, default=1e-4, help="lerning rate")
        parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs")

- To optimize our model, the **ADAM** optimizer with **β1 = 0.9** and **β2 = 0.99** is chosen.

### Train & Test
You can train or test directly from the command line as such:  

    python train.py --cuda --datasetName CAVE --upscale_factor 4  
    python fine.py --cuda --model_name checkpoint/model_4_epoch_xx.pth  

## Result  
- To evaluate the performance of the proposed method, this Section introduces six approaches to compare generalization ability on different datasets and scales, including **LTTR**, **CMS**, **PZRes-Net**, **MHF-net**, **MoG-DCN**, and **UAL**. Here, LTTR and CMS are unsupervised, while the other competitors are supervised. Note that LTTR and CMS introduce spectral response function to optimize object function. In particular, UAL incorporates the spectral response function into the loss function to learn the model in the adaptation module. To make a fair comparison, the corresponding loss term is removed on Chikusei with an unknown spectral response function.
- To evaluate the performance, peak signal-to-noise ratio (**PSNR**), structural similarity (**SSIM**), spectral angle mapper (**SAM**), relative dimensionless global error in synthesis (**ERGAS**), and root mean-squared error (**RMSE**) are exploited. Among these metrics, the higher the PSNR and SSIM values, the better the performance. The lower the SAM, ERGAS, and RMSE values, the better the reconstruction quality.

### Quantitative Evaluation

<div align="center">

  ![TABLE_IV](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/TABLE_IV.png)  

</div>

### Qualitative Evaluation

<div align="center">
      
  ![Fig4](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig4.png)

</div>
  
* Fig. 4. Visual comparison of spatial reconstruction. The first to three lines represent the visual results of the 15th band, 15th band, and 100th band, respectively.*

<div align="center">
   
  ![Fig5](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig5.png)  
    
</div>

* Fig.5. Visualcomparisonofspectraldistortionforcorrespondingimagesbyselectingtwopixels. (Left toright)Visual resultsofaboveimages.*

### Application on Real Hyperspectral Image

<div align="center">
  
  ![Fig7](https://raw.githubusercontent.com/qianngli/Images/master/IFMSR/Fig7.png)
  
</div>

* Fig. 6. Visual comparison on real HSI dataset. We choose the 2-3-5 bands after SR to synthesize the pseudo-color image.*  

## Citation 

[1] **Q. Li**, Q. Wang and X. Li, "Exploring the Relationship Between 2D/3D Convolution for Hyperspectral Image Super-Resolution," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, no. 10, pp. 8693-8703, 2021.  
[2] **Q. Li**, Y. Yuan, X. Jia and Q. Wang, "Dual-Stage Approach Toward Hyperspectral Image Super-Resolution," *IEEE Transactions on Image Processing*, vol. 31, pp. 7252-7263, 2022.  
[3] **Q. Li**, M. Gong, Y. Yuan and Q. Wang, "Symmetrical Feature Propagation Network for Hyperspectral Image Super-Resolution," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, pp. 1-12, 2022.  

--------
If you has any questions, please send e-mail to liqmges@gmail.com.

</div>
