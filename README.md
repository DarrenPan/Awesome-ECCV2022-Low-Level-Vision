# Awesome-ECCV2022-Low-Level-Vision
A Collection of Papers and Codes in ECCV2022 related to Low-Level Vision

## Related collections for low-level vision
- [CVPR2022-Low-Level-Vision](https://github.com/DarrenPan/CVPR2022-Low-Level-Vision)
- [Awesome-AAAI2022-Low-Level-Vision](https://github.com/DarrenPan/Awesome-AAAI2022-Low-Level-Vision)
- [Awesome-NeurIPS2021-Low-Level-Vision](https://github.com/DarrenPan/Awesome-NeurIPS2021-Low-Level-Vision)
- [Awesome-ICCV2021-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ICCV2021-Low-Level-Vision)
- [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision)
- [Awesome-ECCV2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision)


## Catalogue

- [Image Restoration](#ImageRetoration)
<!--
  - [Burst Restoration](#BurstRestoration)
  - [Video Restoration](#VideoRestoration)
-->
- [Super Resolution](#SuperResolution)
  - [Image Super Resolution](#ImageSuperResolution)
  - [Video Super Resolution](#VideoSuperResolution)
<!--
- [Image Rescaling](#Rescaling)

-->
- [Denoising](#Denoising)
  - [Image Denoising](#ImageDenoising)
  - [Video Denoising](#VideoDenoising)


- [Deblurring](#Deblurring)
  - [Image Deblurring](#ImageDeblurring)
  - [Video Deblurring](#VideoDeblurring)

- [Image Decomposition](#Decomposition)

<!--
- [Deraining](#Deraining)

- [Dehazing](#Dehazing)

- [Demosaicing](#Demosaicing)
-->
- [HDR Imaging / Multi-Exposure Image Fusion](#HDR)

- [Frame Interpolation](#FrameInterpolation)

- [Image Enhancement](#Enhancement)
  - [Low-Light Image Enhancement](#LowLight)

- [Image Harmonization](#Harmonization)

- [Image Completion/Inpainting](#Inpainting)

<!--
- [Image Matting](#Matting)
-->

- [Shadow Removal](#ShadowRemoval)

<!--
- [Relighting](#Relighting)

- [Image Stitching](#Stitching)

- [Image Compression](#ImageCompression)

-->

- [Image Quality Assessment](#ImageQualityAssessment)

<!--
- [Style Transfer](#StyleTransfer)

- [Image Editing](#ImageEditing)

-->

- [Image Generation/Synthesis/ Image-to-Image Translation](#ImageGeneration)
  - [Video Generation](#VideoGeneration)


- [Others](#Others)


<a name="ImageRetoration"></a>
# Image Restoration - 图像恢复

**Simple Baselines for Image Restoration**
- Paper: https://arxiv.org/abs/2204.04676
- Code: https://github.com/megvii-research/NAFNet

**D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration**
- Paper: https://arxiv.org/abs/2207.03294
- Code: https://github.com/zhaoyuzhi/D2HNet


<!--
<a name="BurstRestoration"></a>
## Burst Restoration

<a name="VideoRestoration"></a>
## Video Restoration

-->

<a name="SuperResolution"></a>
# Super Resolution - 超分辨率
<a name="ImageSuperResolution"></a>
## Image Super Resolution

**ARM: Any-Time Super-Resolution Method**
- Paper: https://arxiv.org/abs/2203.10812
- Code: https://github.com/chenbong/ARM-Net

**Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks**
- Paper: https://arxiv.org/abs/2203.03844
- Code: https://github.com/zysxmu/DDTB

**Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution**
- Paper: 
- Code: https://github.com/palmdong/MMSR
- Tags: Self-Supervised

**Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations**
- Paper: https://arxiv.org/abs/2203.01325
- Code: https://github.com/cszhilu1998/SelfDZSR
- Tags: Self-Supervised


<!--

<a name="VideoSuperResolution"></a>
## Video Super Resolution


<a name="Rescaling"></a>
# Image Rescaling - 图像缩放

-->
<a name="Denoising"></a>
# Denoising - 去噪

<a name="ImageDenoising"></a>
## Image Denoising

<a name="VideoDenoising"></a>
## Video Denoising

**Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones**
- Paper: 
- Code: https://github.com/nagejacob/FloRNN


<a name="Deblurring"></a>
# Deblurring - 去模糊
<a name="ImageDeblurring"></a>
## Image Deblurring

<a name="VideoDeblurring"></a>
## Video Deblurring

**Spatio-Temporal Deformable Attention Network for Video Deblurring**
- Paper:
- Code: https://github.com/huicongzhang/STDAN


**DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting**
- Paper: https://arxiv.org/abs/2111.09985
- Code: https://github.com/JihyongOh/DeMFI
- Tags: Joint Deblurring and Frame Interpolation

<a name="Decomposition"></a>
# Image Decomposition

**Blind Image Decomposition**
- Paper: https://arxiv.org/abs/2108.11364
- Code: https://github.com/JunlinHan/BID

<!--
<a name="Deraining"></a>
# Deraining - 去雨


<a name="Dehazing"></a>
# Dehazing - 去雾


<a name="Demosaicing"></a>
# Demosaicing - 去马赛克

-->

 <a name="HDR"></a>
# HDR Imaging / Multi-Exposure Image Fusion - HDR图像生成 / 多曝光图像融合

**Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging**
- Paper: 
- Code: https://github.com/viengiaan/EDWL


<a name="FrameInterpolation"></a>
# Frame Interpolation - 插帧

**Real-Time Intermediate Flow Estimation for Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2011.06294
- Code: https://github.com/hzwer/ECCV2022-RIFE 

**FILM: Frame Interpolation for Large Motion**
- Paper: https://arxiv.org/abs/2202.04901
- Code: https://github.com/google-research/frame-interpolation


<a name="Enhancement"></a>
# Image Enhancement - 图像增强

<a name="LowLight"></a>
## Low-Light Image Enhancement

**LEDNet: Joint Low-light Enhancement and Deblurring in the Dark**
- Paper: 
- Code: https://github.com/sczhou/LEDNet


<a name="Harmonization"></a>
# Image Harmonization - 图像协调

**Harmonizer: Learning to Perform White-Box Image and Video Harmonization**
- Paper: https://arxiv.org/abs/2207.01322
- Code: https://github.com/ZHKKKe/Harmonizer


<a name="Inpainting"></a>
# Image Completion/Inpainting - 图像修复

**Learning Prior Feature and Attention Enhanced Image Inpainting**
- Paper:
- Code: https://github.com/ewrfcas/MAE-FAR

<!--
<a name="Matting"></a>
# Image Matting - 图像抠图
-->


<a name="ShadowRemoval"></a>
# Shadow Removal - 阴影消除

**Style-Guided Shadow Removal**
- Paper:
- Code: https://github.com/jinwan1994/SG-ShadowNet


<!--
<a name="Relighting"></a>
# Relighting


<a name="Stitching"></a>
# Image Stitching - 图像拼接



<a name="ImageCompression"></a>
# Image Compression - 图像压缩

-->

<a name="ImageQualityAssessment"></a>
# Image Quality Assessment - 图像质量评价

**FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling**
- Paper: https://arxiv.org/abs/2207.02595
- Code: https://github.com/TimothyHTimothy/FAST-VQA

<!--
<a name="StyleTransfer"></a>
# Style Transfer - 风格迁移


<a name="ImageEditing"></a>
# Image Editing - 图像编辑

-->
<a name=ImageGeneration></a>
# Image Generation/Synthesis / Image-to-Image Translation - 图像生成/合成/转换

**ManiFest: Manifold Deformation for Few-shot Image Translation**
- Paper: https://arxiv.org/abs/2111.13681
- Code: https://github.com/cv-rits/ManiFest

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**
- Paper: https://arxiv.org/abs/2207.02196
- Code: https://github.com/fudan-zvg/PDS

**GAN Cocktail: mixing GANs without dataset access**
- Paper: https://arxiv.org/abs/2106.03847
- Code: https://github.com/omriav/GAN-cocktail

**Compositional Visual Generation with Composable Diffusion Models**
- Paper: https://arxiv.org/abs/2206.01714
- Code: https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch

**VecGAN: Image-to-Image Translation with Interpretable Latent Directions**
- Paper: https://arxiv.org/abs/2207.03411

**Image-Based CLIP-Guided Essence Transfer**
- Paper: https://arxiv.org/abs/2110.12427 
- Code: https://github.com/hila-chefer/TargetCLIP


<a name="VideoGeneration"></a>
## Video Generation

**Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer**
- Paper: https://arxiv.org/abs/2204.03638
- Code: https://github.com/SongweiGe/TATS

**Controllable Video Generation through Global and Local Motion Dynamics**
- Paper: 
- Code: https://github.com/Araachie/glass

<a name="Others"></a>
# Others

**Learning Local Implicit Fourier Representation for Image Warping**
- Paper: https://ipl.dgist.ac.kr/LTEW.pdf
- Code: https://github.com/jaewon-lee-b/ltew
- Tags: Image Warping

**Dress Code: High-Resolution Multi-Category Virtual Try-On**
- Paper: https://arxiv.org/abs/2204.08532
- Code: https://github.com/aimagelab/dress-code
- Tags: Virtual Try-On

**High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions**
- Paper: https://arxiv.org/abs/2206.14180
- Code: https://github.com/sangyun884/HR-VITON
- Tags: Virtual Try-On

<!--
****
- Paper: 
- Code: 
-->
