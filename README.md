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
-->
- [Dehazing](#Dehazing)

- [Demoireing](#Demoireing)
<!--
- [Demosaicing](#Demosaicing)
-->
- [HDR Imaging / Multi-Exposure Image Fusion](#HDR)

- [Frame Interpolation](#FrameInterpolation)
  - [Spatial-Temporal Video Super-Resolution](#STVSR)
  
- [Image Enhancement](#Enhancement)
  - [Low-Light Image Enhancement](#LowLight)

- [Image Harmonization](#Harmonization)

- [Image Completion/Inpainting](#Inpainting)

- [Image Colorization](#Colorization)

- [Image Matting](#Matting)

- [Shadow Removal](#ShadowRemoval)

<!--
- [Image Stitching](#Stitching)
-->

- [Image Compression](#ImageCompression)

- [Image Quality Assessment](#ImageQualityAssessment)

- [Style Transfer](#StyleTransfer)

- [Image Editing](#ImageEditing)

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

**Seeing Far in the Dark with Patterned Flash**
- Paper: 
- Code: https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash

**BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks**
- Paper: https://arxiv.org/abs/2207.06873
- Code: https://github.com/ExplainableML/BayesCap

**Improving Image Restoration by Revisiting Global Information Aggregation**
- Paper: https://arxiv.org/abs/2112.04491
- Code: https://github.com/megvii-research/TLC

**Fast Two-step Blind Optical Aberration Correction**
- Paper: https://arxiv.org/abs/2208.00950
- Code: https://github.com/teboli/fast_two_stage_psf_correction
- Tags: Optical Aberration Correction

**VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder**
- Paper: https://arxiv.org/abs/2205.06803
- Code: https://github.com/TencentARC/VQFR
- Tags: Blind Face Restoration

**Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model**
- Paper: https://arxiv.org/abs/2207.10040
- Code: https://github.com/VITA-Group/TurbNet
- Tags: Atmospheric Turbulence Mitigation, Transformer

**Modeling Mask Uncertainty in Hyperspectral Image Reconstruction**
- Paper: https://arxiv.org/abs/2112.15362
- Code: https://github.com/Jiamian-Wang/mask_uncertainty_spectral_SCI
- Tags: Hyperspectral Image Reconstruction

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
- Paper: https://arxiv.org/abs/2207.09156
- Code: https://github.com/palmdong/MMSR
- Tags: Self-Supervised

**Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations**
- Paper: https://arxiv.org/abs/2203.01325
- Code: https://github.com/cszhilu1998/SelfDZSR
- Tags: Self-Supervised

**CADyQ : Contents-Aware Dynamic Quantization for Image Super Resolution**
- Paper: https://arxiv.org/abs/2207.10345
- Code: https://github.com/Cheeun/CADyQ

**From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution**
- Paper: 
- Code: https://github.com/csxmli2016/ReDegNet

**Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images**
- Paper:
- Code: https://github.com/HaomingCai/SRPO

**Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution**
- Paper: https://arxiv.org/abs/2207.12987
- Code: https://github.com/zhjy2016/SPLUT

**KXNet: A Model-Driven Deep Neural Network for Blind Super-Resolution**
- Paper:
- Code: https://github.com/jiahong-fu/KXNet

**Image Super-Resolution with Deep Dictionary**
- Paper: https://arxiv.org/abs/2207.09228
- Code: https://github.com/shuntama/srdd

**Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution**
- Paper: http://www4.comp.polyu.edu.hk/~cslzhang/paper/ECCV2022_DASR.pdf
- Code: https://github.com/csjliang/DASR

**Adaptive Patch Exiting for Scalable Single Image Super-Resolution**
- Paper:
- Code: https://github.com/littlepure2333/APE

**Efficient Long-Range Attention Network for Image Super-resolution**
- Paper: https://arxiv.org/abs/2203.06697
- Code: https://github.com/xindongzhang/ELAN

**D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution**
- Paper: https://arxiv.org/abs/2103.14373
- Code: https://github.com/megvii-research/D2C-SR
- Tag: Real-World

**Reference-based Image Super-Resolution with Deformable Attention Transformer**
- Paper: https://arxiv.org/abs/2207.11938
- Code: https://github.com/caojiezhang/DATSR
- Tags: Reference-based, Transformer

<a name="VideoSuperResolution"></a>
## Video Super Resolution

**Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution**
- Paper: https://arxiv.org/abs/2208.03012
- Code: https://github.com/researchmm/FTVSR
- Tags: Compressed Video SR

<!--
<a name="Rescaling"></a>
# Image Rescaling - 图像缩放

-->
<a name="Denoising"></a>
# Denoising - 去噪

<a name="ImageDenoising"></a>
## Image Denoising

**Deep Semantic Statistics Matching (D2SM) Denoising Network**
- Paper: https://arxiv.org/abs/2207.09302
- Code: https://github.com/MKFMIKU/d2sm

<a name="VideoDenoising"></a>
## Video Denoising

**Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones**
- Paper: 
- Code: https://github.com/nagejacob/FloRNN


<a name="Deblurring"></a>
# Deblurring - 去模糊
<a name="ImageDeblurring"></a>
## Image Deblurring

**Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance**
- Paper: https://arxiv.org/abs/2207.10123
- Code: https://github.com/zzh-tech/Animation-from-Blur
- Tags: recovering detailed motion from a single motion-blurred image

**Event-based Fusion for Motion Deblurring with Cross-modal Attention**
- Paper:https://arxiv.org/abs/2112.00167
- Code: https://github.com/AHupuJR/EFNet
- Tags: Event-based

<a name="VideoDeblurring"></a>
## Video Deblurring

**Spatio-Temporal Deformable Attention Network for Video Deblurring**
- Paper: https://arxiv.org/abs/2207.10852
- Code: https://github.com/huicongzhang/STDAN

**Efficient Video Deblurring Guided by Motion Magnitude**
- Paper: https://arxiv.org/abs/2207.13374
- Code: https://github.com/sollynoay/MMP-RNN

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
-->

<a name="Dehazing"></a>
# Dehazing - 去雾

**Frequency and Spatial Dual Guidance for Image Dehazing**
- Paper: 
- Code: https://github.com/yuhuUSTC/FSDGN 

**Perceiving and Modeling Density is All You Need for Image Dehazing**
- Paper: https://arxiv.org/abs/2111.09733
- Code: https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing

<a name="Demoireing"></a>
# Demoireing - 去摩尔纹

**Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing**
- Paper: https://arxiv.org/abs/2207.09935
- Code: https://github.com/XinYu-Andy/uhdm-page

<!--
<a name="Demosaicing"></a>
# Demosaicing - 去马赛克

-->

 <a name="HDR"></a>
# HDR Imaging / Multi-Exposure Image Fusion - HDR图像生成 / 多曝光图像融合

**Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging**
- Paper: 
- Code: https://github.com/viengiaan/EDWL

**Ghost-free High Dynamic Range Imaging with Context-aware Transformer**
- Paper:
- Code: https://github.com/megvii-research/HDR-Transformer


<a name="FrameInterpolation"></a>
# Frame Interpolation - 插帧

**Real-Time Intermediate Flow Estimation for Video Frame Interpolation**
- Paper: https://arxiv.org/abs/2011.06294
- Code: https://github.com/hzwer/ECCV2022-RIFE 

**FILM: Frame Interpolation for Large Motion**
- Paper: https://arxiv.org/abs/2202.04901
- Code: https://github.com/google-research/frame-interpolation

<a name="STVSR"></a>
## Spatial-Temporal Video Super-Resolution

**Towards Interpretable Video Super-Resolution via Alternating Optimization**
- Paper: https://arxiv.org/abs/2207.10765
- Code: https://github.com/caojiezhang/DAVSR


<a name="Enhancement"></a>
# Image Enhancement - 图像增强

**Local Color Distributions Prior for Image Enhancement**
- Paper: https://www.cs.cityu.edu.hk/~rynson/papers/eccv22b.pdf
- Code: https://github.com/hywang99/LCDPNet

**SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement**
- Paper: https://arxiv.org/abs/2207.08351

**Neural Color Operators for Sequential Image Retouching**
- Paper: https://arxiv.org/abs/2207.08080
- Code: https://github.com/amberwangyili/neurop

**Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression**
- Paper: https://arxiv.org/abs/2207.10564
- Code: https://github.com/jinyeying/night-enhancement

<a name="LowLight"></a>
## Low-Light Image Enhancement

**LEDNet: Joint Low-light Enhancement and Deblurring in the Dark**
- Paper: https://arxiv.org/abs/2202.03373
- Code: https://github.com/sczhou/LEDNet


<a name="Harmonization"></a>
# Image Harmonization - 图像协调

**Harmonizer: Learning to Perform White-Box Image and Video Harmonization**
- Paper: https://arxiv.org/abs/2207.01322
- Code: https://github.com/ZHKKKe/Harmonizer

**DCCF: Deep Comprehensible Color Filter Learning Framework for High-Resolution Image Harmonization**
- Paper: https://arxiv.org/abs/2207.04788
- Code: https://github.com/rockeyben/DCCF

<a name="Inpainting"></a>
# Image Completion/Inpainting - 图像修复

**Learning Prior Feature and Attention Enhanced Image Inpainting**
- Paper: https://arxiv.org/abs/2208.01837
- Code: https://github.com/ewrfcas/MAE-FAR

**Perceptual Artifacts Localization for Inpainting**
- Paper:
- Code: https://github.com/owenzlz/PAL4Inpaint

## Video Inpainting

**Error Compensation Framework for Flow-Guided Video Inpainting**
- Paper: https://arxiv.org/abs/2207.10391

**Flow-Guided Transformer for Video Inpainting**
- Paper: 
- Code: https://github.com/hitachinsk/FGT


<a name="Colorization"></a>
# Image Colorization - 图像上色

**Eliminating Gradient Conflict in Reference-based Line-art Colorization**
- Paper: https://arxiv.org/abs/2207.06095
- Code: https://github.com/kunkun0w0/SGA

**Bridging the Domain Gap towards Generalization in Automatic Colorization**
- Paper: 
- Code: https://github.com/Lhyejin/DG-Colorization

<a name="Matting"></a>
# Image Matting - 图像抠图

**TransMatting: Enhancing Transparent Objects Matting with Transformers**
- Paper: https://arxiv.org/abs/2208.03007
- Code: https://github.com/AceCHQ/TransMatting


**One-Trimap Video Matting**
- Paper: https://arxiv.org/abs/2207.13353
- Code: https://github.com/Hongje/OTVM


<a name="ShadowRemoval"></a>
# Shadow Removal - 阴影消除

**Style-Guided Shadow Removal**
- Paper:
- Code: https://github.com/jinwan1994/SG-ShadowNet


<!--


<a name="Stitching"></a>
# Image Stitching - 图像拼接

-->

<a name="ImageCompression"></a>
# Image Compression - 图像压缩

**Optimizing Image Compression via Joint Learning with Denoising**
- Paper: https://arxiv.org/abs/2207.10869
- Code: https://github.com/felixcheng97/DenoiseCompression

**Implicit Neural Representations for Image Compression**
- Paper: https://arxiv.org/abs/2112.04267
- Code：https://github.com/YannickStruempler/inr_based_compression

**Expanded Adaptive Scaling Normalization for End to End Image Compression**
- Paper: https://arxiv.org/abs/2208.03049

## Video Compression

**AlphaVC: High-Performance and Efficient Learned Video Compression**
- Paper: https://arxiv.org/abs/2207.14678

<a name="ImageQualityAssessment"></a>
# Image Quality Assessment - 图像质量评价

**FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling**
- Paper: https://arxiv.org/abs/2207.02595
- Code: https://github.com/TimothyHTimothy/FAST-VQA

**Shift-tolerant Perceptual Similarity Metric**
- Paper: https://arxiv.org/abs/2207.13686
- Code: https://github.com/abhijay9/ShiftTolerant-LPIPS/

**Telepresence Video Quality Assessment**
- Paper: https://arxiv.org/abs/2207.09956


<a name="StyleTransfer"></a>
# Style Transfer - 风格迁移

**CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer**
- Paper: https://arxiv.org/abs/2207.04808
- Code: https://github.com/JarrentWu1031/CCPL

**Image-Based CLIP-Guided Essence Transfer**
- Paper: https://arxiv.org/abs/2110.12427 
- Code: https://github.com/hila-chefer/TargetCLIP

**Learning Graph Neural Networks for Image Style Transfer**
- Paper: https://arxiv.org/abs/2207.11681

**WISE: Whitebox Image Stylization by Example-based Learning**
- Paper: https://arxiv.org/abs/2207.14606
- Code: https://github.com/winfried-loetzsch/wise

<a name="ImageEditing"></a>
# Image Editing - 图像编辑

**Context-Consistent Semantic Image Editing with Style-Preserved Modulation**
- Paper: https://arxiv.org/abs/2207.06252
- Code: https://github.com/WuyangLuo/SPMPGAN

**GAN with Multivariate Disentangling for Controllable Hair Editing**
- Paper: https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/main/database/CtrlHair/CtrlHair.pdf
- Code: https://github.com/XuyangGuo/CtrlHair

**Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing**
- Paper:
- Code: https://github.com/1jsingh/paint2pix

**High-fidelity GAN Inversion with Padding Space**
- Paper: https://arxiv.org/abs/2203.11105
- Code: https://github.com/EzioBy/padinv

**Text2LIVE: Text-Driven Layered Image and Video Editing**
- Paper: https://arxiv.org/abs/2204.02491
- Code: https://github.com/omerbt/Text2LIVE

<a name=ImageGeneration></a>
# Image Generation/Synthesis / Image-to-Image Translation - 图像生成/合成/转换

**TIPS: Text-Induced Pose Synthesis**
- Paper: https://arxiv.org/abs/2207.11718
- Code: https://github.com/prasunroy/tips

**TISE: A Toolbox for Text-to-Image Synthesis Evaluation**
- Paper: https://arxiv.org/abs/2112.01398
- Code: https://github.com/VinAIResearch/tise-toolbox

**Learning Visual Styles from Audio-Visual Associations**
- Paper: https://arxiv.org/abs/2205.05072
- Code: https://github.com/Tinglok/avstyle

**End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement**
- Paper: https://arxiv.org/abs/2207.13268

**ManiFest: Manifold Deformation for Few-shot Image Translation**
- Paper: https://arxiv.org/abs/2111.13681
- Code: https://github.com/cv-rits/ManiFest

**VecGAN: Image-to-Image Translation with Interpretable Latent Directions**
- Paper: https://arxiv.org/abs/2207.03411

**DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation**
- Paper: https://arxiv.org/abs/2207.06124
- Code: https://github.com/Huage001/DynaST

**Cross Attention Based Style Distribution for Controllable Person Image Synthesis**
- Paper: https://arxiv.org/abs/2208.00712
- Code: https://github.com/xyzhouo/CASD

**EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer**
- Paper: https://arxiv.org/abs/2207.09840
- Code: https://github.com/Chenyu-Yang-2000/EleGANt

**Vector Quantized Image-to-Image Translation**
- Paper:
- Code: https://github.com/cyj407/VQ-I2I

**URUST: Ultra-resolution unpaired stain transformation via Kernelized Instance Normalization**
- Paper:
- Code: https://github.com/Kaminyou/URUST

**General Object Pose Transformation Network from Unpaired Data**
- Paper:
- Code: https://github.com/suyukun666/UFO-PT

**Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment**
- Paper: 
- Code: https://github.com/Taeu/Style-Your-Hair

**StyleLight: HDR Panorama Generation for Lighting Estimation and Editing**
- Paper: https://arxiv.org/abs/2207.14811
- Code: https://github.com/Wanggcong/StyleLight

**Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling**
- Paper: https://arxiv.org/abs/2207.02196
- Code: https://github.com/fudan-zvg/PDS

**GAN Cocktail: mixing GANs without dataset access**
- Paper: https://arxiv.org/abs/2106.03847
- Code: https://github.com/omriav/GAN-cocktail

**Compositional Visual Generation with Composable Diffusion Models**
- Paper: https://arxiv.org/abs/2206.01714
- Code: https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch

**Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation**
- Paper: https://arxiv.org/abs/2112.02450
- Code: https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation

**StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pretrained StyleGAN**
- Paper: https://arxiv.org/abs/2203.04036
- Code: https://github.com/FeiiYin/StyleHEAT

**WaveGAN: An Frequency-aware GAN for High-Fidelity Few-shot Image Generation**
- Paper: https://arxiv.org/abs/2207.07288
- Code: https://github.com/kobeshegu/ECCV2022_WaveGAN

**Supervised Attribute Information Removal and Reconstruction for Image Manipulation**
- Paper: https://arxiv.org/abs/2207.06555
- Code: https://github.com/NannanLi999/AIRR

**FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs**
- Paper: https://arxiv.org/abs/2207.08630
- Code: https://github.com/iceli1007/FakeCLR

**Auto-regressive Image Synthesis with Integrated Quantization**
- Paper: https://arxiv.org/abs/2207.10776
- Code: https://github.com/fnzhan/IQ-VAE

**PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation**
- Paper: https://arxiv.org/abs/2204.00833
- Code: https://github.com/BlingHe/PixelFolder

**DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta**
- Paper: https://arxiv.org/abs/2207.10271
- Code: https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation

**Generator Knows What Discriminator Should Learn in Unconditional GANs**
- Paper: https://arxiv.org/abs/2207.13320
- Code: https://github.com/naver-ai/GGDR

<a name="VideoGeneration"></a>
## Video Generation

**Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer**
- Paper: https://arxiv.org/abs/2204.03638
- Code: https://github.com/SongweiGe/TATS

**Controllable Video Generation through Global and Local Motion Dynamics**
- Paper: 
- Code: https://github.com/Araachie/glass

**Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis**
- Paper: https://arxiv.org/abs/2207.05049
- Code: https://github.com/fast-vid2vid/fast-vid2vid

**Synthesizing Light Field Video from Monocular Video**
- Paper: https://arxiv.org/abs/2207.10357
- Code: https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video

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

**Single Stage Virtual Try-on via Deformable Attention Flows**
- Paper: https://arxiv.org/abs/2207.09161
- Tags: Virtual Try-On

**Outpainting by Queries**
- Paper: https://arxiv.org/abs/2207.05312
- Code: https://github.com/Kaiseem/QueryOTR
- Tags: Outpainting

**Geometry-aware Single-image Full-body Human Relighting**
- Paper: https://arxiv.org/abs/2207.04750

**NeRF for Outdoor Scene Relighting**
- Paper: https://arxiv.org/abs/2112.05140
- Code: https://github.com/r00tman/NeRF-OSR

**Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**
- Paper: https://arxiv.org/abs/2207.08178
- Code: https://github.com/thinwayliu/Watermark-Vaccine
- Tags: Watermark Protection

**Efficient Meta-Tuning for Content-aware Neural Video Delivery**
- Paper: https://arxiv.org/abs/2207.09691
- Code: https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022
- Tags: Video Delivery

**Human-centric Image Cropping with Partition-aware and Content-preserving Features**
- Paper: https://arxiv.org/abs/2207.10269
- Code: https://github.com/bcmi/Human-Centric-Image-Cropping

**CelebV-HQ: A Large-Scale Video Facial Attributes Dataset**
- Paper: https://arxiv.org/abs/2207.12393
- Code: https://github.com/CelebV-HQ/CelebV-HQ
- Tags: Dataset

**Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis**
- Paper: https://arxiv.org/abs/2207.11770
- Code: https://github.com/sstzal/DFRF
- Tags: Talking Head Synthesis

**Contrastive Monotonic Pixel-Level Modulation**
- Paper: https://arxiv.org/abs/2207.11517
- Code: https://github.com/lukun199/MonoPix

**AutoTransition: Learning to Recommend Video Transition Effects**
- Paper: https://arxiv.org/abs/2207.13479
- Code: https://github.com/acherstyx/AutoTransition


<!--
****
- Paper: 
- Code: 
-->
