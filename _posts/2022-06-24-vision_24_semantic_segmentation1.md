---
title: (CV summary) 24. Semantic Segmentation - FCN, DeepLab, DeconvNet, UNet
categories: [CV]
tags: []
excerpt: FCN, DeepLab, DeconvNet, UNet
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# Semantic Segmentation - FCN, DeepLab, DeconvNet, UNet

<br>

## 1. FCN (Fully Convolutional Networks)

( Fully convoluKonal networks for semanKc segmentaKon, Long et al., CVPR 2015 )

Summary

- End-to-End architecture for **semantic segmentation**
- use **ONLY CNN layers**

![figure2](/assets/img/cv/cv326.png)

<br>

Output of FCN

- tensor with **spatial information**

  ( no flattening .... preserve spatial information )

- interpretation : **class score** over local image regions

<br>

Limitation

- Predicted score map in a very low-resolution

$$\rightarrow$$ solution : enlarge the score map

<br>

Enlarging score map

- (1) add simple **bilinear interpolation** on the top of the netowkr
- (2) ***add skip connection***

![figure2](/assets/img/cv/cv327.png)

<br>

Advantages : (1) Faster & (2) Accurate

- (1) Faster :
  - end-to-end
  - do not rely on off-the-shelf proposal
- (2) Accurate :
  - not bounded by the quality of proposal

<br>

## 2. DeepLab

( DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs., Chen et al., TPAMI 2017 )

Summary

- Deep Lab = Advanced version of FCN

- use **Atrous Convolution**
- post-processing technique : **Fully-Connected CRF ( Conditional Random Field )**

- backbone : ResNet-101

<br>

### (1) Atrous Convolution ( = Dilated Convolution )

![figure2](/assets/img/cv/cv328.png)

- trous = "hole"

  $$\rightarrow$$ convolution kernel with holes

- additional parameter : dilation rate $$r$$

- enlarge receptive field

  ( w.o max-pooling )

- Atrous Spatial Pyramid Pooling (ASPP)

  $$\rightarrow$$ captures **multi-scale**

![figure2](/assets/img/cv/cv330.png)

![figure2](/assets/img/cv/cv331.png)

<br>

### (2) Fully-Connected CRF ( Conditional Random Field )

![figure2](/assets/img/cv/cv329.png)

Energy Minimization

- energy = composed of **unary & pairwise potentials**
- $$E(\boldsymbol{x})=\sum_{i} \phi_{i}\left(x_{i}\right)+\sum_{i j} \psi_{i j}\left(x_{i}, x_{j}\right)$$.
  - unary potential : output scores
  - pairwise potential : $$\psi_{i j}\left(x_{i}, x_{j}\right)=I\left(x_{i}, x_{j}\right)\left[w_{1} \exp \left(-\frac{ \mid \mid p_{i}-p_{j} \mid \mid ^{2}}{2 \sigma_{\alpha}^{2}}-\frac{ \mid \mid c_{i}-c_{j} \mid \mid ^{2}}{2 \sigma_{\beta}^{2}}\right)+w_{2} \exp \left(-\frac{ \mid \mid p_{i}-p_{j} \mid \mid ^{2}}{2 \sigma_{\gamma}^{2}}\right)\right]$$.

<br>

## 3. DeconvNet ( Deconvolution Net )

( Learning deconvolution network for semantic segmentation., Noh et al., ICCV 2015 )

Summary

- convolutional ENC & DEC

- ENCODER

  - convolution & max pooling

- DECODER

  - deconvolution & un-pooling

  ( = mirrored version of ENCODER )

![figure2](/assets/img/cv/cv332.png)

<br>

## (1) Deconvolution & Un-pooling

![figure2](/assets/img/cv/cv333.png)

Un-pooling

- Activation maps are UP-sampled

Deconvolution

- coarse activation maps are **densified**

![figure2](/assets/img/cv/cv334.png)

<br>

## 4. U-Net

- https://seunghan96.github.io/dl/cv/cv-UNet/



###### 
