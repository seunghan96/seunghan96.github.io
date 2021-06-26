---
title: Image Segmentation이란
categories: [DL,CV]
tags: [Deep Learning, CV]
excerpt: Image Segmentation
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : Fastcampus 강의 )

# [ Image Segmentation이란 ]

<br>

# 1. Introdunction to Image Segmentation

### (1) Image Segmentation의 활용 예시

![figure2](/assets/img/cv/cv99.png)

- 픽셀 단위로 정보를 얻어내야하는 분야들
- ex) 자율주행 자동차, 의학 이미지, 위성 사진

<br>

## (2) Goal

![figure2](/assets/img/cv/cv100.png)

![figure2](/assets/img/cv/cv101.png)

이미지의 (대략적인) Region 뿐만 아니라, (구체적인) Structure 또한 포착하는 것이 목적!

<br>

## (3) Old Segmentation Methods

- 1) Thresholding
  - grey scale로 변경하게 될 경우
  - 경계(edge) 부분에서 두드러져서 구분 가능
- 2) K-means
  - edge는 그룹 간의 경계가 되는 지점이므로, 이를 통해 비슷한 그룹끼리 묶음
- 3) Histogram-based Image Segmentation
- 4) Edge Detection
  - sharp change 부분 / discontinous 부분 잡아내기

<br>

# 2. Segmentation with DL

## (1) Encoder & Decoder

![figure2](/assets/img/cv/cv102.png)

- Down Sampling 후 Up Sampling 하여 구조를 잡아냄
- ex) FCN, SegNet, UNet

<br>

## (2) 대표적 구조

![figure2](/assets/img/cv/cv103.png)

Bottom-up

- 여러 후보를 두고, 이러한 region들을 **병합해나감**

Top-down

- 처음에는 큼직하게 잡아낸 뒤, 점차 **세밀화해나감**

<br>

## (3) Segmentation의 분류

### a) "Region Based" Semantic Segmentation

- Object detection의 결과를 기반으로함
- ex) SDS, Hypercolumns, Mask R-CNN

### b) Fully Convolutional Network-Based Semantic Segmentation

- Encoder-Decoder 구조
- region proposal을 따로 뽑아내지 않음
- ex) SegNet, DeepLab-CRF, Dilated Convolutions

### c) Weakly Supervised Semantic Segmentation

- annotated bounding box 사용
- ex) Boxsup

<br>

아래의 그림은 차례대로 a),b),c) 방법이다.

![figure2](/assets/img/cv/cv104.png)