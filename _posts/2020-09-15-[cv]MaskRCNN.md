---
title: Mask R-CNN
categories: [DL,CV]
tags: [Deep Learning, CV]
excerpt: Image Segmentation, Mask R-CNN
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : Fastcampus 강의 )

# [ Mask R-CNN (2017) ]

<br>

# 1. R-CNN 계열

- Detection을 위한 : R-CNN, Fast R-CNN, Faster R-CNN

- Segmentation을 위한 : Mask R-CNN

***어떻게 하면 detection을 위한 R-CNN 모델들을 개선해서 "Segmentation"에 활용할 수 있을까?***

<br>

# 2. R-CNN 계열 복습

## (1) R-CNN

![figure2](/assets/img/cv/cv121.png)

**(1) Region Proposal**

- selective search를 사용하여 2000개의 bounding box 생성
- 엄청 오래 걸림

**(2) Warping**

- CNN의 input 크기에 맞게 이미지를 조정
- 왜곡이 발생할 수 밖에 없음

**(3) SVM classifier 사용**

<br>

## (2) Fast R-CNN

R-CNN의 문제점 : Multi-stage (총 3단계)로 나눠져 있다.

- 1) Feature Extractor
- 2) Classifier
- 3) Regressor

이를 **unified framework**로 만든 것이 Fast R-CNN!

![figure2](/assets/img/cv/cv122.png)

<br>

### ROI Pooling

- 2000개의 CNN이 아닌, 하나의 통합된 CNN을 위해 사용
- input image 사이즈를 맞춰주는 pooling ( 더 이상 warping X )

<br>

## (3) Faster R-CNN

Selective search 사용하면 너무 오래 걸림!

**Region Proposal Network 사용하자!**

- Anchor box 사용
- Objectness(0/1)를 판단하는 classifier
- Bounding Box를 찾는 regressor

<br>

# 3. Mask R-CNN

Segmentation을 하기 위해, Faster R-CNN을 변형한 모델이다.

- Objectiveness Binary Mask 
- ROI Allign

<br>

## (1) Objectiveness Binary Mask

Binary Mask

- Masking Layer ( 0과 1로 이루어진 행렬 )
- 픽셀 단위로, 해당 픽셀에 object가 있으면 1, 없으면 0
- class는 구분하지 않음

![figure2](/assets/img/cv/cv125.png)

- Input : Feature Map

- Output : 0/1로 이루어진 행렬 ( = binary mask )

<br>

## (2) RoI Alignment

Input Image가 cnn layer를 거쳐서 feature map으로 나오게 되면, 해상도가 줄어들게 된다.

Example

- Input Image : 128x128, 그 안에 15x15 RoI

- Feature Map : 25x25, 그 안에 $$(15 \times \frac{25}{128})$$ x $$(15 \times \frac{25}{128})$$ RoI ( = 2.93 x 2.93 )

  - 해당 RoI는 2.93픽셀을 차지하게 되는데, 이는 버림하여 2픽셀을 차지했었다. ( in RoI Pooling )

  - 0.93만큼의 정보 손실!

    $$\rightarrow$$ 이를 해결하기 위한 RoI Alignment

<br>

### Bilinear Interpolation

- 거리에 기반하여 weighted mean이라고 생각하면 된다.

![figure2](/assets/img/cv/cv126.png)

