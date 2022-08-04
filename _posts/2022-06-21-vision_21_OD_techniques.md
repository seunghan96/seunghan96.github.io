---
title: (CV summary) 21. Additional Object Detection Techniques
categories: [CV]
tags: []
excerpt: FPN, PANet, EfficientDet
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# Additional Object Detection Techniques

## 1. FPN (Feature Pyramid Network)

Challenge in Object Detection :

$$\rightarrow$$ ***vastly DIFFERENT SCALES***

<br>

Previous Solutions :

- ex 1) Image Pyramid
- ex 2) Feature Hierarchy

![figure2](/assets/img/cv/cv309.png)

<br>

### FPN (Feature Pyramid Network)

( Lin et al., Feature Pyramid Networks for Object Detection, CVPR 2017 )

![figure2](/assets/img/cv/cv310.png)

- add skip connections between each level!

- similar speed as **Feature Hierarchy**
- Feature Pyamid >> Image Pyramid

<br>

### Variations of Feature Pyramid

![figure2](/assets/img/cv/cv311.png)

<br>

## 2. PANet (Path Aggregation Network)

( Path Aggregation Network for Instance Segmentation, Liu et al., CVPR 2018 )

PANet = top-down path of FPN **+ bottom-up path**

![figure2](/assets/img/cv/cv312.png)

<br>

## 3. EfficientDet

( EfficientDet: Scalable and Efficient Object Detec`on, Tan et al., CVPR 2020 )

Efficient **Model Search**

By using **BiFPN structure**, propogate both

- (1) top-down info
- (2) bottom-up info

![figure2](/assets/img/cv/cv313.png)

![figure2](/assets/img/cv/cv314.png)

