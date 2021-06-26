---
title: LeNet & AlexNet
categories: [DL,CV]
tags: [Deep Learning, CV]
excerpt: LeNet, AlexNet
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ LeNet & AlexNet ]

<br>

# 1. CNN의 등장 ( + MLP의 문제점 )

이미지 인식에 있어서, MLP의 경우 **input 값이 조금만 변해도 output이 크게** 변함

- ex) 숫자 1을 살짝만 회전해도, 1로 인식 못할 수도!

<br>

**[ CNN의 세 가지 step ]**

![figure2](/assets/img/cv/cv6.png)

**1) Feature Extraction : 특징 추출**

- receptive field & convolutional filter

**2) Shift and Distortion Invarance**

- topology 변화에 영향 X기 위해
- Pooling ( Max/Average Pooling 등 )

**3) Classification**

- FCNN사용하여 최종 분류 ( softmax )

<br>

# 2. LeNet

- Yann LeCun et al. in 1989
- CNN의 시초
- (1) LeNet1 :
  - input size : 28x28
  - **average pooling** 사용
  - FCNN 1개
  - 1.7% error rate
- (2) LeNet4 :
  - input size : 32x32 
  - average pooling 사용
  - FCNN 2개
  - 1.1% error rate
- (3) LeNet5 :
  - LeNet4에 FCNN 1개 추가
  - 0.95% error rate
- (4) Boosted LeNet4:
  - **여러 LeNet을 ensemble**
  - 0.7% error rate

<br>

![figure2](/assets/img/cv/cv7.png)

![figure2](/assets/img/cv/cv8.png)

<br>

# 3. AlexNet

## 1) ILSVRC

- ImageNet Large Scale Visual Recognition Challenge ( ImageNet 대회 )
- 3개 분야의 대회
  - **1) Image Classification**
  - **2) Single Object Localization**
    - "하나"의 물체, (1) 물체가 무엇인지 + (2) 어디에 있는지
  - **3) Object Detection**
    - "여러 개"의 물체, (1) 물체가 무엇인지 + (2) 어디에 있는지

<br>

## 2) AlexNet

- Alex Krizhevsky et al, 2012

- ILSVRC의 첫 번째 우승자

![figure2](/assets/img/cv/cv9.png)

<br>

### 핵심 특징

1. **병렬 구조** : 2개의 group일 때 가장 좋은 결과

2. **ReLU 사용**
- (LeNet5) Tanh vs (AlexNet) ReLU
   
- DEEPER network 가능
   - **Tanh보다 6배정도 빠름**
   
3. **Dropout** : overfitting 방지

4. **Max Pooling**
- Average Pooling보다 좋은 효과
   - Overlapping Pooling (kernel들이 겹치게끔)
   
5. **Data Augmentation** : 대량의 데이터 확보를 위해
- ex) *scale, cropping, mirroring...*
   
6. **Local Response Normalization (LRN)**
- "강한 자극은 잘 보이도록, 약한 자극은 잘 안보이도록"
   - $$b_{x, y}^{i}=a_{x, y}^{i} /\left(k+\alpha \sum_{j=\max (0, i-n / 2)}^{\min (N-1, i+n / 2)}\left(a_{x, y}^{j}\right)^{2}\right)^{\beta}$$.