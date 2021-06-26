---
title: ZFNet & Network in Network
categories: [DL,CV]
tags: [Deep Learning, CV]
excerpt: LeNet, AlexNet
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ ZFNet & Network in Network ]

# 1. ZFNet

**AlexNet**의 변형 ( for 보다 효율적인 구조)

## (1) De-convolution

- de-convolution을 수행해서, feature map이 잘 학습되는지 확인
- 과정) **Unpooling → ReLU → Deconvolution**
- convolution filter의 **transpose**

<br>

**Convolution** 연산은, 아래 그림과 같이 **Matrix Multiplication으로 변형**해서 표현할 수 있다.

( 출처 : https://analysisbugs.tistory.com/104 )

![figure2](/assets/img/cv/cv11.png)

<br>

위의 **Sparse Matrix $$C$$의 transpose**를 $$Y$$에 곱하면, $$X$$를 복원할 수 있는데, 이를 **Deconvolution**이라고 한다

![figure2](/assets/img/cv/cv12.png)

<br>

## (2) UnPooling

- **Maxpooling된 지점을 저장** 한 뒤, Unpooling할 때 해당 지점에서 **재생성**

![figure2](/assets/img/cv/cv10.png)

<br>

## (3) Overall Architecture

![figure2](/assets/img/cv/cv13.png)

<br>

## (4) Layers

![figure2](/assets/img/cv/cv14.png)

**Layer 1,2**

- low level feature ( edge, color 등 )
- 비슷한 class에 대해 invariant

**Layer 3**

- middle level features ( texture 등 )
- 보다 정교한/세밀한 것 포착

**Layer 4,5**

- high level feature ( 개체의 일부분, 위치, 자세 등 )
- 가장 세밀한 부분 포착

<br>

## Summary

1. **Rotation, Scaling, Translation에 robust**하다!
2. Image의 **일부를 가려도**, output에 민감하게 변화 X
3. AlexNet과의 차이점 : **1개의 GPU만**을 사용 + 70 epoch + 12일

<br>

# 2. Network in Network

*GoogLeNet에 아이디어를 제공한 알고리즘이다*

complex structure를 포착하기 위해, **네트워크 내에 네트워크를 추가** ( convolution을 MLP로 대체 )

![figure2](/assets/img/cv/cv15.png)

<br>

CNN vs NIN

- CNN) **linear** filter dot product
- NIN) **NON-linear** MLP

<br>

## (1) Global Average Pooling 

- (출력층 직전의) **FCNN은 parameter 수를 증가시키는** 요인!

  $$\rightarrow$$ 이 대신에 **그냥 CNN을 사용**한 뒤, **global average pooling**을 사용하여 바로 softmax로 넘김 

  ( overfitting 방지 효과 )

- Feature Map의 개수를, classification되는 output의 개수와 일치시키기

![figure2](/assets/img/cv/cv17.png)

<br>

## (2) Overall Architecture

![figure2](/assets/img/cv/cv16.png)

<br>

# 3. GoogLeNet의 1x1 conv

![figure2](/assets/img/cv/cv18.png)

- **Bottle Neck** 구조라고도 함
- **dimension reduction**의 효과
- 1x1 conv : **MLP**와 비슷한 효과

