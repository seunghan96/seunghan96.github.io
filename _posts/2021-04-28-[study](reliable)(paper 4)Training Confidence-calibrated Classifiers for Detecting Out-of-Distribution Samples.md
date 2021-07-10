---
title: \[reliable\] (paper 4) Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: Confidence-calibrated Classifiers, Out-of-Distribution
---

# Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Training Confident Neural Classifiers
   1. Confident Classifier
   2. Adversarial Generator
   3. Joint Training of Confident Classifier & Adversarial Generator

<br>

# 0. Abstract

배경 소개

- detect whether a **test sample is from in-distribution**
- DNN의 문제점 : **highly overconfident**
- 이에 대한 해결책으로 **threshold-based detector**가 제안됨

<br>threshold-based detector의 문제점

- highly depend on **HOW TO TRAIN** the classifiers

  ( $$\because$$ only focus on improving inference procedures )

<br>

이 논문에서는, novel training method for classifier를 제안함

( inference algorithm이 더 잘 작동하도록! )

<br>

기존 (cross entropy) loss에, 2가지 term을 추가할 것을 제안

- 1) first term : force samples from ***o.o.d to be less confident***
- 2) second term : ***generate most effective training samples*** for first term

<br>

# 1. Introduction

Notation

- $$P_{\text {out }}(\mathrm{x})$$ : o.o.d
- $$P_{\text {in }}(\mathrm{x}, y)$$ : i.d

<br>

Goal

- determining if input $$\mathrm{x}$$ is from $$P_{\text {in }}$$ or $$P_{\text {out }}$$

  ( possibly utilizing a well calibrated classifier $$P_{\theta}(y \mid \mathrm{x})$$ )

- 즉 detector $$g(\mathrm{x}): \mathcal{X} \rightarrow\{0,1\}$$ 만들기!

  - 1 if in distribution
  - 0 if out of distribution

<br>

기존의 threshold based detector

- use **pre-trained classifier**
- input $$\mathrm{x}$$에 대해, confidence score $$q(\mathrm{x})$$ 를 ( pre-trained classifier 를 사용하여 ) 계산
- 이 $$q(\mathrm{x})$$와 threshold $$\delta>0$$를 비교하여 o.o.d 판단

- 장점) computationally simple

- 단점) highly depend on pre-trained classifier

  ( fail to work, if classifier does not separate maximum value of predictive distribution well enough! )

<br>

### Contribution

Goal : **detect o.o.d** ( + classification 성능 낮추지 않으면서도 )

- 1) **confidence loss** 제안
  - KL divergence of (1) & (2)
    - (1) predictive distribution of o.o.d samples
    - (2) uniform distribution
- 2) GAN 사용하여 $$P_{\text {out }}(\mathrm{x})$$ 를 모델링 한 뒤, o.o.d sample들을 뽑는다.

<br>

# 2. Training Confident Neural Classifiers

## 2-1) Confident Classifier

아래의 새로운 **confidence loss** ( = CE loss + KL )를 제안한다.

- $$\min _{\theta} \mathbb{E}_{P_{\text {in }}(\widehat{\mathbf{x}}, \widehat{y})}\left[-\log P_{\theta}(y=\widehat{y} \mid \widehat{\mathbf{x}})\right]+\beta \mathbb{E}_{P_{\text {out }}(\mathbf{x})}\left[K L\left(\mathcal{U}(y) \| P_{\theta}(y \mid \mathbf{x})\right)\right]$$.

  - $$\mathcal{U}(y)$$ : uniform distribution
  - $$\beta>0$$ : penalty parameter

- 직관적 의미 :

  ***force predictive distribution of o.o.d to be close to UNIFORM***

<br>

위 loss function의 $$KL$$ term을 계산하기 위해서는, o.o.d 분포에서의 샘플들이 매우 많이 필요.

( 하지만 현실에서는 그렇게 구하기 쉽지 않음 )

$$\because$$ **effective**하게 샘플을 모으자!

- effective하다 = ***in-distribution과 가깝게 o.o.d에서 샘플하자***
- 아래의 그림을 통해 직관적인 이해 가능

![figure2](/assets/img/reli/img7.png)

<br>

## 2-2) Adversarial Generator

**o.o.d 를 생성하기 위한 GAN**을 모델링한다!

Notation

- Discriminator & Generator : $$D$$ & $$G$$
- prior distribution of latent variable $$\mathbf{z}$$ : $$P_{\mathrm{pri}}(\mathrm{z})$$
- generated outputs : $$G(\mathbf{z})$$
- in-distribution : $$P_{\mathrm{in}}(x)$$

<br>

### Loss function ( 목표 : $$P_{G} \approx$$  $$P_{\mathrm{in}}$$ )

(1) 기존의 Loss function

- $$\min _{G} \max _{D} \mathbb{E}_{P_{\text {in }}(\mathbf{x})}[\log D(\mathbf{x})]+\mathbb{E}_{P_{\text {pri }}(\mathbf{z})}[\log (1-D(G(\mathbf{z})))]$$.

(2) 제안한 Loss function

- want to make the generator recover an **effective** out-of distribution $$P_{\text {out }}$$ 

  ( 위에서 말했 듯, effective하다 = ***in-distribution과 가깝게 o.o.d에서 샘플하자*** )

- $$\begin{aligned}
  \min _{G} \max _{D} & \beta \underbrace{\mathbb{E}_{P_{G}(\mathbf{x})}\left[K L\left(\mathcal{U}(y) \| P_{\theta}(y \mid \mathbf{x})\right)\right]}_{(\mathrm{a})} \\
  &+\underbrace{\mathbb{E}_{P_{\text {in }}(\mathbf{x})}[\log D(\mathbf{x})]+\mathbb{E}_{P_{G}(\mathbf{x})}[\log (1-D(\mathbf{x}))]}_{(\mathrm{b})}
  \end{aligned}$$.
- 직관적 의미 : (a) + (b)
  - (a) **force predictive distribution of o.o.d to be close to UNIFORM**
  - (b) **기존의 Loss function ( 위의 (1) 식 )**
    - out of distribution이 in distribution과 유사하도록 유도!

<br>

## 2-3) Joint Training of (1) Confident Classifier & (2) Adversarial Generator

위에서 배운 2-1) & 2-2) 두 모델을 **jointly train**

최종적인 Loss function :

![figure2](/assets/img/reli/img8.png)

- (1) Confidence Loss ( 빨간색 ) : (c) + (d)

- (2) GAN loss ( 파란색 ) : (d) + (e)

  ( (d)는 (1), (2)의 중첩되는 부분 )

<br>

### 최종적인 Algorithm

![figure2](/assets/img/reli/img9.png)

