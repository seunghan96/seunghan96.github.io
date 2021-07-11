---
title: \[reliable\] (paper 9) Augmix \: A simple data preprocessing method to improve robustness and uncertainty
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: Augmix
---

# Augmix : A simple data preprocessing method to improve robustness and uncertainty

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Related Work
   1. Robustness under data shift
   2. Calibration under data shift
   3. Data Augmentation
3. Augmix

<br>

# 0. Abstract

- 실제 test data는 i.i.d가 아니다!

- 최근 들어서, unforeseen data shifts에 대해 robustness를 증가시키는 여러 방법론들이 나왔다.

- 이 논문도 이러한 맥락에서 **Augmix**를 제안한다

<br>

# 1. Introduction

배경

- mismatch between Train & Test data
- modern moddel = OVERCONFIDENT predictions

<br>

data distn에 corruption 부여

- technique to improve corruption robustness

- But, 이러한 corruption 하에서 성능을 높이기는 어려운 상황! 이유?

  - 1) training against corruptions only encourages networks to memorize the specific corruptions seen during training

    ( leaves the model  unable to generalize to NEW corruption )

  - 2) highly sensitive to images shifted by single pixel

<br>

이 논문은, **data shift**하에서,  **(1) Robustness**와 **(2) Uncertainty estimate**를 동시에 improve하는 Augmix를 제안한다!

<br>

# 2. Related Work

## 2-1) Robustness under data shift

- training with various blur augmentations can fail to generalize to **unseen blurs**
- propose **measuring generalization to unseen corruptions**
- **robustness to data shift** greatly affects the **reliablity of real world ML system**

<br>

## 2-2) Calibration under data shift

- propose **metrics for determining the calibration**
- **ensembling classifier prediction** improves prediction calibration
- model calibration substantially **deteriorates under data shift**

<br>

## 2-3) Data Augmentation

- greatly improve **generalization performance**

- ex) flipping, cropping, cutout, cutmix, mixup..

  - **cutout** : random occlusion

  - **cutmix** : replace a portion of image with a portion of a different image

  - **mixup** : use information of 2 images

    ( element-wise convex combination of 2 image )

<br>

# 3. Augmix

**(1) model robustness와 (2) uncertainty estimate**를 모두 improve한다!

알고리즘 간단 소개

- 1) **operations are sampled** stochastically

- 2) produce a **high diversity** of augmented images

- 3) **consistent embedding**

  ( use of **Jensen-Shannon divergence** as consistency loss )

<br>

### 알고리즘

![figure2](/assets/img/reli/img15.png)

<br>

### Jensen-Shannon Divergence Consistency loss

- Notation 
  - $$p_{\text {orig }}=\hat{p}(y \mid \left.x_{\text {orig }}\right), p_{\text {augmix } 1}=\hat{p}\left(y \mid x_{\text {augmix } 1}\right), p_{\text {augmix } 2}=\hat{p}\left(y \mid x_{\text {augmix } 2}\right)$$.

- enforce SMOOTHER neural network responses
- 계산 방법
  - step 1) $$M=\left(p_{\text {orig }}+p_{\text {augmix } 1}+p_{\text {augmix } 2}\right) / 3$$
  - step 2) $$\operatorname{JS}\left(p_{\text {orig }} ; p_{\text {augmix } 1} ; p_{\text {augmix } 2}\right)=\frac{1}{3}\left(\operatorname{KL}\left[p_{\text {orig }} \mid M\right]+\operatorname{KL}\left[p_{\text {augmix } 1} \mid M\right]+\operatorname{KL}\left[p_{\text {augmix2 }} \mid M\right]\right) $$

- 최종적인 Loss Function :
  - $$\mathcal{L}\left(p_{\text {orig }}, y\right)+\lambda \operatorname{JS}\left(p_{\text {orig }} ; p_{\text {augmix } 1} ; p_{\text {augmix } 2}\right)$$.



