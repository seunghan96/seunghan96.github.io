---
title: \[reliable\] (paper 3) Enhancing the Reliability of Out-of-distribution Image detection in Neural Networks
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: ODIN, Out-of-distribution detection
---

# Enhancing the Reliability of Out-of-distribution Image detection in Neural Networks

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract

1. Introduction
2. Problem Statement
3. ODIN : Out-of-distribution Detector
   1. Temperature scaling
   2. Input Preprocessing
   3. Out-of-distribution Detector

<br>

# 0. Abstract

out-of-distribution 이미지를 detect하는 **ODIN**알고리즘을 제안함

**ODIN = out-of-distribution detector**

- pre-trained NN의 change를 요구하지 않음

- 2가지 key method

  - 1) temperature scaling

  - 2) adding small perturbation to input

    $$\rightarrow$$ IN & OUT of distribution image의 softmax score를 구분할 수 있음!

<br>

# 1. Introduction

NN의 문제점 : ***high confidence predictions***

- 과신해서는 안되는 o.o.d 데이터에 대해 너무 과신하는 경향이 있음

- 해결책 : in & out of distribution을 모두 input으로 넣어서 학습시키면 되지 않나?

  $$\rightarrow$$ 현실적으로 NO.... o.o.d가 너무 많을 수 있다!

<br>

[ Related works ]

(1) Hendrycks & Gimpel (2017) : baseline method 제안

- well-traned NN =  **softmax score를 o.o.d에는 낮게, i.o.d에는 높게**

(2) 이 Paper

- 1) temperature scaling

- 2) adding small perturbation to input

  $$\rightarrow$$ IN & OUT of distribution image의 softmax score gap이 더 커진다!

<br>

# 2. Problem Statement

Notation

- $$P_{X}$$ : in-distribution

- $$Q_{X}$$ : out-distribution

<br>

Test 단계 : mixture distribution $$\mathbb{P}_{\boldsymbol{X} \times Z}$$ 에서 image를 샘플함 ( $$\mathcal{X} \times\{0,1\}$$ )

- $$\mathbb{P}_{\boldsymbol{X} \mid Z=0}=P_{\boldsymbol{X}}$$.
- $$\mathbb{P}_{\boldsymbol{X} \mid Z=1}=Q_{\boldsymbol{X}}$$.

<br>

Question :

- $$\mathbb{P}_{\boldsymbol{X} \times Z}$$ 에서 뽑은 image $$X$$가 주어졌을 때, 이것이 in-distribution인가? out-distribution인가?

<br>

# 3. ODIN: Out-of-distribution Detector

2가지 key point

- 1) Temperature scaling

- 2) Input Preprocessing

<br>

## 3-1) Temperature scaling

Formula : $$S_{i}(x ; T)=\frac{\exp \left(f_{i}(x) / T\right)}{\sum_{j=1}^{N} \exp \left(f_{j}(x) / T\right)}$$

- $$T \in \mathbb{R}^{+}$$ : temperature scaling parameter 

  ( training 중에는 $$T=1$$로 설정한다 )

- **Softmax score** : $$S_{\hat{y}}(x ; T)=\max _{i} S_{i}(x ; T)$$

  ( = maximum softmax probability )

- $$T$$를 잘 설정하면, in & out distribution의 softmax score gap을 키울 수 있다!

<br>

## 3-2) Input Preprocessing

- add small **perturbation**

- $$\tilde{\boldsymbol{x}}=\boldsymbol{x}-\varepsilon \operatorname{sign}\left(-\nabla_{\boldsymbol{x}} \log S_{\hat{y}}(\boldsymbol{x} ; T)\right)$$.

- **decrease** the **softmax score** for the true label 

  & **force** the neural network to make a **wrong prediction**

<br>

## 3-3) Out-of-distribution Detector

- image $$x$$의 softmax score가 일정 threshold 넘으면 **in-distribution**이라고 판단
- $$g(x ; \delta, T, \varepsilon)=\left\{\begin{array}{ll}
  1 & \operatorname{ifmax}_{i} p(\tilde{x} ; T) \leq \delta \\
  0 & \operatorname{ifmax}_{i} p(\tilde{x} ; T)>\delta
  \end{array}\right.$$.

