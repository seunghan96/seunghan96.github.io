---
title: \[continual\] (paper 4) Continual Learning with Deep Generative Replay
categories: [CONT,STUDY]
tags: [Continual Learning]
excerpt: Deep Generative Replay
---

# Continual Learning with Deep Generative Replay

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Contents

0. Abstract
2. Introduction
2. Related Works
   1. Comparable Methods
   2. Deep Generative Models
3. Generative Replay
   1. Proposed Method

<br>

# 0. Abstract

Catastrophic Forgetting 해결 위해, replay all previous data?

$$\rightarrow$$ 효과는 있겠지만... **LARGE memory** 필요

<br>

이 논문은 인간의 뇌 부분인 **hippocampus**에 영감을 받아, **Deep Generative Replay**를 제안함

DGR의 두 main model

- 1) generator ( = deep generative model )
- 2) solver ( = task solving model )

<br>

# 1. Introduction

**Catastrophic forgetting** = training new objective causes forgetting of former knowledge!

이를 해결하기 위한 시도로, 제안되었던 "memory-based approach"

- 과거의 data를 저장하는 **episodic memory system**에 의존
- 한계점 ) require large memory!

<br>

이에 대한 대안으로, ***Deep Generative Replay를 제안***

- **과거 데이터를 저장하지 않는** DNN 알고리즘!

- 그렇다면 과거 데이터를 어떻게 활용?

  $$\rightarrow$$ concurrent replay of **generated pseudo-data**

- past data를 mimic하기 위해, **GAN framework** 사용

<br>

# 2. Related Works

## 2-1. Comparable Methods

### (1) Optimization

- regularization
- ex) dropout, L2, EWC

<br>

### (2) Sequentially train

- ( multiple task를 수행할 수 있는 ) sequentially train DNN 

- ex) augment networks with task-specific parameters

  ( input 부근의 parameter는 common param, output 부근은 task-specific )

- lower learning rates on some parameter 또한 forgetting 방지하는 것으로 알려짐

<br>

## 2-2. Deep Generative Models

GAN framework 사용

- $$\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$$.

<br>

# 3. Generative Replay

Notation :

- task sequence : $$\mathbf{T}=\left(T_{1}, T_{2}, \cdots, T_{N}\right)$$.

- task의 data : $$D_{i} $$..... $$\left(\boldsymbol{x}_{i}, \boldsymbol{y}_{i}\right)$$ sample을 뽑음

- scholar : $$H=\langle G, S\rangle$$

  - 여기서 $$G$$ (generator)는, GAN의 generator & discriminator를 모두 포함한 개념

  - $$S$$ (solver) : classifier

    $$\rightarrow$$ solver는 $$\mathbf{T}$$의 모든 task에 대해서 수행한다 ( loss function = $$\mathbb{E}_{(\boldsymbol{x}, \boldsymbol{y}) \sim D}[L(S(\boldsymbol{x} ; \theta), \boldsymbol{y})]$$ )

<br>

## 3-1. Proposed Method

Sequential Training ( task들이 순차적으로 유입됨 )

***과거의 DB를 사용하는 것이 아니라, 과거의 DB를 생성해낼 법한 generator를 학습시킴***

- generator는 cumulative하게 **모든 task들의 data**를 잘 생성해내는 방향으로 학습됨

![figure2](/assets/img/CONT/img4.png)

- Step 1) [Generator]  $$x$$를 사용하여, **replayed input** $$x^{'}$$를 생성하는 모델 학습

- Step 2) [Solver] 아래의 두 종류의 데이터를 사용하여 모델 학습

  - 데이터 1) real input : $$(x,y)$$ 

  - 데이터 2) generated inputs : $$(x^{'},y^{'})$$

    ( 여기서 $$y^{'}$$는 previous solver에 input을 넣었을때 나오는 output값이다 )