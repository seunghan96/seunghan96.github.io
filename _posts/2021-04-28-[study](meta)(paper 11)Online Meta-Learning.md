---
title: \[meta\] (paper 11) Online Meta-Learning
categories: [META,STUDY]
tags: [Meta Learning]
excerpt: Online Learning, Meta Learning
---

# Online Meta-Learning (2019)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Foundations
   1. Few-shot Learning
   2. Meta-Learning & MAML
   3. Online Learning
3. The Online Meta-Learning Problem
4. FTML (Follow the Meta Leader)

<br>

# 0. Abstract

Online Meta Learning = (1) + (2)

- (1) Meta Learning : learning a prior over parameters / for fast adaptation
  - 문제점 ) data가 "batch" 단위로써 들어온다고 가정함
- (2) Online Learning : sequential settings 
  - 문제점 ) task-specific adaptation없이 하나의 single model을 주로 사용

<br>

Online Meta Learning을 수행하는 **"FTML" (Follow The Meta Leader) 알고리즘을 제안함**

<br>

# 1. Introduction

### Meta Learning

- learning to learnin
- past experience = prior over model params
- (문제점) **neglects the "sequential & non-stationary aspects"**

<br>

### Online Learning

- sequential settings ( tasks are revealed one after another )
- (문제점) **how past experience can accelerate adaptation to new task를 고려 못함**

$$\rightarrow$$ NEITHER is ideal for continual lifelong learning!

<br>

# 2. Foundations

우선, (1) meta-learning의 대표모델 MAML & (2) Online-learning에 대해서 review

## 2-1. Few-shot Learning

관심 대상 : family of "TASKS"

Notation :

- task $$\mathcal{T}_i$$에 해당하는 data : $$\mathcal{D}_{i}:=\left\{\mathrm{x}_{i}, \mathbf{y}_{i}\right\}$$
- task의 개수 : $$M$$

- predictive model : $$h(\cdot ; \mathrm{w})$$
- population risk of the model : $$f_{i}(\mathbf{w}):=\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{T}_{i}}[\ell(\mathbf{x}, \mathbf{y}, \mathbf{w})]$$
- loss function 
  - example) squared error loss : $$\ell(\mathbf{x}, \mathbf{y}, \mathbf{w})=\|\mathbf{y}-\boldsymbol{h}(\mathbf{x} ; \mathbf{w})\|^{2}$$

- average loss on $$\mathcal{D}_i$$ :  $$\mathcal{L}\left(\mathcal{D}_{i}, \mathbf{w}\right)$$

<br>

$$\rightarrow$$ 오직 $$\mathcal{D}_i$$에만 의존해서는, $$f_i(\mathbf{w})$$를 minimize하기는 어렵다 ( TOO SMALL DATASET )

- 우리는 그러한 $$\mathcal{D}_i$$들이 여러개 있고, 이들이 순차적(sequentially) 들어온다

<br>

## 2-2. Meta-Learning & MAML

task는 fixed distn에서 뽑힌다 : $$\mathcal{T} \sim \mathbb{P}(\mathcal{T})$$.

- [ Meta Training Time ] 

  - task $$M$$개를 뽑는다 : $$\left\{\mathcal{T}_{i}\right\}_{i=1}^{M}$$ ( + Dataset도 함께 )

- [ Deployment Time ]

  - 새로운 task 하나를 뽑는다 : $$\mathcal{T}_{j} \sim \mathbb{P}(\mathcal{T})$$

    ( 특징 : small labeled dataset, $$\mathcal{D}_{j}:=\left\{\mathbf{x}_{j}, \mathbf{y}_{j}\right\}$$ )

<br>

Meta-learning algorithm의 목적 :

- $$M$$ 개의 training task를 통해서 모델을 만들어서,

- 새로운 task의 데이터 $$\mathcal{D}_{j}:=\left\{\mathbf{x}_{j}, \mathbf{y}_{j}\right\}$$가 주어졌을 때, 

  이에 대해 $$f_{j}(\mathbf{w})$$를 minimize하도록 빠르게 update되게끔!

<br>

### MAML (Model Agnostic Meta Learning)

- learning an "initial set of parameters" $$\mathrm{w}_{\mathrm{MAML}}$$ 
- [idea] meta-test time에, $$D_j$$를 사용해서 $$\mathrm{w}_{\mathrm{MAML}}$$에서 few step의 GD 만으로도 $$f_j(\cdot)$$가 minimize되도록

- Optimization Problem

  - $$\mathrm{w}_{\mathrm{MAML}}:=\arg \min _{\mathbf{w}} \frac{1}{M} \sum_{i=1}^{M} f_{i}\left(\mathbf{w}-\alpha \nabla \hat{f}_{i}(\mathbf{w})\right)$$.
  - 위 식의 inner gradient $$\nabla \hat{f}_{i}(\mathbf{w})$$ 는 small mini-batch of $$D_i$$ 에 의해 계산

- 위 식의 solution :

  $$\mathbf{w}_{j} \leftarrow \mathbf{w}_{\mathrm{MAML}}-\alpha \nabla \hat{f}_{j}\left(\mathbf{w}_{\mathrm{MAML}}\right)$$.

- 위 식에 대한 해석

  - step 1) learning a prior over model params
  - step 2) fine-tuning as inference

<br>

### MAML (및 기타 meta-learning)의 문제점

2가지 이유로, Sequential Setting에 부적합하다

- 이유 1) 2개의 distinct phase를 가진다
  - meta-training & meta-testing(=deployment)
  - continuous learning 방식이 아님
- 이유 2) assume that task com from FIXED distn
  - 현실에는 non-stationary task distn이 많음

<br>

## 2-3. Online Learning

Setting :

- sequence of loss functions $$\left\{f_{t}\right\}_{t=1}^{\infty}$$, one in each round $$t$$. 

  ( 이러한 function들은 need not be drawn from a fixed distribution )

- Goal : **sequentially** decide on model parameters $$\left\{\mathbf{w}_{t}\right\}_{t=1}^{\infty}$$ that perform well on the loss sequence

<br>

Standard Objective :

- minimize some notion of 'regret'

- ex) "compare to the cumulative loss of the best fixed model in hindsight"

  $$\operatorname{Regret}_{T}=\sum_{t=1}^{T} f_{t}\left(\mathbf{w}_{t}\right)-\min _{\mathbf{w}} \sum_{t=1}^{T} f_{t}(\mathbf{w})$$.

<br>

**목표 : 위의 regret이 $$T$$가 커짐에 따라 최대한 "느리게" 커지도록!**

<br>

### 대표적 알고리즘 : FTL (Follow The Leader)

$$\mathbf{w}_{t+1}=\arg \min _{\mathbf{w}} \sum_{k=1}^{t} f_{k}(\mathbf{w})$$.

<br>

# 3. The Online Meta-Learning Problem

Introduction

- Sequential Setting을 가정한다

- (용어) "round" : task의 "번째"

- Learner의 목표 : 해당 round에서 perform 잘하는 model param $$\mathbf{w}_t$$ 찾기

  ( monitored by $$f_{t}: \mathrm{w} \in \mathcal{W} \rightarrow \mathbb{R}$$ .... 이걸 minimize하도록 )

- 매 round에서 deployed/evaluated 되기 이전에, **'task-specific update'가 이루어진다**

<br>

Task-specific Update

- 매 round마다 다음의 mapping : $$U_{t}: \mathrm{w} \in \mathcal{W} \rightarrow \tilde{\mathrm{w}} \in \mathcal{W}$$

- example) Gradient Descent

  $$\boldsymbol{U}_{t}(\mathbf{w})=\mathbf{w}-\alpha \nabla \hat{f}_{t}(\mathbf{w})$$.

<br>

전체적인 Process

![figure2](/assets/img/META/img32.png)

<br>

GOAL : **Minimize regret over the rounds**

- $$\text { Regret }_{T}=\sum_{t=1}^{T} f_{t}\left(\boldsymbol{U}_{t}\left(\mathbf{w}_{t}\right)\right)-\min _{\mathbf{w}} \sum_{t=1}^{T} f_{t}\left(\boldsymbol{U}_{t}(\mathbf{w})\right)$$.

<br>

# 4. FTML (Follow the Meta Leader)

수식 한줄로서 끝!

- $$\mathbf{w}_{t+1}=\arg \min _{\mathbf{w}} \sum_{k=1}^{t} f_{k}\left(\boldsymbol{U}_{k}(\mathbf{w})\right)$$.

해석 : play the best meta-learner in hindsight, "if the learning process were to stop at round $$t$$"

