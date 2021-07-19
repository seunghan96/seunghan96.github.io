---
title: (paper) An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis (2019)
categories: [NLP,ABSA]
tags: [NLP, ABSA]
excerpt: An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis (2019)
---

# An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis (2019)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract & Introduction
1. Related Work
2. Proposed Method
   1. Aspect-level Tasks
   2. Document-level Tasks
   3. Message Passing Mechanism

<br>

# 0. Abstract & Introduction

ABSA : 주로 **pipeline** 방식으로 수행

- step 1) **aspect term** extraction
- step 2) **sentiment** prediction

위 방법의 문제점?

- (문제 1) 두 개의 subtask의 joint information을 충분히 활용하지 못해
- (문제 2) document-labeled sentiment corpus 등의 정보도 활용 X

<br>
이 논문에서는 위 문제들을 극복할 **Interactive Multi-task learning Network (IMN)**을 제안한다

- 특징 1) **JOINTLY** learn multiple related tasks simultaneously $$\rightarrow$$ (문제 1) 극복

  ​	      at both **TOKEN level & DOCUMENT level** $$\rightarrow$$ (문제 2) 극복

- 특징 2) Novel **message passing** mechanism $$\rightarrow$$ (문제 1) 극복

  ( tasks들 사이의 informative interaction을 가능케함! )

  - (a) 그러한 information은 **shared latent representation**에 combine됨
  - (b) not only shared features, but also **EXPLICIT INTERACTION**

- 특징 3)  **document-level classification** task 도 함께 수행

<br>

# 1. Related Work

### (a) ABSA

- 생략

<br>

### (b) Multi-task learning

전통적인 Multi-task : (1) & (2) 사용

- (1) shared feature spaces
- (2) task-specific feature spaces

$$\rightarrow$$ capture **CORRELATIONS** between tasks & improve **MODEL GENERALIZATION** ability

<br>

문제점 : ***기존의 multi-task learning은 "EXPLICLITY(명시적으로)" task들 사이의 interaction을 모델링하지는 못한다!***  $$\rightarrow$$ IMN은 가능해!

- 1) [implicitly] latent representation을 공유할 뿐만 아니라,
- 2) [explicitly] interaction between tasks를 **iterative message passing scheme**을 통해 잡아내!

<br>

### Message Passing Architecture

- message passing graphical model inference algorithms

  ( RNN 아키텍쳐 자주 사용함 )

- 이 논문 또한 이 아이디어를 사용!

  - **propagate information in network** 

  - **learn the update operators**

- 매 iteration마다 shared latent variable를 update한다

<br>

# 2. Proposed Method

Notation

- input : $$\{x_1,..,x_n\}$$

- feature extraction component : $$f_{\theta_s}$$

  ( shared across tasks )

<br>

$$f_{\theta_s}$$ ( feature extraction component )

- 구성 1) word embedding layer

- 구성 2) CNN을 통한 feature extraction

- output of $$f_{\theta_s}$$ : sequence of latent vectors, $$\left\{\mathbf{h}_{1}^{s}, \mathbf{h}_{2}^{s}, \ldots, \mathbf{h}_{n}^{s}\right\}$$

  $$\rightarrow$$ 이 output은 각기 다른 task-specific components의 input으로 들어간다

<br>

AE/AS/DS/DD task 각각 수행한다!

- 매 iteration마다, 각 task (AE혹은 AS)에서 나온 최종 정보들은 다시 passed back 되어서

  shared latent vectors에 combined 된다. ( 그림 참조 )

<br>

![figure2](/assets/img/nlp/nlp55.png)

<br>

## 2-1) Aspect-level Tasks

### AE

- extract all the **Aspect & Opinion** terms in sentence
- sequence tagging ( BIO tagging 문제 )
  - $$Y^{a e}=\{B A, I A, B P, I P, O\}$$.
- parameter & output : $$f_{\theta_{a e}}$$ &  $$\left\{\hat{\mathbf{y}}_{1}^{a e}, \ldots, \hat{\mathbf{y}}_{n}^{a e}\right\} .$$
- encoder : $$m^{a e}$$ layers of CNNs
  - map into shared representations : $$\left\{\mathbf{h}_{1}^{a e}, \mathbf{h}_{2}^{a e}, \ldots, \mathbf{h}_{n}^{a e}\right\}$$

<br>

### AS

- (classification X) sequence tagging 문제로써 푼다! ( AE랑 맞춰주기 위해서 )
  - $$Y^{a s}=\{$$ pos, neg, neu $$\}$$
- encoder : $$m^{a s}$$ layers of CNNs
  - map into shared representations : $$\left\{\mathbf{h}_{1}^{a s}, \mathbf{h}_{2}^{a s}, \ldots, \mathbf{h}_{n}^{a s}\right\}$$
- AS encoder의 경우, AE encoder과는 다르게, 추가적으로 **self-attention layer**를 CNN위에 넣는다.

<br>

## 2-2) Document-level Tasks

핵심 취지 : **exploit knowledge from DOCUMENT-level classification tasks**

- 위의 3-1)과 마찬가지로, DS & DD 수행

<br>

## 2-3) Message Passing Mechanism

서로 다른 **tasks들 사이의 interaction**을 잡아내기 위해 사용!

- 직전 iteration의 여러 tasks들의 prediction 값들을 aggregate한다
- 이 knowledge를 shared latent vectors $$\left\{\mathbf{h}_{1}^{s}, \mathbf{h}_{2}^{s}, \ldots, \mathbf{h}_{n}^{s}\right\}$$를 update하는데에 사용

$$\begin{aligned}
\mathbf{h}_{i}^{s(t)}=& f_{\theta_{r e}}\left(\mathbf{h}_{i}^{s(t-1)}: \hat{\mathbf{y}}_{i}^{a e(t-1)}: \hat{\mathbf{y}}_{i}^{a s(t-1)}:\right. \left.\hat{\mathbf{y}}^{d s(t-1)}: a_{i}^{d s(t-1)}: a_{i}^{d d(t-1)}\right)
\end{aligned}$$,