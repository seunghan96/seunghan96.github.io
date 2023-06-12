---
title: (paper 80) Learning Fast and Slow for Online TS Forecasting
categories: [TS, CONT]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Learning Fast and Slow for Online TS Forecasting

<br>

![figure2](/assets/img/ts/img380.png)

## Contents

0. Abstract
1. 


<br>

# 0. Abstract

Data arrives sequentially. 

Training **deep neural forecasters** on the fly is notoriously challenging

- limited ability to adapt to **non-stationary environments**
- **fast adaptation capability of DNN** is critica

<br>

### Fast and Slow learning Network (FSNet)

- inspired by **Complementary Learning Systems (CLS) theory**
- address the challenges of **online forecasting.** 
- improves the slowly-learned backbone 
  - by dynamically balancing fast adaptation to recent changes and retrieving similar old knowledge. 
- via an interaction between two novel complementary components: 
  - **(i) a per-layer adapter** 
    - to support fast learning from individual layers
  - **(ii) an associative memory** 
    - to support remembering, updating, and recalling repeating events

<br>

# 1.  Introduction

Batch learning setting 

- whole training dataset to be made available a priori
- implies the relationship between the input and outputs remains static

<br>

Desirable to train the deep forecaster **“online”** (Anava et al., 2013; Liu et al., 2016) 

- using only new samples to capture the changing dynamic of the environment. 

- remains challenging for 2 reasons. 

  - (1) **Naively train DNNs on data streams REQUIRES MANY SAMPLES to converge** 

    - offline training benefits ( = such as mini-batches or training for multiple epochs ) are not available. 
    - when a distribution shift happens, such models would require many samples

     $\rightarrow$  DNN lack a mechanism to facilitate successful learning on data streams. 

  - (2) **TS often exhibit RECURRENT patterns** 

    - one pattern could become inactive and re-emerge in the future. 
    - catastrophic forgetting : cannot retain prior knowledge 

$\rightarrow$ Consequently, online TS forecasting with deep models presents a promising yet challenging problem. 

<br>

This paper

- formulate **online TS forecasting** as an **online, task-free continual learning problem**

- continual learning requires balancing two objectives: 

  - (1) utilizing past knowledge to facilitate fast learning of current patterns
  - (2) maintaining and updating the already acquired knowledge

  ( = stability-plasticity dilemma )

$\rightarrow$ develop an effective **Online TS forecasting framework**

- motivated by the Complementary Learning Systems (CLS)

<br>

### FSNet (Fast-and-Slow learning Network) 

to enhance the sample efficiency of DNN when dealing with distribution shifts or recurring concepts in online TS forecasting. 

Key idea for **Fast Learning** 

- always improve the learning at current steps 

  ( = instead of explicitly detecting changes in the environment )

- employs a **perlayer adapter** to model the temporal consistency in TS & adjust each intermediate layer to learn better

For recurring patterns..

- equip each adapter with an **associative memory** 
- When encountering such events, the adapter **interacts with its memory** to retrieve and update the previous actions to further facilitate fast learning. 

<br>

### Contributions

- (1) Radically formulate learning fast in online TS forecasting as a **continual learning problem**
- (2) Propose a **fast-and-slow learning paradigm** of FSNet
  - to handle both the fast changing and long-term knowledge in TS
- (3) Extensive experiments 
  - with both real and synthetic datasets

<br>

# 2. Preliminary & Related Work

## (1) TS settings

Notation

- $\mathcal{X}=\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_T\right) \in \mathbb{R}^{T \times n}$ : a time series of $T$ observations, each has $n$ dimensions. 
- Input :  $\mathcal{X}_{i, e}=$ $\left(\boldsymbol{x}_{i-e+1}, \ldots, \boldsymbol{x}_i\right)$, 
- Predict : $f_\omega\left(\mathcal{X}_{i, H}\right)=\left(\boldsymbol{x}_{i+1}, \ldots, \boldsymbol{x}_{i+H}\right)$, 

<br>

## (2) Online Time Series Forecasting

Sequential nature of data

<br>

No separation of training and evaluation

- Instead, learning occurs over a sequence of rounds. 
  - At each round, the model receives a look-back window and predicts the forecast window. 
  - Then, the true answer is revealed to improve the model's predictions of the incoming rounds
- evaluation : accumulated errors throughout learning

<br>

Challenging sub-problems

- learning under concept drifts
- dealing with missing values because of the irregularly-sampled data

$\rightarrow$ focus on the problem of fast learning (in terms of sample efficiency) under concept drifts 

<br>

Bayesian continual learning

- to address regression problems
- However, such formulation follow the Bayesian framework, which allows for forgetting of past knowledge and does not have an explicit mechanism for fast learning
- did not focus on DNNs 

<br>

## (3) Continual Learning

Learn a series of tasks sequentially

( with only limited access to past experiences )

<br>

Continual learner : must achieve a good trade-off between …

- (1) maintaining the acquired knowledge of previous tasks
- (2) facilitating the learning of future tasks

<br>

Continual learning methods inspired from the CLS theory 

- augments the slow, deep networks with the ability to quickly learn on data streams, either via 
  - (1) Experience replay mechanism
  - (2) Explicit modeling of each of the fast and slow learning components

<br>

# 3. Proposed Framework

Formulates the “online TS forecasting” as a “task-free online continual learning problem”

Proposed FSNet framework. 

<br>

3.1 ONLINE TIME SERIES FORECASTING AS A CONTINUAL LEARNING PROBLEM Our formulation is motivated by the locally stationary stochastic processes observation, where a time series can be split into a sequence of stationary segments (Vogt, 2012; Dahlhaus, 2012; Das & Nason, 2016). Since the same underlying process generates samples from a stationary segment, we refer to forecasting each stationary segment as a learning task for continual learning. We note that this formulation is general and encompasses existing learning paradigms. For example, splitting into only one segment indicates no concept drifts, and learning reduces to online learning in stationary environments (Hazan, 2019). Online continual learning (Aljundi et al., 2019a) corresponds to the case of there are at least two segments. Moreover, we also do not assume that the points of task switch are given to the model, which is a common setting in many continual learning studies (Kirkpatrick et al., 2017; Lopez-Paz & Ranzato, 2017). Manually obtaining such information in real-world data can be expensive because of the missing or irregularly sampled data (Li & Mar
