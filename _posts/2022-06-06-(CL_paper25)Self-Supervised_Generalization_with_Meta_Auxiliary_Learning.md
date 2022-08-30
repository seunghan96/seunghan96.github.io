---
title: (paper 25) Self-Supervised Generalization with Meta Auxiliary Learning
categories: [CL, CV]
tags: []
excerpt: 2019
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Self-Supervised Generalization with Meta Auxiliary Learning

<br>

## Contents

0. Abstract
0. 

<br>

# 0. Abstract

learning with **auxiliary task** $\rightarrow$ Improve **generalization ability of primary task**

- but … cost of manually labeling auxiliary data

<br>

### MAXL ( Meta AuXiliary Learning )

- **automatically** learns appropriate **labels** for **auxiliary task**
- train 2 NNs
  - (1) **label-generarion network** : to predict auxiliary labels
  - (2) **multi-task network** : to trian the primary task with auxiliary task

<br>

# 1. Introduction

Auxiliary Learning (AL) vs. Multi-task Learning (ML)

- AL : focus only on **primary task**
- MTL : focus on both **primary task** & **auxiliary task**

<br>

### MAXL

- simple & general meta-learning algorithm

- defining a task = defining a label

  ( = optimal auxiliary task = one which has optimal labels )

  $\rightarrow$ goal : **automatically discover these auxiliary labels**, using **labels for primary task**

<br>

### 2 NNs

1. Multi-task network 
   - Trains **primary task** & **auxiliary task**
2. Label-generation network
   - learns the **labels for auxiliary task**

<br>

![figure2](/assets/img/cl/img53.png)

<br>

### Key idea of MAXL

- use the performance of the primary task, to improve the **auxiliary labels** for the next iteration
- achieved by ***defining a loss for the label-generation network as a function of multi-task network’s performance on primary task training data***

<br>

# 2. Related Work

## (1) Multi-task & Transfer Learning

MTL : shared representation & set of related learning tasks

TL : to improve generatliaztion / incorporate knowledge from other domains

<br>

## (2) Auxiliary Learning

Goal : focus only on **single primary task**

Can also perform auxiliary learning **without GT labels** ( = in **unsupervised manner** )

<br>

## (3) Meta Learning

aims to **induce the learning algorithm itself**

<br>

MAXL : designd to **learn to generate useful auxiliary labels**, which themselves are **used in another learning procedure**

<br>

# 3. Meta Auxilary Learning

task : **classification task** ( both for primary & auxiliary task )

- auxiliary task : **sub-class labelling problem**
- ex) primary - auxiliary : Dog - Labrador

<br>

## (1) Problem Setup

Notation

- $f_{\theta_1}(x)$ : multi-task network
  - updated by loss of **primary & auxiliary** tasks
- $g_{\theta_2}(x)$ : label-generation network
  - updated by loss of **primary task**

<br>

Multi-task Network

- apply **hard parameter sharing approach**

  ( common & task-specific parameters )

- notation

  - primary task prediction : $f_{\theta_1}^{\text {pri }}(x)$

    ( ground truth : $y^{\text {pri }}$ )

  - auxiliary task prediction : $f_{\theta_1}^{\text {aux }}(x)$

    ( ground truth : $y^{\text {aux }}$ )

<br>





We found during experiments that training benefited from assigning each primary class its own unique set of possible auxiliary classes, rather than sharing all auxiliary classes across all primary classes. In the label-generation network, we therefore define a hierarchical structure $\psi$ which determines the number of auxiliary classes for each primary class. At the output layer of the label-generation network, we then apply a masked SoftMax function to ensure that each output node represents an auxiliary class corresponding to only one primary class, as described further in Section 3.3. Given input data $x$, the label-generation network then takes in the hierarchy $\psi$ together with the groundtruth primary task label $y^{\text {pri }}$, and applies Mask SoftMax to predict the auxiliary labels, denoted by $y^{\mathrm{aux}}=g_{\theta_2}^{\mathrm{gen}}\left(x, y^{\mathrm{pri}}, \psi\right)$. A visualisation of the overall MAXL framework is shown in Figure 2. Note
