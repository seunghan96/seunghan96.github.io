---
title: \[Paper Review\] 01.(evaluation) Improved Techniques for Training GANs
categories: [GAN]
tags: [GAN]
excerpt: 2016, Feature Matching, Minibatch Discrimination,Virtual Batch Normalization (VBN), Semi-supervised Learning
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 01.Improved Techniques for Training GANs

<br>

### Contents

0. Abstract
1. Introduction
2. Related Work
3. Toward Convergent GAN Training
   1. Feature Matching
   2. Minibatch Discrimination
   3. Historical Averaging
   4. One-sided label smoothing
   5. Virtual Batch Normalization (VBN)
4. Semi-supervised Learning

<br>

# 0. Abstract

GAN의 2가지 application에 대해 다룸

- 1) semi-supervised learning
- 2) generation of images that humans find visually realistic

Achieve SOTA in semi-supervised classifictation

<br>

# 1. Introduction

training GANs :

"requires finding a **Nash Equilibrium** of a NON-convex game, with CONTINUOUS, HIGH-dimensional parameters" $$\rightarrow$$ fail to converge

<br>

introduce **several techniques to ENCOURAGE CONVERGENCE**  of GANs game

lead to..

- 1) improved **semi-supervised learning performance**
- 2) improved **sample generation**

<br>

# 2. Related Work

- several recent papers focus on **improving the STABILITIY of TRAINING**
- this paper use some of **DCGAN** architectures

<br>

Propose 2 techniques

- 1) **feature matching**
  - use **maximum mean discrepency** to train Generator
- 2) **minibatch features**
  - based on batch normalization
  - propose **VIRTUAL** batch normalization (VBN)

<br>

# 3. Toward Convergent GAN Training

Problems

- 1) costs functions are non-convex
- 2) parameters are continuous
- 3) parameter space is extremely high-dimensional

<br>

## 3-1) Feature Matching

- specify a new objective for GENERATOR that prevents it from overtraining DISCRIMINATOR

- (X) directly maximizing output of discriminator

  (O) requires the GENERATOR to generate data that "matches the statistics of real data"

  $$\rightarrow$$ use discriminator only to specify the statistics that is worth matching

<br>

$$\mathbf{f}(\boldsymbol{x})$$ : activations on an intermediate layer of the **discriminator**,

$$ \mid \mid \mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}} \mathbf{f}(\boldsymbol{x})-\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})} \mathbf{f}(G(\boldsymbol{z})) \mid \mid _{2}^{2}$$ : new objective

<br>

## 3-2) Minibatch Discrimination

problem : **collapse** to same point!

- ( *all outputs race toward a single point that D currently believes is highly realistic* )

solution : Minibatch Discrimination

- allow D to look at **multiple data examples** in combination

<br>

Modeling the closeness between examples in minibatches!

Notation

- $$\mathbf{f}\left(\boldsymbol{x}_{i}\right)$$ : embedded images
- $$T \in \mathbb{R}^{A \times B \times C}$$ : tensor to multiply
- $$M_{i} \in \mathbb{R}^{B \times C}$$ : $$\mathbf{f}\left(\boldsymbol{x}_{i}\right)$$ $$T$$
- $$c_{b}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(- \mid \mid M_{i, b}-M_{j, b} \mid \mid _{L_{1}}\right) \in \mathbb{R}$$ : negative exponential

![figure2](/assets/img/gan/img1.png)

<br>

## 3-3) Historical Averaging

include term $$ \mid \mid \boldsymbol{\theta}-\frac{1}{t} \sum_{i=1}^{t} \boldsymbol{\theta}[i] \mid \mid ^{2}$$ to each players' cost

- $$\theta[i]$$ : value of the parameters at past time $$i$$

help find equilibria of low-dimensional, continuous non-convex games

<br>

## 3-4) One-sided label smoothing

Label smoothing

- 0 $$\rightarrow$$ 0.1
- 1 $$\rightarrow$$ 0.9

to reduce the vulnerability of NN to **adversarial examples**

<br>

Replacing positive classification targets with $$\alpha$$ and negative targets with $$\beta$$!

Optimal discriminator : $$D(\boldsymbol{x})=\frac{\alpha p_{\text {data }}(\boldsymbol{x})+\beta p_{\text {model }}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{\text {model }}(\boldsymbol{x})} $$

- smooth only the POISTIVE labels to $$\alpha$$ ( negative labels are still 0 )

<br>

## 3-5) Virtual Batch Normalization (VBN)

problem of BN : **highly dependent on several other inputs $$x^{'}$$ in same minibatch

$$\rightarrow$$ normalized based on **statistics collected on REFERENCE batch of examples**, 
which are chosen once & fixed at the start of training

<br>

# 4. Semi-supervised Learning

standard classifier

- data point $$x$$ $$\rightarrow$$ $$K$$ possible classes
- softmax : $$p_{\text {model }}(y=j \mid x)=\frac{\exp \left(l_{j}\right)}{\sum_{k=1}^{K} \exp \left(l_{k}\right)} $$
- minimize CE loss

<br>
can also do **SEMI-supervised learning**, with standard classifier, by..

***simply adding samples*** from GAN generator $$G$$ to our dataset

- new class "generated" : $$y=K+1$$

<br>

use $$p_{\text {model }}(y=K+1 \mid \boldsymbol{x})$$ to supply the probability that $$\boldsymbol{x}$$ is fake

( = corresponding to $$1-D(\boldsymbol{x})$$ in the original GAN )

<br>

Loss function for training the classifier 
$$
\begin{aligned}
L &=-\mathbb{E}_{\boldsymbol{x}, y \sim p_{\text {data }}(\boldsymbol{x}, y)}\left[\log p_{\text {model }}(y \mid \boldsymbol{x})\right]-\mathbb{E}_{\boldsymbol{x} \sim G}\left[\log p_{\text {model }}(y=K+1 \mid \boldsymbol{x})\right] \\
&=L_{\text {supervised }}+L_{\text {unsupervised }}, \text { where } \\
L_{\text {supervised }} &=-\mathbb{E}_{\boldsymbol{x}, y \sim p_{\text {data }}(\boldsymbol{x}, y)} \log p_{\text {model }}(y \mid \boldsymbol{x}, y<K+1) \\
L_{\text {unsupervised }} &=-\left\{\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})} \log \left[1-p_{\text {model }}(y=K+1 \mid \boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim G} \log \left[p_{\text {model }}(y=K+1 \mid x)\right]\right\}
\end{aligned}
$$
