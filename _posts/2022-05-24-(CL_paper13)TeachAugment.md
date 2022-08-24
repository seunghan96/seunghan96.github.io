---
title: (paper 13) TeachAugment
categories: [CL, CV]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# TeachAugment : Data Augmentation Optimization using Teacher Knowledge

<br>

## Contents

0. Abstract
1. Introduction
2. Related Work
3. Data Augmentation Optimization using teacher knowledge
   1. Preliminaries
   2. TeachAugment
   3. Improvement Techniques
4. Data Augmentation using NN

<br>

# 0. Abstract

***Adversarial*** Data Augmentation strategies :

- search augmentation, **maximizing task loss**
- show improvement in **model generalization**

$$\rightarrow$$ but require ***careful parameter tunining***

<br>

### TeachAugment

- propose a DA optimization method based on ***adversarial strategy***

- ***without requiring careful tuning***, by leveraging a ***teacher model***

<br>

# 1. Introduction

**AutoAugment** : requires thousands of GPU

<br>

Online DA optimization 

- alternately update **(1) augmentation policies** & **(2) target network**
- Advantages
  - a) reduce computational costs
  - b) simplify the DA pipeline

- mostly based on ***adversarial strategy***

  - searches augmentation by **maximizing task loss** for target model

    ( = improve model generalization )

  - problem : ***unstable***

    - maximizing loss = can be achieved by **collapsing the inherent images**

    - to avoid collapse ….. **regularize augmentation based on prior knowledge**

      $$\rightarrow$$ need lots of tuned parameters!

<br>

To alleviate **tuning problem** …. propose ***TeachAugment***

- **online** DA optimization using **teacher knowledge**
- based on **adversarial DA strategy**
- search augmentation where ***transformed image is RECOGNIZABLE for a TEACHER MODEL***
- do not require priors / hyperparameters

<br>

Propose DA using NN that represent 2 functions

- (1) ***geometric augmentation***
- (2) ***color augmentation***

- why NN?
  - a) update using GD
  - B) reduce \# of functions in the search space to “2” 

<br>

### Contributions

1. **online DA ( w.o careful parameter tuning )**
2. **DA using NN**

<br>

![figure2](/assets/img/cl/img34.png)

<br>

# 2. Related Work

Conventional DA :

- geometric & color transformation are widely used!

- using DA, improvements are made on…

  - (1) image recognition accuracy
  - (2) un/semi-supervised representation learning

- usually improves **model generalization**,

  but sometimes **hurts performance, or induce unexpected biases**

  $$\rightarrow$$ need to find ***effective augmentation policies***

<br>

ex) AutoAugment

$$\rightarrow$$ ***automatically search for effective data augmentation***

<br>

Data Augmentation search 

- category 1) ***proxy task based***
- category 2) ***proxy task free***

<br>

### Proxy Task based

- search DA strategies on proxy tasks, that uses **subsets of data** and/or **small models** to reduce computational costs
- thus, might be ***SUB-OPTIMAL***

<br>

### Proxy Task free

- ***DIRECTLY*** search DA strategies on the **target network** with **all data**

- thus, potentially **OPTIMAL**

- Ex) RandAugment, Trivial Augment

  - randomize the parameters search & reduce the size of search space

- Ex) Adversarial AutoAugment, PointAugment

  - update augmentation policies in an **online** manner

    ( = alternately update **target network** & **augmentation policies** )

<br>

This paper focus on **PROXY TASK FREE** methods , updating policies in an **ONLINE** manner

- reason 1) can ***directly search*** DA strategies on target network with ***all data***
- reason 2) ***unify the search & training process***

<br>

# 3. Data Augmentation Optimization using teacher knowledge

## (1) Preliminaries

Notation

- dataset : $$x \sim \mathcal{X}$$

- $$a_{\phi}$$ : augmentation function, parameterized by $$\phi$$
- $$f_\theta$$ : target network
  - fed into target network : $$f_{\theta}\left(a_{\phi}(x)\right)$$

<br>

Training procedure : $$\min _{\theta} \mathbb{E}_{x \sim \mathcal{X}} L\left(f_{\theta}\left(a_{\phi}(x)\right)\right.$$.

( Adversarial DA : searches $$\phi$$, maximizing the loss )

$$\rightarrow$$ $$\max _{\phi} \min _{\theta} \mathbb{E}_{x \sim \mathcal{X}} L\left(f_{\theta}\left(a_{\phi}(x)\right)\right.$$

- alternately updating $$\phi$$ & $$\theta$$

<br>

PROBLEM?

***maximizing the loss, w.r.t $$\phi$$ can be just obtained by collapsing the inherent meanings of $$x$$***

$$\rightarrow$$ solution : utilize ***teacher model*** to avoid the collapse !

<br>

## (2) TeachAugment

Notation :

- $$f_{\hat{\theta}}$$ : teacher model
- $$f_{\theta}$$ : target model

<br>

Suggest 2 types of teacher model

- (1) **pre-trained** teacher

- (2) **EMA** teacher

  ( = weights are updated as an **exponential moving average** of **target model’s weights** )

<br>

Proposed Objective :

- $$\max _{\phi} \min _{\theta} \mathbb{E}_{x \sim \mathcal{X}}\left[L\left(f_{\theta}\left(a_{\phi}(x)\right)\right)-L\left(f_{\hat{\theta}}\left(a_{\phi}(x)\right)\right)\right]$$.

  - maximize for TARGET model
  - minimize for TEACHER model

- avoids **collapsing** the inherent meanings of images

  ( $$\because$$ if not, loss for TEACHER model will explode!! )

<br>

![figure2](/assets/img/cl/img35.png)

- objective is solved by ***ALTERNATIVELY** updating the **augmentation function** & **target model**
- process
  - step 1) update TARGET network for $$n_{inner}$$ steps
  - step 2) update AUGMENTATION function

<br>

![figure2](/assets/img/cl/img36.png)

<br>

## (3) Improvement Techniques

training procedure : similar to GANs & actor-critic in RL

( lots of strategies to mitigate instabilities & improve training )

<br>

# 4. Data Augmentation using NN

Two NNs

- (1) color augmentation model : $$c_{\phi_{c}}$$
- (2) geometric augmentation model : $$g_{\phi_{g}}$$
- (1) + (2) = $$a_{\phi}=g_{\phi_{g}} \circ c_{\phi_{c}}$$
  - parameters : $$\phi=\left\{\phi_{c}, \phi_{g}\right\}$$

<br>

![figure2](/assets/img/cl/img37.png)

- input image : $$x \in \mathbb{R}^{M \times 3}$$

  ( $$M$$ : number of pixels )

- data augmentation probability

  - color : $$p_{c} \in(0,1)$$
  - gemoetric : $$p_{g} \in(0,1)$$

<br>

Data Augmentation

- [ color ] $$\tilde{x}_{i}=t\left(\alpha_{i} \odot x_{i}+\beta_{i}\right),\left(\alpha_{i}, \beta_{i}\right)=c_{\phi_{c}}\left(x_{i}, z, c\right)$$
- [ geometric ] $$\hat{x}=\operatorname{Affine}(\tilde{x}, A+I), A=g_{\phi_{g}}(z, c)$$
  - affine transformation of $$\tilde{x}$$ with a parameter $$A+I$$
- ( + also learn the probabilities $$p_g $$ & $$p_c$$ )

<br>

