---
title: \[Paper Review\] 19.(Analysis,Manipulation) GAN Dissection ; Visualizing and Understanding GANs
categories: [GAN]
tags: [GAN]
excerpt: 2018, GAN Dissection
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 19. GAN Dissection : Visualizing and Understanding GANs

<br>

### Contents

0. Abstract
1. Introduction
2. Method
   1. Characterizing units by Dissection
   2. Measuring Causal Relationships using Intervention

<br>

# 0. Abstract

present an analytic framework to visualize & understand GANs at

- unit / object / scene level

<br>

Step 1 ) identify a group of **interpretable units**

- that are **closely related to object concepts**, 
- using a **segmentation-based network dissection method**

Step 2 ) **quantify the causal effect** of interpretable units

- by measuring the **ability of interventions to control** objects in the output

<br>

# 1. Introduction

General method for visualizing & understanding GANs 

- **at different level of abstractions**
- from each **neuron / object / contextual relationship**

<br>

# 2. Method

- analyze how objects such as trees are **encoded** by **internal representations of GAN generator**

- notation

  - GAN : $$G : \mathbf{z} \rightarrow \mathbf{x}$$ .... where $$\mathrm{x} \in \mathbb{R}^{H \times W \times 3}$$
  - $$\mathrm{x}=f(\mathbf{r})=f(h(\mathbf{z}))=G(\mathbf{z})$$.

- $$\mathbf{r}$$ has necessary data to produce images!

  Question

  - Whether information about concept $$c$$ is in $$\mathbf{r}$$? (X)
  - How such information is encoded in $$\mathbf{r}$$ ! (O)

- $$\mathbf{r}_{\mathbb{U}, \mathrm{P}}=\left(\mathbf{r}_{\mathrm{U}, \mathrm{P}}, \mathbf{r}_{\overline{\mathrm{U}}, \mathrm{P}}\right)$$.

  - generation of object $$c$$ at location $$P$$ **depends mainly** on $$\mathbf{r}_{\mathrm{U}, \mathrm{P}}$$
  - insensitive to other units $$\mathbf{r}_{\bar{U}, \mathrm{P}}$$

<br>

Structure of $$\mathbf{r}$$ in 2 phases

1. **Dissection**

   measuring agreement between **individual units of $$\mathbf{r}$$** & every class $$c$$

2. **Intervention**

   for the represented classes (identified through 1),

   identify causal sets of units & measure causal effects

   ( by forcing sets of units on/off )

<br>

## (1) Characterizing units by Dissection

$$\mathbf{r}_{u, \mathbb{P}}$$ : one-channel $$h \times w$$ feature map of unit $$u$$

$$\rightarrow$$ Q) does $$\mathbf{r}_{u, \mathbb{P}}$$ encodes semantic class (ex.tree)?

<br>

(1) Select a universe of concepts $$c \in \mathbb{C}$$, for which we have semantic segmentation $$s_c(x)$$ for each class

(2) Then, quantify **spatial agreement** between

- 1) unit $$u$$'s thresholded feature map
- 2) concept $$c$$'s segmentation

with IOU measure

<br>

![figure2](/assets/img/gan/img46.png)

<br>

## (2) Measuring Causal Relationships using Intervention

test whether a set of units $$U$$ in $$\mathbf{r}$$ **cause the generation** of $$c$$ !

- via **turning on/off the units of $$U$$**

<br>

Decompose feature map $$\mathbf{r}$$ into 2 parts : $$\left(\mathbf{r}_{\mathrm{U}, \mathrm{P}}, \mathrm{r}_{\overline{U, \mathrm{P}}}\right)$$

( where $$\mathrm{r}_{\overline{U, P}}$$ : unforced components of $$\mathbf{r}$$ )

<br>

Original Image :

- $$\mathbf{x}=G(\mathbf{z}) \equiv f(\mathbf{r}) \equiv f\left(\mathbf{r}_{\mathrm{U}, \mathrm{P}}, \mathbf{r}_{\overline{\mathrm{U}, \mathrm{P}}}\right)$$.

Image with $$U$$ ablated at pixels $$P$$ :

- $$\mathbf{x}_{a}=f\left(\mathbf{0}, \mathbf{r}_{\overline{\mathrm{U}, \mathrm{P}}}\right)$$

Image with $$U$$ inserted at pixels $$P$$ :

- $$\mathbf{x}_{i}=f\left(\mathbf{k}, \mathbf{r}_{\overline{U, \mathrm{P}}}\right)$$.

<br>

Object is caused by $$U$$ if  ...

- the object appears in $$x_i$$ 
- the object disappears from $$x_a$$

![figure2](/assets/img/gan/img47.png)

<br>

This causality can be quantified by..

- comparing presence of trees in $$x_i$$  & $$x_a$$ 
- average effects over all locations & images

<br>

ACE

- average causal effect (ACE) of units $$\mathrm{U}$$ on the generation of on class $$c$$ 
- $$\delta_{\mathrm{U} \rightarrow c} \equiv \mathbb{E}_{\mathbf{z}, \mathrm{P}}\left[\mathbf{s}_{c}\left(\mathbf{x}_{i}\right)\right]-\mathbb{E}_{\mathbf{z}, \mathrm{P}}\left[\mathbf{s}_{c}\left(\mathbf{x}_{a}\right)\right]$$.

<br>

![figure2](/assets/img/gan/img48.png)