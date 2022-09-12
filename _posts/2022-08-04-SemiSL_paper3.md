---
title: (paper) SSL03 - Noisy Sutdent
categories: [ML]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Self-training with Noisy Student improves ImageNet classification (2020)

<br>

## Contents

0. Abstract
1. NoisyStudent : Iterative Self-training with Noise

<br>

# 0. Abstract

propose simple self-training method 

- step 1) first train an EfficientNet model on labeled ImageNet
- step 2) use it as **teacher** to generate **pseudo labels**
- step 3) train a larger EfficientNet as a student model
  - with both **labeled** & **unlabeled**

<br>

# 1. NoisyStudent : Iterative Self-training with Noise

Step 1) train a teacher model

- with labeled images

<br>

Step 2) generate pseudo-labels

- generated with **teacher model** on **unlabeled images**
- soft & hard version

<br>

Step 3) train a student model

- minimizes the combined cross entropy loss on both labeled images and unlabeled images

<br>

Step 4) iterate the process

- putting back the student as a teacher

<br>

![figure2](/assets/img/semi/img9.png)

<br>

![figure2](/assets/img/semi/img10.png)
