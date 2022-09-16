---
title: (paper 31) Invariance Propagation
categories: [CL, CV]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Unsupervised Representation Learning by Invariance Propagation

<br>

## Contents

0. Abstract
0. Introduction
0. Method
   0. Self-labeling


<br>

# 0. Abstract

Unsupervised learning based on contrastive learning

- aim to learn representations, invariant to **instance-level variations**

<br>

propose **Invariance Propagation**

- focus on learning representations, invariant to **category-level variations**

- **recursively** discovers semantically consistent samples, which are in the **same high-density regions**
- **hard sampling** 

combining (1) clustering + (2) representation learning

$$\rightarrow$$ doing it naively...leads to **degenerate solutions**

<br>

solution : propose a method, that **maximizes the information between labels & input data indicies**

<br>

# 1. Introduction
