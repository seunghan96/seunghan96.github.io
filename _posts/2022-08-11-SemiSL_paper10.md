---
title: (paper) SSL10 - FixMatch
categories: [SSL]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (2020)

<br>

## Contents

0. Abstract
1. Procedure of FixMatch

<br>

# 0. Abstract

FixMatch

- significant simplification of existing SSL methods

<br>

# 1. Procedure of FixMatch

step 1) generates **pseudo-labels** 

- using the model’s predictions on **weakly augmented unlabeled images**
- ( pseudo-label is only retained if the model produces a **high-confidence prediction** )

step 2) trained to **predict the pseudo-label**

- when fed a **strongly-augmented version** of the same image

<br>

![figure2](/assets/img/semi/img21.png)
