---
title: Data Augmentation for Seizure Prediction with Generative Diffusion Model
categories: [TS,GAN,CL]
tags: []
excerpt: arXiv 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Data Augmentation for Seizure Prediction with Generative Diffusion Model

<br>

# Abstract

Task: EEG-based seizure prediction

Existing DA methodos generate samples by **overrlapping / recombining** prerictal data for each seizure

$$\rightarrow$$ limited by original data

Solution: propose **DiffEEG**

<br>

# Previous Methods

![figure2](/assets/img/ts/img579.png)

<br>

# DiffEEG

Diffusion network

- conditined on the **initial STFT spectogram** to provide the guiding time-frequency features

![figure2](/assets/img/ts/img580.png)
