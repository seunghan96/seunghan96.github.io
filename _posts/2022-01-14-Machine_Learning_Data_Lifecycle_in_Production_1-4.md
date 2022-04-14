---
title: \[Week 1-4\] Validating Data
categories: [MLOPS]
tags: []
excerpt: (coursera) Machine Learning Data Lifecycle in Production - Collecting, Labeling and Validating Data
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( reference : [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) ) 

# Validating Data

## [1] Detecting Data Issues

### Outline

- Data Issues : **drift & skew**
  - data & concept **drift**
  - schema & distribution **skew**
- Detecting Data Issues

<br>

### Drift & Skew

Drift : change over time

- **Data drift**
  - change in statistical properties of features
- **Concept drift**
  - change in statistical properties of labels
- model/performance may decay due to those drifts!

<br>

Skew : difference between 2 static versions

- ex) training & serving dataset

<br>

![figure2](/assets/img/mlops/img82.png)

<br>

### Detecting data issues

Detect …

- (1) Schema skew
- (2) Distribution skew

-> requires **continuous evaluation**

<br>

Schema skew

- Training data’s schema & Serving data’s schema are different

<br>

Distribution skew

- dataset / covariate / concepth shift

![figure2](/assets/img/mlops/img83.png)

<br>

### Skew detection workflow

![figure2](/assets/img/mlops/img84.png)

- when significant changes, trigger an alert!

<br>

## [2] TFDV ( TF Data Validation )

Purpose : **understand / validate / monitor ML data at scale**

Capabilities of TFDV

- generate data statistics ( + visualization )
- infers data schema & validity check
- detect training-serving skew

<br>

### Skew detection

![figure2](/assets/img/mlops/img84.png)

<br>

(1) Schema skew : 

training & serving data have differnent schema

- ex) int != float

<br>

(2) Feature skew :

training & serving data have different feature values

- ex) transformation is applied to only one

<br>

(3) Distribution skew :

training & serving data have different distribution

- ex) different data sources
- ex) change in trend/seasonality

<br>

## [3] Summary

1. Traditional ML modeling vs Production ML System
2. Responsible Data Collection ( for fair production ML )
3. Process Feedback & Human Labeling
4. Detect Data Issues

<br>

https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html
