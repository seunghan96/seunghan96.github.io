---
title: \[Week 1-3\] Labeling Data
categories: [MLOPS]
tags: []
excerpt: (coursera) Machine Learning Data Lifecycle in Production - Collecting, Labeling and Validating Data
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( reference : [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) ) 

# Labeling Data

## [1] Degrated Model Performance

example ) Situation, where…

- goal : predict **CTR**, to decide how much **inventory to order**
- problem : **AUC, accuracy dropped** on men’s dress shoes

<br>

### Key Questions

- (1) how to **DETECT problems** early?
- (2) what are the **CAUSES of problems**?
- (3) what are the **SOLUTIONS** ?

<br>

### Cause of problems

2 categories :

- (1) slow problems
- (2) fast problems

<br>

(1) Slow : gradual problems

- ex) Drift
  - world changes, season changes, competitors enter…

(2) Fast : sudden problems

- ex) bad sensor, bad SW update

<br>

### Gradual Problems

(1) DATA changes

- change in “trend & seasonality”
- change in “distn of feature”
- change in “relative importance of features”

<br>

(2) WORLD changes

- Style, competitors changes

<br>

### Sudden Problems

(1) DATA COLLECTION problem

- bad sensor, log data

(2) SYSTEM problem

- bad SW update, system down, …

<br>

### Understanding the model

Why do we need to understand our model??

- all mispredictions do not have same cost!
- rarely have all the data you want
- model objective : just **proxy** for business goal
  - ex) “inventory prediction” & “CTR prediction”

most of all, *** THE WORLD KEEPS CHANGING***

<br>

## [2] Data & Concept change

- detecting problems with deployed models
  - data & concept keeps changing!
- changing ground truth
  - easy / hard / harder problems

<br>

### Detecting problems

(1) Data & Scope changes

-> need to **monitor models** & **validate data**, to find problems early

(2) Ground truth changes

-> have to label **NEW training data**

<br>

### Categories of problems

(1) Easy problems : 

- cause : SLOW changes
- motivation of model retrain :
  - model improvement, better data
  - changes in SW, systems
- labeling :
  - crowd-based

<br>

(2) Hard problems

- cause : FAST changes
- motivation of model retrain : (1) +
  - **decline** in model performance
- labeling :
  - direct feedback
  - Crowd-based

<br>

(3) VERY HARD problems

- cause : VERY FAST changes
- motivation of model retrain : (2)
- Labeling :
  - direct feedback
  - weak supervision

<br>

### Keypoints

- Model performance **DECAYS** over time

- Need to **RETRAIN MODEL**, when needed

<br>

## [3] Process Feedback & Human Labeling

### Methods for Data Labeling

[ basic methods ]

- process feedback ( = direct labeling )
- human labeling

<br>

[ advanced ]

- semi-supervised learning
- active learning
- weak supervision

<br>

### Process Feedback ( = Direct Labeling )

continuous creation of training dataset

- can get feedback ( ex. with log data )

- ex) actual vs predicted CTR

![figure2](/assets/img/mlops/img79.png)

<br>

Advantages

- continuous creation of dataset
- labels evolve quickly

Disadvantages

- failture to capture ground truth

<br>

example of Open source **log analysis tool**

- Logstash, Fluentd, Google Cloud Logging, AWS ElasticSearch, Azure Monitor

<br>

### Human Labeling

Humans directly label the data manually!

- ex) labeling MRI images with Cardiologists

![figure2](/assets/img/mlops/img80.png)

<br>

Steps of human labeling

![figure2](/assets/img/mlops/img81.png)

<br>

Advantages

- more labels
- Pure supervised learning

Disadvantages

- quality consistency
- slow & expensive
