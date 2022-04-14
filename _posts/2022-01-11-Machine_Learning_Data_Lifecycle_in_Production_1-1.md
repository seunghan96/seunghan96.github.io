---
title: \[Week 1-1\] Introduction to Machine Learning Engineering in Production
categories: [MLOPS]
tags: []
excerpt: (coursera) Machine Learning Data Lifecycle in Production - Collecting, Labeling, Validating Data
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( reference : [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) ) 

# Introduction to Machine Learning Engineering in Production

## [1] Overview

- ML enginerring for **PRODUCTION**
- **Production ML** = (1) + (2)
  - (1) ML development
  - (2) software development
- Challenges in **production ML**

<br>

### Traditional ML vs Producton ML

Main difference :

- ***production ML requires much more than just a modeling code!!***
- ***data is NOT STATIC in production ML!!***

<br>

[ Traditional ML ]

![figure2](/assets/img/mlops/img67.png)

<br>

[ Production ML ]

![figure2](/assets/img/mlops/img68.png)

<br>

![figure2](/assets/img/mlops/img69.png)

<br>

### Manage the entire life cycle of data

- labeling
  - is it properly labeled?
- feature space coverage
  - do they always have the same feature space?
- minimal dimensionality
  - reduce the dimension of feature to optimize performance
- maximum predictive data
  - does the data have predictive information?

<br>

### Production ML system

![figure2](/assets/img/mlops/img70.png)

![figure2](/assets/img/mlops/img71.png)

- continuosly moniter the model performance,
  ingest new data,
  retrain when needed,
  redeploy to maintain / improve the performance

<br>

### Challenges in production grade ML

- have to build an **INTEGRATED** ML system
- need to **CONTINUOSLY** operate it in production
- handle **CONTINUOSLY CHANGING DATA**
- optimimze compute resource costs

<br>

## [2] ML Pipelines

Outline

- ML Pipelines
- DAG (Directed Acyclic Graphs) & Pipeline Orchestration Frameworks
- TFX ( Tensorflow Extended )

<br>

### ML pipeline

![figure2](/assets/img/mlops/img72.png)

<br>

### DAG ( Directed Acyclic Graphs )

- directed graphs with NO cycles
- ML pipeline workflows : usually DAGs
  - sqeuencing of tasks
  - have relationships/dependencies with each other

![figure2](/assets/img/mlops/img73.png)

<br>

### Pipeline Orchestration Frameworks

- GOAL : **schedule components in ML pipelines**
- make **pipeline automation**
- ex) Airflow, Argo, Celery, Luigi, Kubeflow

![figure2](/assets/img/mlops/img74.png)

<br>

### TFX ( Tensorflow Extended, TFX )

- end-to-end platform,

  for deploying **production ML pipelines**

![figure2](/assets/img/mlops/img75.png)

<br>
