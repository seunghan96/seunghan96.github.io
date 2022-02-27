---
title: \[Week 1-1\] Introduction
categories: [MLOPS]
tags: []
excerpt: (coursera) Introduction to ML in production \\ - Overview of the ML Lifecycle and Deployment
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# ML engineering production, MLOps 

accurate ML model in Jupyter Notebook

$\rightarrow$ have to put the model into production, 



Will learn skills you need to build  & deploy Production ML systems. 

<br>

Full ML project life-cycle

- 1) scoping : 
- 2) data : 
- 3) modeling
- 4) deployment

![figure2](/assets/img/mlops/img1.png)

<br>

Discipline of building & maintaining production systems, 

$\rightarrow$ "MLOps" ( = Machine Learning Operations )

<br>

## MLOps Process

![figure2](/assets/img/mlops/img2.png)

<br>

Data drift

- distribution of the data you trained $\neq$ distribution of the data that you're running inference

<br>

key point is "CHANGE"

- world changes & model needs to be aware of that change. 

<br>

Will deal with...

- building "data pipelines" by gathering, cleaning, validating data sets using TFX + $\alpha$
- analytics to address model fairness & explainability issues
- "deployment" : serve the users' requests





