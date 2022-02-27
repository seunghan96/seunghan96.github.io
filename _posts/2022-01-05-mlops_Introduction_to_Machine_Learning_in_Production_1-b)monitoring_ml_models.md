---
title: \[Week 1-b\] Monitoring Machine Learning Models in Production
categories: [MLOPS]
tags: []
excerpt: (coursera) Introduction to ML in production - 1.Overview of the ML Lifecycle and Deployment
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Monitoring Machine Learning Models in Production

( reference : https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/ )

<br>

Contents

1. ML System Life Cycle
2. What makes ML System Monitoring hard
3. Why you need monitoring

<br>

# 1. ML System Life Cycle

monitoring of ML models :

- track & understand model performance "in production"
- perspective in both (1) data science & (2) operation

<br>

[Continuous Delivery for Machine Learning (CD4ML)](https://martinfowler.com/articles/cd4ml.html) ( Martin Fowler )

![figure2](/assets/img/mlops/img25.png)

<br>

Phase 1. **Model Building**: 

- define the problem
- data preparation, feature engineering, (initial) modeling code

<br>

Phase 2. **Model Evaluation and Experimentation**: 

- feature selection, hyperparameter tuning
- compare with different algorithm
- check metrics

<br>

Phase 3. **Productionize Model**: 

- preparation for deployment

- result : production-grade code

  ( can be in a completely different programming language / framework )

<br>

Phase 4. **Testing**: 

- check if it works as we expected

  ( check if it matches with (step 2) )

<br>

Phase 5. **Deployment**: 

- put into production
- use APIs to access the model.

<br>

Phase 6. **Monitoring and Observability**: 

- keep monitoring if it is working as we expected

<br>

Phase 1 & 2 ( model building & evaluation )

$$\rightarrow$$ ***research environment*** ( by data scientists )

<br>

Phase 3 & 4 & 5 & 6

$$\rightarrow$$ ***engineering, DevOps***

<br>

## Monitoring Scenarios

![figure2](/assets/img/mlops/img26.png)

Scenario \# 1 : deployment of brand new model

Scenario \# 2 : replacement of existing model

Scenario \# 3 : making small tweaks in current model

- ex) change in certain variable ( unable to use )
- ex) found a super feature! ( will add that feature )

<br>

# 2. What makes ML System Monitoring hard

***The model is a tiny fraction of an overall ML system!***

![figure2](/assets/img/mlops/img27.png)

<br>

3 components of ML system

- 1) Code ( + config )
- 2) Model
- 3) Data

<br>

[ CAUTION ]

ML system’s is governed "not just by code", but also by **"model behavior learned from data"**. 

\+ **data inputs change over time!**  

<br>

Complexity of Code & Data ( + config )

- Entaglement
  - Data changed $$\rightarrow$$ Model weights changes
  - ***changing anything changes everything***
- Configuration
  - slight change in hyperparmeter/version/feature can cause big change!

<br>

## Responsibility Challenge

Teams of ML Systems

![figure2](/assets/img/mlops/img28.png)

<br>

Data Scientists & Engineers/Devops

- Data Scientists
  - focus on statistical tests on model inputs and outputs
- Engineers/Devops
  - monitoring, production, deployment...

$$\rightarrow$$ need BOTH perspectives!

<br>

Traditionally ...

- monitoring & observability  : just part of DevOps

But nowadays (ML System) .....

- unlikely that your DevOps team will have the necessary expertise to correctly monitor ML models

<br>

THEREFORE, **whole team needs to work together** !! **MLOps!!**

<br>

# 3. Why you need monitoring

***“By failing to prepare, you are preparing to fail”*** ( Benjamin Franklin )

<br>

look for xxxxx

<br>