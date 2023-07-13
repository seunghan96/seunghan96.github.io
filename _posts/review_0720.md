# Accelerating Batch Active Learning Using Continual Learning Techniques

<br>

# 0. Abstract

Problem of Active Learning (AL) = high training costs

- since models are typically retrained from scratch after every query round. 

<br>

Standard AL on NN with warm stating ..

- Fails to accelerating training
- Fails to avoid catastrophic forgetting

when fine-tuning over AL query rounds

<br>

### Proposal) Continual Active Learning

By **biasing** further training towards **previously labeled sets**

- via replay-based Continual Learning (CL) algorithms
  - effective at quickly learning the new without forgetting the old

<br>

# 1. Introduction

Problems of NN

- labeled-data hungry
- require significant computation

<br>

Active learning (AL) 

- selects subsets of points to label from a large pool of unlabeled data
- incrementally add points to the labeled pool
- shown to reduce the amount of training data required
- Procedure of AL
  - step 1)  Train a model fom scratch ( using $D_L$ )
  - step 2) Measeure model uncertainty / Diversity to select a set of points to query
- Problem of AL
  - can be computationally expensive since it requires retraining a model after each query round.

<br>

### [ Solution 1 ]

**Warm start** the model parameters between query rounds

<br>

BUT not a good solution..

- reason 1) tend to still be limited, since the model must make several passes through an **ever-increasing pool of data**
- reason 2) warm starting alone in some cases can hurt generalization

<br>

### [ Solution 2 ]

Solely train on the newly labeled batch of examples to avoid re-initialization. 

STILL not a good solution ...

- fails to retain accuracy on previously seen examples,

  since the distn of the query pool may drastically change with each round.

  ( = catastrophic foregtting )

$\rightarrow$ ***Continual Learning***

<br>

### Continual Active Learning (CAL)

