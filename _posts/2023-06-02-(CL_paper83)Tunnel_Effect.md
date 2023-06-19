---
title: (paper 83) The Tunnel Effect; Building Data Representations in Deep Neural NEtworks
categories: [CV, CL]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# The Tunnel Effect: Building Data Representations in Deep Neural NEtworks

<br>

## Contents

0. Abstract
1. Introduction
2. The Tunnel Effect
   1. Experimental Steup
   2. Main Result
3. Tunnel Effect Analysis
   1. Tunnel Development
   2. Compression and OOD generalization
   3. Network Capacity & Dataset Complexity
4. The Tunenl Effect under Data Distribution Shift
   1. Exploring the effect of task incremental learning on extractor and tunnel
   2. Reducing catastrophic forgetting by adjusting network depth


<br>

# 0. Abstract

DNN :  learn more complex data representations. 

<br>

This paper shows that sufficiently deep networks trained for supervised image classification **split into TWO distinct parts** that contribute to the resulting data representations differently. 

- ***Initial layers*** : create linearly separable representations
- ***Subsequent layers ( = tunnel )*** : compress these representations & have a minimal impact on the overall performance.

<br>

Explore the tunnel’s behavior  ( via empirical studies )

- emerges early in the training process
- depth depends on the relation between the network’s capacity and task complexity.
- tunnel degrades OOD generalization ( + implications for continual learning )

<br>

# 1. Introduction

Consensus : Networks learn to use **layers in the hierarchy** 

- by extracting more complex features than the layers before

  ( = each layer contributes to the final network performance )

$$\rightarrow$$ is this really true?

<br>

However, practical scenarios :

Deep and Overparameterized NN tend to simplify representations with increasing depth

- WHY?? Despite their large capacity, these networks strive to reduce dimensionality and focus on discriminative patterns during supervised training

$$\rightarrow$$ This paper is motivated by these contradictory findings!

***How do representations depend on the depth of a layer?***

<br>

Focuses on severely overparameterized NN

Challenge the common intuition that deeper layers are responsible for capturing more complex and task-specific features

$$\rightarrow$$ Demonstrate that DNN split into **two parts** exhibiting distinct behavior. 

<br>

Two parts

- FIRST part ( = Extractor ) : builds representations

- SECOND part ( = Tunnel ) : propagates the representations further to the model’s output

  (= compress the representations )

<br>

Findings

- (1) Discover **Tunnel effect** = DNN naturally split into 

  - Extractor : responsible for building representations
  - Compressing tunnel : minimally contributes to the final performance. 

  Extractor-tunnel split emerges EARLY in training and persists later on

- Show that the **tunnel deteriorates the generalization ability on OOD data**
- Show that the **tunnel exhibits task-agnostic behavior** in a **continual learning** scenario. 
  - leads to higher catastrophic forgetting of the model.

<br>

# 2. The Tunnel Effect

Dynamics of representation building in overparameterized DNN

Section Introduction

- (3.1) Tunnel effect is present from the initial stages & persists throughout the training process. 

- (3.2) Focuses on the OOD generalization and representations compression.

- (3.3) Important factors that impact the depth of the tunnel. 

- (4) Auxiliary question: 

  - ***How does the tunnel’s existence impact a model’s adaptability to changing tasks and its vulnerability to catastrophic forgetting?***

    $$\rightarrow$$ Main claim : The tunnel effect hypothesis

<br>

***Tunnel Effect Hypothesis*** : 

Sufficiently large NN develop a configuration in which network layers split into two distinct groups. 

- (1) extractor : builds linearly separable representations

- (2) tunnel : compresses these representations

  ( = hinders the model’s OOD generalization )

<br>

## (1) Experimental setup 

**a) Architectures** 

- MLP, VGGs, and ResNets. 

- vary the number of layers & width of networks to test the generalizability of results. 

<br>

**b) Tasks** 

- CIFAR-10, CIFAR-100, and CINIC-10. 
  - number of classes: 10 for CIFAR-10 and CINIC-10 and 100 for CIFAR-100, 
  - number of samples: 50000 for CIFAR-10 and CIFAR-100 and 250000 for CINIC-10

<br>

**c) Probe the effects using**

- (1) Average accuracy of linear probing
- (2) Spectral analysis of representations
- (3) CKA similarity between representations. 

( = Unless stated otherwise, we report the average of 3 runs. )

<br>

\* Accuracy of linear probing: 

- linear classification layer ( train this layer on the classification task )
- measures to what extent $$l$$‘s’ representations are linearly separable. 

<br>

\* Numerical rank of representations: 

- compute singular values of the sample covariance matrix for a given layer $$l$$ of NN
- estimate the numerical rank of the given representations matrix as the **number of singular values** above a certain threshold
- can be interpreted as the measure of the **degeneracy of the matrix.** 

<br>

\* CKA similarity: 

- similarity between two representations matrices.
- can identify the blocks of similar representations within the network. 

<br>

\* Inter and Intra class variance: 

- Inter-class variance = measures of dispersion or dissimilarity between different classes
- Intra-class variance = measures the variability within a single class

<br>

## (2) Main Result

![figure2](/assets/img/cl/img209.png)

![figure2](/assets/img/cl/img210.png)

<br>

### Figure 1 & 2

**Early layers** of the networks ( = 5 for MLP and 8 for VGG ) are responsible for building **linearly-separable representations**. 

These layers mark the transition between the **extractor** & the **tunnel** 

For ResNets, the transition takes place in deeper stages of the network  ( = 19th layer )

- linear probe performance nearly saturates in the tunnel part, the representations are further refined. 

<br>

### Figure 2

Numerical rank of the representations is reduced to approximately the **number of CIFAR-10 classes**

For ResNets, the numerical rank is more dynamic, 

- exhibiting a spike at 29th layer ( = coincides with the end of the penultimate residual block )

Rank is higher than in the case of MLPs and VGGs.

<br>

![figure2](/assets/img/cl/img211.png)

<br>

### Figure 3

(For VGG-19) 

Inter-class variation 

- decreases throughout the tunnel

Inter-class variance

- increases throughout the tunnel

<br>

Right plot : intuitive explanation of the behavior with UMAP plots 

<br>

![figure2](/assets/img/cl/img212.png)

<br>

### Figure 4

Similarity of MLPs representations using the CKA index & the L1 norm of representations differences between the layers. 

- representations change significantly in early layers & remain similar in the tunnel part

<br>

# 3. Tunnel Effect Analysis

Empirical evidence for tunnel effect.

- A) Tunnel develops early during training time

- B) Tunnel compresses the representations & hinders OOD generalization

- C) Tunnel size is correlated with network capacity and dataset complexity

<br>

## (1) Tunnel Development

### a) Motivation

- Understand whether the tunnel is a phenomenon exclusively related to the representations 

- Understand which part of the training is crucial for tunnel formation. 

<br>

### b) Experiments

- train VGG19 on CIFAR-10
- checkpoint every 10 epochs

<br>

### c) Results

![figure2](/assets/img/cl/img213.png)

<br>

Split between the EXTRACTOR & TUNNEL is also visible in the parameters space. 

- at the early stages, and after that, its length stays roughly constant. 

  ( = change significantly less than layers from the extractor. )

$$\rightarrow$$ Question of whether the weight change affects the network’s final output.

<br>

Reset the weights of these layers to the state before optimization. 

However, the performance of the model deteriorated significantly. 

$$\rightarrow$$ Although the change within the tunnel’s parameters is relatively small, it plays an important role in the model’s performance!

<br>

![figure2](/assets/img/cl/img214.png)

The rank collapses to values near-the-number of classes. 

<br>

### d) Takeaway

- Tunnel formation is observable in the representation and parameter space
- Emerges early in training & persists throughout the whole optimization. 
- The collapse in the numerical rank of deeper layers suggest that they preserve ***only the necessary information required for the task.***

<br>

## (2) Compression and OOD generalization

### a) Motivation

Intermediate layers perform better than the penultimate ones for transfer learning … but WHY??

$$\rightarrow$$  Investigate whether the tunnel &  the collapse of numerical rank within the tunnel impacts the performance on OOD data.

<br>

### b) Experiments

- architecture : MLPs, VGG-10, ResNet-34
- Source task : CIFAR-10
- Target Task : OOD task ( with linear probes )
  - subset of 10 classes from CIFAR-100
- metric : accuracy of linear probing & numerical rank of representations

<br>

### c) Results

![figure2](/assets/img/cl/img215.png)

Tunnel is responsible for the degradation of OOD performance

$$\rightarrow$$ Last layer before the tunnel is the optimal choice for training a linear classifier on external data. 

$$\rightarrow$$ OOD performance is tightly coupled with numerical rank of representations

<br>

![figure2](/assets/img/cl/img216.png)

Additional dataset )

- Train a model on different subsets of CIFAR100
- Evaluate it with linear probes on CIFAR-10. 

<br>

Consistent relationship between the start of the tunnel & the drop in OOD performance. 

- Increasing number of classes in the source task $$\rightarrow$$ shorter tunnel & later drop in OOD performance. 

- Aligns with our earlier findings suggesting that the **tunnel is a prevalent characteristic of the model** 

  ( rather than an artifact of a particular training or dataset setup )

<br>

### d) Takeaway

Compression of representations ( in the tunnel )

$$\rightarrow$$ severely degrades the OOD performance!! 

( = by drop of representations rank )

<br>

## (3) Network Capacity & Dataset Complexity

### a) Motivation



Explore what factors contribute to the tunnel’s emergence. 

- explore the impact of dataset complexity, network’s depth, and width on tunnel emergence. 

<br>

### b) Experiments

(1) Examine the impact of **networks’ depth and width on the tunnel** 

- using MLPs, VGGs, and ResNets trained on CIFAR-10. 

(2) Investigate the role of **dataset complexity** on the tunnel

- using VGG-19 and ResNet34 on CIFAR-{10,100} and CINIC-10 dataset 

<br>

### c) Results

![figure2](/assets/img/cl/img217.png)

[Figure 9]

Depth of the MLP network has no impact on the length of the extractor part. 

$$\rightarrow$$ ***Increasing the network’s depth contributes only to the tunnel’s length!*** 

$$\rightarrow$$ Overparameterized NN allocate a fixed capacity for a given task independent of the overall capacity of the model.

<br>

[Table 2]

Tunnel length increases as the width of the network grows

( = implying that representations are formed using fewer layers. )

$$\leftrightarrow$$ However, this trend does not hold for ResNet34

<br>

![figure2](/assets/img/cl/img218.png)

[Table 3]

***The number of classes in the dataset directly affects the length of the tunnel.***

<br>

**a) Data size**

Even though the CINIC10 training dataset is three times larger than CIFAR-10, the tunnel length remains the same for both datasets. 

$$\rightarrow$$ Number of samples in the dataset does not impact the length of the tunnel.

<br>

**b) Number of classes**

In contrast, when examining CIFAR-100 subsets, the tunnel length for both VGGs and ResNets increase. 

$$\rightarrow$$ Clear relationship between the dataset’s number of classes and the tunnel’s length.

<br>

### d) Takeaway

Deeper or wider networks result in longer tunnels. 

& Networks trained on datasets with fewer classes have longer tunnels.

<br>

# 4. The Tunnel Effect under Data Distribution Shift

Investigate the dynamics of the tunnel in continual learning 

-  large models are often used on smaller tasks typically containing only a few classes. 

<br>

Focus on understanding the impact of the tunnel effect on ..

- (1) Transfer learning
- (2) Catastrophic forgetting 

Examine how the tunnel and extractor are altered after training on a new task.

<br>

## (1) Exploring the effect of task incremental learning on extractor and tunnel

### a) Motivation

Examine whether the extractor and the tunnel are **equally prone to catastrophic forgetting.**

<br>

### b) Experiments

- Architecture : VGG-19

- Two tasks from CIFAR-10

  - each task = 5 class

  - Subsequently train on the first and second tasks 

    & save the corresponding extractors $$E_t$$ and tunnels $$T_t$$ , where $$t \in \{1,2\}$$ are task numbers

- Separate CLS head for each task

<br>

### c) Results

![figure2](/assets/img/cl/img219.png)

<br>

### d) Takeaway

The tunnel’s **task-agnostic compression** of representations **provides immunity against catastrophic forgetting** when the number of classes is equal. 

<br>

## (2) Reducing Catastrophic Forgetting by adjusting network depth

### a) Motivation 

Whether it is possible to retain the performance of the original model by training a shorter version of the network. 

- A **shallower** model should also exhibit **less forgetting** in sequential training. 

<br>

### b) Experiments 

- Architecture : VGG-19 networks 
  - with different numbers of convolutional layers. 
- Each network : trained on two tasks from CIFAR-10. 
  - Each task consists of 5 classes

<br>

### c) Results

![figure2](/assets/img/cl/img220.png)

<br>

### d) Takeaway

Train **shallower networks** that retain the performance of the original networks & significantly less forgetting. 

However, the shorter networks need to have **at least the same capacity as the extractor part** of the original network.
