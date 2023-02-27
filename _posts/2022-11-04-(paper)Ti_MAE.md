---
title: (paper) Ti-MAE ; Self-Supervised Masked TS Auto Encoders
categories: [TS,CL]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Ti-MAE : Self-Supervised Masked Time Series Auto Encoders

( https://openreview.net/pdf?id=9AuIMiZhkL2 )

<br>

## Contents

0. Abstract
1. Introduction
2. Related Works
3. Methodology
   1. Problem Definition
   2. Model Architecture
4. Experiments
   1. Experimental Setup
   2. TS Forecasting
   3. TS Classification

<br>

# 1. Abstract

**Contrastive learning** & **Transformer-based models** : good performance on **long-term** TS forecasting

<br>

Problems :

- (1) Contrastive learning :

  - training paradigm of contrastive learning and downstream prediction tasks are **inconsistent**

- (2)  Transformer-based models :

  - resort to similar patterns in historical time series data for predicting future values

    $$\rightarrow$$ induce severe **distribution shift problems**

  - do not fully leverage the **sequence information** ( compared to SSL )

<br>

### Ti-MAE

- input TS : assumed to follow an **integrate distribution**

- **randomly masks out TS & reconstruct them**

- adopts **”mask modeling”** (rather than contrastive learning) as the auxiliary task 

  - bridges the connection between existing representation learning & generative Transformer-based methods

    $$\rightarrow$$ **reducing the difference between upstream and downstream forecasting tasks**

<br>

# 1. Introduction

generative Transformer-based models 

= a special kind of denoising autoencoders

( where we only mask the future values and reconstruct them )

![figure2](/assets/img/cl/img314.png)

<br>

2 problems in **continuous masking strategy**

- (1) captures only the information of the visible sequence and some mapping relationship between the **historical and the future segments**

- (2) induce severe distribution shift problems

  - especially when the prediction horizon is longer than input sequence

    ( In reality : most are **non-stationary** )

<br>

### Disentangled TS

- $$\boldsymbol{y}(t)=\operatorname{Trend}(t)+\operatorname{Seasonality}(t)+\text { Noises}$$.
  - Trend : $$\sum_n t^n$$
    - while the moments of the trend part change continuously over time
  - Seasonality : $$\sum_n \cos ^n t$$
    - stationary, when we set a proper observation horizon

<br>

 ***Size of sliding window*** to obtain trend is vital !

![figure2](/assets/img/cl/img315.png)

<br>

### Problem in decomposition

1. Natural time series data generally have more complex periodic patterns

   $$\rightarrow$$ have to employ longer sliding windows or other hierarchical disposals

2. Ends of a sequence need to be padded for alignment

   $$\rightarrow$$ causes inevitable data distortion at the head and tail

<br>

### Ti-MAE

proposes a novel Transformer-based framework : **Ti-MAE**

![figure2](/assets/img/cl/img316.png)

- randomly masks out parts of embedded TS
- learns AE to reconstruct them ( at the point-level )

<br>

Random masking ( vs. Fixed continuous masking )

- (1) takes the overall distribution of inputs 

  $$\rightarrow$$ alleviate the distribution shift problem

- (2) Encoder-Decoder structure : provides a universal scheme for both forecasting and classification

<br>

### Contributions

1. Novel perspective to bridge the connection between existing **(1) contrastive learning** and **(2) generative Transformer-based models** on TS

   & point out the inconsistency and deficiencies of them on downstream tasks

2. Propose Ti-MAE

   - a masked time series autoencoders 

   - learn strong representations with less inductive bias or hierarchical trick

     - pros 1) adequately leverages the input TS & successfully alleviates the distribution shift problem

     - pros 2) due to the flexible setting of masking ratio….

       $$\rightarrow$$ can adapt to complex scenarios which require the trained model to make forecasting simultaneously for multiple time windows with various sizes without re-training

3. achieved excellent performance for both **(1) forecasting** and **(2) classification** tasks

<br>

# 2. Related Works

## (1) Transformer-based TS model

Transformer : can capture long-range dependencies

<br>

ex 1) Song et al. (2018); Ma et al. (2019); LI et al. (2019) 

- directly apply vanilla Transformer to TS

- failed in long sequence TS forecasting tasks

  $$\because$$ self-attention operation scales quadratically with the input TS length

<br>

ex 2) Child et al. (2019); Zhou et al. (2021); Liu et al. (2022) 

- noticed the long tail distribution in self-attention feature map
- utilized **sparse attention mechanism** to reduce time complexity and memory usage
- but applying too long input TS in training stage will degrade the forecasting accuracy of the model

<br>

ex 3) latest works : ETSformer (Woo et al., 2022b) & FEDformer (Zhou et al., 2022)

- rely heavily on disentanglement and extra introduced domain knowledge

<br>

## (2) TS Representation Learning

SSL : good performance in TS domain ( especially **contrastive learning** )

<br>

ex 1) Lei et al. (2019); Franceschi et al. (2019) 

- used **loss function** of metric learning to preserve **pairwise similarities in the time domain**

<br>

ex 2) **CPC (van den Oord et al., 2018)** 

- first proposed **contrastive predictive coding and InfoNCE**
- which treats the …
  - data from the same sequence : **POS pairs**
  - different noise data from the mini-batch : **NEG pairs**

<br>

ex 3) **DA on TS data**

- capture **transformation-invariant features** at semantic level (Eldele et al., 2021; Yue et al., 2022)

<br>

ex 4) **CoST (Woo et al., 2022a) **

- introduced extra inductive biases in **frequency domain** through DFT
- separately processed disentangled **“trend and seasonality”** parts of the original TS
  - to encourage discriminative seasonal and trend representations

<br>

$$\rightarrow$$ Almost all of these methods rely on **heavily data augmentation** or other **domain knowledge** 

<br>

# 3. Methodology

## (1) Problem Definition

MTS : $$\mathcal{X}=\left(\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_T\right) \in \mathbb{R}^{T \times m}$$

<br>

Forecasing Task :

- input : $$\mathcal{X}_h \in \mathbb{R}^{h \times m}$$ with length of $$h$$
- target : $$\mathcal{X}_f \in \mathbb{R}^{k \times n}$$ : next $$k$$ steps values, where $$n \leq m$$

<br>

Classification Task : (pass)

<br>

## (2) Model Architecture

Encoder : maps $$\mathcal{X} \in \mathbb{R}^{T \times m}$$ to $$\mathcal{H} \in \mathbb{R}^{T \times n}$$

Decpder : reconstructs the original sequence from the embedding

<br>

\+ adopt an **asymmetric design**

- [Encoder] only operates visible tokens after applying masking on input embedding

- [Decoder] 
  - processes encoded tokens padded with masked tokens
  - reconstructs the original TS at the point-level

<br>

### a) Input Embedding

Model : **1d-conv**

- ( do not adopted any multi-scale or complex convolution scheme ( ex. dilated convolution ) )
- extract local temporal features on timestamp across channels

<br>

Positional Embeddings;

- fixed sinusoidal positional embeddings

<br>

\+ do not add any handcrafting task-specific / date-specific embeddings 

( so as to introduce as little inductive bias as possible )

<br>

### b) Masking

After tokenizing …. 

(1) randomly sample a subset of tokens 

- without replacement  + from uniform distribution

(2) mask the remaining parts



### Masking Ratio

(He et al., 2021; Feichtenhofer et al., 2022) 

- related to the information density and redundancy of the data

- immense impact on the performance of the AE

- applications
  - natural language : higher information density ( due to its highly discrete word distn )
    - BERT : 15%
  - images : heavy spatial redundancy ( single pixel in one image has lower semantic information )
    - MAE for image : 75%
    - MAE for videos : 90%

- data with lower information density : ***should be applied a higher masking ratio***
- TS data : ( similar as images ) also have local continuity 
  - determine a high masking ratio ( = 75% )

<br>

### c) Ti-MAE Encoder

vanilla Transformer blocks

- utilizes pre-norm ( instead of post-norm )
- applied **only on visible tokens** after embedding and random masking
  - significantly reduces time complexity and memory usage ( compared to full encoding )

![figure2](/assets/img/cl/img317.png)

<br>

### d) Ti-MAE Decoder

vanilla Transformer blocks 

- applied on the union of the **(1) encoded visible tokens** & **(2) learnable randomly initialized mask tokens**
- smaller than the encoder
- add positional embeddings to all tokens after padding
- last layer : linear projection
  - reconstructs the input by predicting all the values at the point-level

<br>

Loss function : MSE ( on **masking regions** )

<br>

encoder and decoder of Ti-MAE :

- both agnostic to the sequential data with as less domain knowledge as possible
- no date-specific embedding, hierarchy or disentanglement
- point-level modeling rather than patch embedding 
  - for the consistency between **(1) masked modeling** & **(2) downstream forecasting tasks**

<br>

# 4. Experiments

## (1) Experimental Setup

### a) Dataset & Tasks

Task :

- **(1) TS forecasting**
- **(2) TS classification**

<br>

Dataset :

- (1) **ETT (Electricity Transformer Temperature)**  (Zhou et al., 2021) 
  - data collected from electricity transformers
  - recording six power load features and oil temperature
- (2) **Weather1** 
  - contains 21 meteorological indicators like humidity, pressure in 2020 year from nearly 1600 locations in the U.S
- (3) **Exchange** (Lai et al., 2018) 
  - collection of exchange rates among eight different countries from 1990 to 2016
- (4) **ILI2** 
  - records the weekly influenza-like illness (ILI) patients dat
- (5) **The UCR archive** (Dau et al., 2019) 
  - 128 different datasets covering multiple domains 

<br>

Data Split

- (ETT) 6 : 2 : 2
- (others) 7 : 1 : 2

<br>

Classification ( UCR archive ) : has been already divided into training and test set

-  where the size of **test >> train**

<br>

### b) Baselines

2 types of baselines

- (1) Transformer-based end-to-end
- (2) Representation learning methods ( which have public official codes )

<br>

Time series forecasting 

- 4 SOTA **Representation learning models** : 
  - CoST (Woo et al., 2022a)
  - TS2Vec (Yue et al., 2022)
  - TNC (Tonekaboni et al., 2021)
  - MoCo (Chen et al., 2021)
- 4 SOTA **Transformer-based end-to-end models** : 
  - FEDformer (Zhou et al., 2022)
  - ETSformer (Woo et al., 2022b)
  - Autoformer (Wu et al., 2021)
  - Informer (Zhou et al., 2021)

<br>

Time series classification 

- include more competitive unsupervised representation learning methods
  - TS2Vec (Franceschi et al., 2019)
  - T-Loss (Franceschi et al., 2019)
  - TS-TCC (Eldele et al., 2021)
  - TST (Zerveas et al., 2021)
  - TNC (Tonekaboni et al., 2021)
  - DTW (Chen et al., 2013)

<br>

### c) Implementation Details

Encoder and Decoder : 

- 2 layers of vanilla Transformer blocks 
  - with 4 heads self-attention

- hidden dimension is = 64

<br>

Others 

- Optimizer : Adam ( lr = 1e-3 )
- Batch Size : 64
- Samplimg Time : 30 in each iteration

<br>

Evaluation metric

- forecasting : MSE & MAE
- classificaiton : average ACC ( + critical difference (CD) ) 

<br>

## (2) TS Forecasting

under different future horizons ( for both short & long term )

### a) Table 1 ( vs. representation learning models )

![figure2](/assets/img/cl/img318.png)

- does not require any extra regressor after pre-trained 

  ( $$\because$$ decoder can directly generate future TS )

<br>

### b) Table 2 ( vs. Transformer-based models )

![figure2](/assets/img/cl/img319.png)

- pre-trained only one Ti-MAE model 

  ( $$\leftrightarrow$$ end-to-end supervised models : should be trained separately for different settings )

- Ti-MAE († : fine-tuned version) : just utilize its (frozen) encoder with an additional linear projection layer

<br>

### c) Ablation Study

![figure2](/assets/img/cl/img320.png)

<br>

## (3) TS Classification

Instance-level representation on classification tasks. 

Dataset : 128 UCR archive

### a) Accuracy

![figure2](/assets/img/cl/img321.png)

<br>

### b) Critical Difference diagram (Demsar, 2006) 

classifiers connected by a bold line do not have a significant difference.

![figure2](/assets/img/cl/img322.png)

<br>

# 5. Conclusion

novel self-supervised framework : Ti-MAE 

- **randomly masks out** tokenized TS

- learns an **AE** to reconstruct them at the point-level

- bridges the connection between 

  - (1) **contrastive representation learning**
  - (2) **generative Transformer-based method**

- improves the performance on forecasting tasks …

  - **due to reducing the inconsistency of upstream and downstream tasks**

- **Random masking strategy** :

  - leverages all the input sequence
  - alleviates the distribution shift problem

  $$\rightarrow$$ makes Ti-MAE more adaptive to various prediction scenarios with different time steps

