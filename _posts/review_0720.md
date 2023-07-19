# Accelerating Batch Active Learning Using Continual Learning Techniques

<br>

# 0. Abstract

Problem of **Active Learning (AL)** = high training costs

- since models are typically ***retrained from scratch*** after every query round. 

<br>

Standard AL on NN with **warm stating** ??

- (1) Fails to **accelerating training**
- (2) Fails to **avoid catastrophic forgetting**

when fine-tuning over AL query rounds

<br>

### Proposal) Continual Active Learning

By **biasing** further training towards **previously labeled sets**

- via **replay-based Continual Learning (CL)** algorithms
  
  ( = effective at quickly learning the new without forgetting the old )

<br>

# 1. Introduction

Problems of NN

- labeled-data hungry
- require significant computation

<br>

**Active learning (AL)** 

- **selects subsets of points to label** from a large pool of unlabeled data
- incrementally add points to the labeled pool
- shown to reduce the amount of training data required
- Procedure of AL
  - step 1)  Train a model fom scratch ( using $D_L$ )
  - step 2) Measure model ***uncertainty / diversity*** to select a set of points to query
- Problem of AL
  - can be ***"computationally expensive"*** since it requires retraining a model after each query round.

<br>

### [ Solution 1 ]

**WARM START** the model parameters between query rounds

<br>

BUT not a good solution..

- reason 1) tend to still be limited, since the model must make several passes through an **ever-increasing pool of data**
- reason 2) warm starting alone in some cases can **hurt generalization**

<br>

### [ Solution 2 ]

**SOLELY train on the newly labeled batch** of examples to avoid re-initialization. 

<Br>

BUT still not a good solution ...

- fails to retain accuracy on **previously seen examples**,

  since the distn of the query pool may drastically change with each round.

  ( = catastrophic foregtting )

$\rightarrow$ ***Continual Learning***

<br>

### Continual Active Learning (CAL)

***Applies CL strategies to accelerate batch Active Learning***

- apply CL to enable the model ...

  - (1) to learn the **NEWLY** labeled points, 
  - (2) without forgetting **PREVIOUSLY** labeled points

  while using past samples efficiently, using **replay-based** methods. 

- results) attains significant training time **speedups** 

<br>

Beneficial for the following reasons

- (1) **Cost save**
- (2) Makes AL more accessible for **edge computing**
- (3) **Agnostic** to the AL algorithm and the neural architecture

<br>

### Contribution

- **CAL framework**
  
  - demonstrate that **batch active learning** techniques can benefit from **CL**
  
- Study **speedup/performance trade-offs** on datasets 
  
  that vary in modality & architecture & data scale
  
  - modality : natural language, vision, medical imaging, and computational biology
  - neural architecture : Transformers/CNNs/MLPs
  
- Model with CAL & standard AL **behave similarly**

<br>

# 2. Related Work

## (1) Active learning 

**label efficiency** over passive learning

<br>

(1) Kirsch et al. (2019); Pinsler et al. (2019); Sener \& Savarese (2018) 

- reduce the number of query iterations **by having LARGE QUERY batch sizes**

- BUT do not exploit the learned models from previous rounds for the subsequent ones

  $\rightarrow$ complementary to CAL. 

<br>

(2) Coleman et al. (2020a); Ertekin et al. (2007); Mayer \& Timofte (2020); Zhu \& Bento (2017) 

- speeds up the selection of the new query set **by appropriately RESTRICTING the search space or by using GENERATIVE methods**

- can be easily integrated into our framework 

  ( CAL : works on the training side of active learning, not on the query selection )

<br>

(3) Lewis \& Catlett (1994); Coleman et al. (2020b); Yoo \& Kweon (2019) 

- use a **smaller PROXY model** to reduce computation overhead
- can be accelerated when integrated with CAL

<br>

## (2) Continual Learning & Transfer Learnring

(1) Perkonigg et al. (2021) 
- propose an approach that allows **AL to be applied to data streams of medical images** by introducing a module that **detects "domain shifts"**
- Distinct from our work CAL
  - CAL :uses $\mathrm{CL}$ algorithms to prevent catastrophic forgetting and to accelerate learning. 

<br>

(2) Zhou et al. (2021) 

- study when standard AL is used to fine-tune a pre-trained model, and **employs TL ( not CL )**

<br>

(3) Ayub \& Fendley (2022) 

- studies **where a robot observes unlabeled data sampled from a shifting distribution**, but does not explore active learning acceleration.

<br>

## (3) Preventing Catastrophic Foregtting

Focus of this paper = **Replay-based algorithms**

( But can also apply **other methods** such as EWC, Bayesian divergence priors, structural regularization, functional regularization ... )

<br>

## (4) Effect of warm-started model

If pretrained on a source dataset ...
- (1) converges ***faster*** 

- (2) but exhibits ***worse generalization on a target dataset*** 

  ( when compared to a randomly initialized model )

<br>

The above work only considers the setting, where the **source and target datasets are unbiased estimates of the SAME distribution**

$\leftrightarrow$ CAL : data distns are **all DEPENDENT on the model at each AL round**

& employs CL methods in addition to warm-starting

<br>

# 3. Background

## (1) Batch Active Learning

Notation

- $[n]=\{1, \ldots, n\}$, 
- $\mathcal{X}$ and $\mathcal{Y}$ : input and output domains

<br>

**Active Learning**

- starts with an unlabeled dataset $\mathcal{U}=\left\{x_i\right\}_{i \in[n]}$,

- allows the model $f$ ( with params $\theta$ ) to **query a user for labels** for any $x \in \mathcal{U}$,

  ***but with a limit of budget $b$, where $b \leq n$.*** 

- Setting : classification task
- Goal : ensure that $f$ can attain low error 
  - when trained only on the set of $b$ labeled points.

<br>

![image-20230717101431826](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717101431826.png).

Selection criteria 

- generally select samples based on ***model uncertainty*** and/or ***diversity***
- (this paper) **uncertainty sampling**

<br>

### Uncertainty Sampling 

Key Idea 

= selects $\mathcal{U}_t=\left\{x_1, \ldots, x_{b_t}\right\} \subseteq \mathcal{U}$ by choosing the samples that **maximize a notion of model uncertainty**

<br>

Measure = **entropy**

- If $h_\theta(x) \triangleq$ $-\sum_{i \in[k]} f(x ; \theta)_i \log f(x ; \theta)_i$, 

  then $\mathcal{U}_{t+1} \in \underset{\mathcal{A} \subset \mathcal{U}:|\mathcal{A}|=b_t}{\operatorname{argmax}} \sum_{x \in \mathcal{A}} h_{\theta_t}(x)$.

<br>

## (2) Continual Learning

![image-20230717102014996](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717102014996.png)

<br>

[ Dataset ]

$\mathcal{D}_{1: n}=\bigcup_{i \in[n]} \mathcal{D}_i$. 

- dataset consists of $T$ tasks $\left\{\mathcal{D}_1, \ldots, \mathcal{D}_T\right\}$ 

  that are presented to the model ***sequentially***

- $\mathcal{D}_t=\left\{\left(x_i, y_i\right)\right\}_{i \in N_t}$ 
  - $N_t$ are the task- $t$ sample indice & $n_t=\left|N_t\right|$. 

<br>

[ Procedure ( at time $t \in[T]$ )]

- step 1) **sample from the current task** $(x, y) \sim \mathcal{D}_t$,

  ( only limited access to the history $\mathcal{D}_{1: t-1}$. )

- step 2) CL objective

  - (1) efficiently **adapt** the model to $\mathcal{D}_t$ 

  - (2) while ensuring performance on the **history** does not appreciably degrade.

<br>

### Replay-based CL

- this paper : focus on replay-based CL techniques

- attempt to approximately solve $\mathrm{CL}$ optimization,

  **by using samples from $\mathcal{D}_{1: t-1}$ to regularize the model while adapting to $\mathcal{D}_t$**

- many CL consider problem of selecting ***"which samples should be retained"***

  ( this paper considers $\mathcal{H}=\mathcal{D}_{1: t-1}$.)

<br>

# 4. Blending Continual and Active Learning

AL inefficiency

= $f$ is ***retrained from scratch*** on the entire labeled pool after every query round. 

<br>

Potential solution

= **continue training** the model only on the newly AL-queried samples

( due to warm starting, hope that history will not fade )

<br>

However ...

( Warm Start )

![image-20230717102509895](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717102509895.png)



Results tells us that ... 

(1) the distribution of each AL-queried task $t>1$ is different than the data distribution

(2) fine-tuning to task $t$ can result in catastrophic forgetting

(3) techniques to combat catastrophic forgetting are necessary

<br>

### CAL (Continual Active Learning)

***uses CL techniques in AL to combat catastrophic forgetting***

![image-20230717102726971](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717102726971.png)

( key difference with CL vs CAL = blue line )

<br>

Speedup ( Replay-based CL vs. Standard )

- (1) computing gradient only on a **useful SUBSET of the history**

  ( not on **all reasonable choices** )

- (2) converges faster due to **warm start**

<br>

Notation

-  $\mathcal{L}_c(\theta) \triangleq$ $\mathbb{E}_{(x, y) \sim \mathcal{B}_{\text {current }}}[\ell(y, f(x ; \theta))]$. 

<br>

## (1) CAL + Experience Replay (CAL-ER)

Experience Replay

= Simplest and oldest replay-based method

<br>

Minibatch $\mathcal{B}$ of size $m+m^{(h)}$

- $\mathcal{B}_{\text {current }}$ and $\mathcal{B}_{\text {replay }}$ are interleaved 
- $\mathcal{B}_{\text {replay }}$ is chosen uniformly at random from $\mathcal{D}_{1: t-1}$. 

$\rightarrow$ model $f$ are updated based on $\mathcal{B}$.

<br>

## (2) CAL + Maximally Interfered Retrieval (CAL-MIR)

MIR

= chooses a size- $m^{(h)}$ subset of points from $\mathcal{D}_{1: t-1}$ ***"most likely to be forgotten"*** 

<br>

$\theta_v$ is computed by taking a **"virtual" gradient step**

$\theta_v=\theta-\eta \nabla \mathcal{L}_c(\theta)$ , using ... 

- (1) Batch of $m$ labeled samples $\mathcal{B}_{\text {current }}$ ( sampled from $\mathcal{D}_t$ )
- (2) Model parameters $\theta$

<br>

For every $x$ in history ... 

$s_{M I R}(x)=\ell(f(x ; \theta), y)-\ell\left(f\left(x ; \theta_v\right), y\right)$.

( = **change in loss after taking a single gradient step** ) 

<br>

Choosing samples 

= $m^{(h)}$ samples with the ***highest $s_{M I R}$ score*** are selected for $\mathcal{B}_{\text {replay }}$, 

<br>

$\mathcal{B}_{\text {current }}$ and $\mathcal{B}_{\text {replay }}$ are concatenated to form the minibatch ( same as CAL-ER )

<br>

( In practice ) Selection is done on a **random subset** of $\mathcal{D}_{1: t-1}$ for speed

( $\because$ computing $s_{M I R}$ for every historical sample is prohibitively expensive )

<br>



## (3) CAL + Dark Experience Replay (CAL-DER) 

***uses a DISTILLATION approach to regularize updates***

$g(x ; \theta)$ denote the pre-softmax logits of classifier $f(x ; \theta)$, 

( i.e., $f(x ; \theta)=$ $\operatorname{softmax}(g(x ; \theta))$. )

<br>

**Dark Experience Replay ( DER )**

- every $x^{\prime} \in \mathcal{D}_{1: t-1}$ has an associated $z^{\prime}$ 

  ( = model's logits at the end of the task )

- $x^{\prime} \in \mathcal{D}_{t^{\prime}}$, then $\left.z^{\prime} \triangleq g\left(x^{\prime} ; \theta_{t^{\prime}}^*\right)\right)$ where $t^{\prime} \in[t-1]$ and $\theta_{t^{\prime}}^*$ are the parameters obtained after round $t^{\prime}$. 
- minimizes $\mathcal{L}_{\text {DER }}(\theta)$ ..
  - $\mathcal{L}_{\mathrm{DER}}(\theta) \triangleq \mathcal{L}_c(\theta)+\underset{\left(x^{\prime}, y^{\prime}, z^{\prime}\right) \sim \mathcal{B}_{\text {replay }}}{\mathbb{E}}\left[\alpha\left\|g\left(x^{\prime} ; \theta\right)-z^{\prime}\right\|_2^2+\beta \ell\left(y^{\prime}, f\left(x^{\prime} ; \theta\right)\right)\right]$.
    - $\mathcal{B}_{\text {replay }}$ :  batch uniformly at randomly sampled from $\mathcal{D}_{1: t-1}$
  - Term 1) **CURRENT** task
  - Term 2) **REGULARIZER** based on **PREVIOUS** tasks
    - consists of a CE loss & MSE based distillation loss applied to historical samples.

<br>

## (4) CAL + Scaled Distillation (CAL-SD) 

$\mathcal{L}_{\mathrm{SD}}(\theta) \triangleq \lambda_t \mathcal{L}_c(\theta)+\left(1-\lambda_t\right) \mathcal{L}_{\text {replay }}(\theta)$.

- $\mathcal{L}_{\text {replay }}(\theta) \triangleq \underset{\left(x^{\prime}, y^{\prime}, z^{\prime}\right) \sim \mathcal{B}_{\text {replay }}}{\mathbb{E}}\left[\alpha D_{\mathrm{KL}}\left(\operatorname{softmax}\left(z^{\prime}\right) \| f\left(x^{\prime} ; \theta\right)\right)+(1-\alpha) \ell\left(y^{\prime}, f\left(x^{\prime} ; \theta\right)\right)\right]$.

- $\lambda_t \triangleq\left|\mathcal{D}_t\right| /\left(\left|\mathcal{D}_t\right|+\left|D_{1: t-1}\right|\right)$.

<br>

Similar to CAL-DER, $\mathcal{L}_{\text {replay }}$ is a sum of two terms

- (1) Distillation loss	
  - KL-divergence between ...
    - a) Posterior probabilities produced by $f$ 
    - b) $\operatorname{softmax}\left(z^{\prime}\right)$
  - use KL-divergence instead of MSE loss on the logits so that the distillation and the classification losses have the same scale and dynamic range
- (2) Classification loss

<br>

Weight of two loss (1) & (2) term?

$\rightarrow$ determined adaptively by a **"stability/plasticity"** trade-off term $\lambda_t$. 

- stable = effectively retain past information
- plastic = quickly learn new tasks

<br>

CAL : want the model to be plastic early on, and stable later on

$\rightarrow$ Apply it on $\lambda_t$ !!!

( HIGH value = HIGH plasticity )

***Since $\mathcal{D}_{1: t-1}$ increases with $t, \lambda_t$ decreases and the model becomes more stable in later training rounds.***

<br>

## (5)  Scaled Distillation w/ Submodular Sampling (CAL-SDS2) 

Two ideas

- (1) Uses CAL-SD to **regularize** the model 
- (2) Uses a **submodular sampling** procedure 
  - to select a **diverse set** of history points to replay. 

<br>

Submodular functions 

- capture notions of **"diversity"** and **"representativeness"** 

<br>

Define **submodular function** $G$ as..

$G(\mathcal{S}) \triangleq \sum_{x_i \in \mathcal{A}} \max _{x_j \in \mathcal{S}} w_{i j}+\lambda \log \left(1+\sum_{x_i \in \mathcal{S}} h\left(x_i\right)\right)$.

-  $w_{i j}=\exp \left(-\left\|z_i-z_j\right\|^2 / 2 \sigma^2\right)$ .

  ( = similarity score samples $x_i$ and $x_j$ )

  - $z_i$ : penultimate layer of model $f$ for $x_i$ 

<br>

Interpretation

- first term ) **facilitly localization function**
- second term ) concave over modular function
  - $h\left(x_i\right)$ : standard AL measure of model uncertainty
    - Ex) entropy of the model's output distribution. 

$\rightarrow$ Both terms are well known to be monotone non-decreasing submodular, as is their non-negatively weighted sum. 

<br>

In order to speed up SDS2 , 

we randomly subsample from the history before performing submodular maximization so $\mathcal{S} \subset \mathcal{A} \subset \mathcal{D}_{1: t-1}$. 

Goal : ensure that the set of samples that are replayed are both **difficult & diverse**

<br>

# 5. Experiments and Results

Settings :

- different fractions $(b / n)$ of the full dataset. 

<br>

Speedup attained by a CAL method 

= wall-clock time of the baseline AL method / the wall-clock time of the CAL method. 

<br>

Variety of different datasets spanning multiple modalities

<br>

Two baselines ( do not utilize CAL )

- (1) Standard $A L$ ( active learning )
- (2) AL with WS ( active Learning with warm starting )
  - but still training using all the presently available labeled data

<br>

Objective : to demonstrate ...

- (1) At least one CAL-based method exists that can **match or outperform** a standard AL technique while **achieving a significant speedup** for every budget and dataset
- (2) Models that have been trained using a CAL-based method **behave no differently than standard models** 
- (3) Point out that some of the **CAL methods are ablations of each other**
  - ex) CAL-ER is ablation for CAL-DER (or CAL-SD) 
    - when we replace the distillation component. 
  - ex) CAL-SD is ablation of CAL-SDS
    - where we remove the submodular selection part.

<br>

## (1) Datasets and Experimental Setup

Datasets ( various data modalities & scale & class imbalance )

- FMMNIST
- CIFAR-10
- MedMNIST
- Amazon Polarity Review
- COLA
- Single-Cell Cell Type Identity Classification

<br>

Active Learning setup

- Budgets : 
  - (others) 10%, 20%, 30%, 40%, 50%
  - (FMNIST, MedMNIST, Cell-type datasets) 10%, 15%, 20%, 25%, 30%
  - (COLA) 200, 400, 600, 800, 1000
- present the results for an **uncertainty sampling-based acquisition function**

<br>

## (2) Performance vs Speedup.

***Relative gain*** in accuracy over the AL baseline

( baseline accuraxcy = 1 )

- \> 1 : better than baseline
- \< 1 : worse than baseline

<br>

Results 

( Common : TOP & RIGHT are preferable )

- ( Figure 4 ) 
  - budgets fixed to $10 \%, 20 \%$, and $30 \%$ 
  - averaging over the datasets 
    - except COLA ... since it has a different budget
- ( Figure 3 ) 
  - dataset fixed
  - averaging the relative accuracy vs. speedups across different budgets
- ( Figure 1 )
  - averaging the above across different datasets

<br>

![image-20230717154911024](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717154911024.png)

<br>

![image-20230717155607742](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717155607742.png)

.

<img src="/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717155633351.png" alt="image-20230717155633351" style="zoom:50%;" />.

<br>

Summary

1. ( 1& 3 ) CAL method that attains a significant speedup over a standard $A L$ 

   - for every dataset and budget
   -  while preserving test set accuracy

2. ( 3 ) Some datasets only incur a minor drop in performance 

   - but attain the highest speedup

     $\rightarrow$ naively biasing learning towards recent tasks can be sufficient to adapt the model to a new set of points between AL rounds. 

3. ( 4 ) Not universally true for all the datasets (at different budgets). 
   - Hence, the methods which include some type of distillation term (CAL-DER, CAL-SD, CAL-SDS2) generally perform the best out of all CAL methods. 
   - We believe that the submodular sampling-based method (CAL-SDS2) can be accelerated using stochastic methods and results improved by considering other submodular functions, which we leave as future work

<br>

## (3) Comparison btw Standard & CAL methods

Whether CAL training has any adverse effect on the final model's behavior.

1. CAL does not result in any deterioration of model robustness 
2. CAL models and baseline trained models are uncertain about a similar set of unseen examples
3. Sensitivity analysis of our proposed methods
   - CAL methods are robust to the changes to the hyperparameters.

<br>

### a) Robustness

To ensure that models can generalize across different domains. 

<br>

Dataset : CIFAR-10C

- a dataset comprising 19 different corruptions each done at 5 levels of severity. 

<br>

![image-20230717160353057](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717160353057.png)

<br>

Settings

- $50 \%$ budget
- average classification accuracy over each corruption

<br>

### b) Correlation of Uncertainty Scores

To be valid ... standard AL & CAL ***need to query similar samples*** at each AL round

Metric : Pearson correlation 

- between the entropy scores of baseline and CAL models on the validation set after every query round

![image-20230717160545033](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717160545033.png)

<br>

# 6. Conclusion

CAL framework

- No need to  retrain models between batch $\mathrm{AL}$ rounds 
- There is always a CAL-based method that either matches or outperforms standard AL while achieving considerable speedups. ( in various domain datasets )
- Independent of the model architecture and AL strategy
