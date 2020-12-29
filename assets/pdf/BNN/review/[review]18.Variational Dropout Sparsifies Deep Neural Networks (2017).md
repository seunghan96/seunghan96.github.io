## [ Paper review 18 ]

# Variational Dropout Sparsifies Deep Neural Networks

### ( Dmitry Molchannov, et al., 2017 )



## [ Contents ]

0. Abstract
1. Introduction
2. Related Research
3. Dropout as Bayesian Approximation
4. Obtaining Model Uncertainty



# 0. Abstract

Extend Variational Dropout to the case when dropout rates are unbounded

Propose a way to reduce the variance of the gradient estimator



# 1. Introduction

Dropout

- Binary Dropout (Hinton et al., 2012)

- Gaussian Dropout (Srivastava et al, 2014)

  ( multiplies the outputs of the neurons by Gaussian random noise )

- Dropout rates are usually optimized by grid-search

  ( To avoid exponential complexity, dropout rates are usually shared for all layers)

- can be seen as a Bayesian regularization (Gal & Ghahramani, 2015)



Instead of injecting noise,  Sparsity!

- inducing sparsity during training DNN leads regularization (Han et al., 2015a)

- Sparse Bayesian Learning (Tipping, 2001)

  ( provies framework for training of sparse models )



This paper

- 1) study Variational Dropout (Kingma et al, 2015)

  where each weight of a model has its own individual dropout rate

- 2) propose Sparse Variational Dropout

  "extends VD" to all possible values of drop out rates ( = $\alpha$ )

  ( to do this, provide a new approximation of KL-divergence term in VD objective )

- 3) propose a way to reduce variance of stochastic gradient estimator 

  $\rightarrow$ leads to faster convergence



# 2. Related Work

너무 많아..생략

논문 참조



# 3. Preliminaries

## 3.1 Bayesian Inference

HOW to minimize $D_{K L}\left(q_{\phi}(w) \| p(w \mid \mathcal{D})\right) $  ?



Maximize ELBO = (1) Expected Log-likelihood -  (2) KL-divergence

ELBO : $\mathcal{L}(\phi)=L_{\mathcal{D}}(\phi)-D_{K L}\left(q_{\phi}(w) \| p(w)\right) \rightarrow \max _{\phi \in \Phi}$

- (1) Expected Log-likelihood : $L_{\mathcal{D}}(\phi)=\sum_{n=1}^{N} \mathbb{E}_{q_{\phi}(w)}\left[\log p\left(y_{n} \mid x_{n}, w\right)\right]$
- (2) KL-divergence : $D_{K L}\left(q_{\phi}(w) \| p(w)\right)$



## 3.2 Stochastic Variational Inference

### (a) Reparameterization  Trick (Kingma \& Welling, 2013)

- obtain unbiased differentiable minibatch-based MC estimator of expected log-likelihood

  ( that is, find $\nabla_{\phi} L_{\mathcal{D}}\left(q_{\phi}\right) $  )

- trick : decompose into (1) deterministic \& (2) stochastic part

  $w=f(\phi, \epsilon)$ where $\epsilon \sim p(\epsilon)$

- number of data in one mini-batch : $M$

  $\mathcal{L}(\phi) \simeq \mathcal{L}^{S G V B}(\phi)=L_{\mathcal{D}}^{S G V B}(\phi)-D_{K L}\left(q_{\phi}(w) \| p(w)\right)$

  $L_{\mathcal{D}}(\phi) \simeq L_{\mathcal{D}}^{S G V B}(\phi)=\frac{N}{M} \sum_{m=1}^{M} \log p\left(\tilde{y}_{m} \mid \tilde{x}_{m}, f\left(\phi, \epsilon_{m}\right)\right)$

  $\nabla_{\phi} L_{\mathcal{D}}(\phi) \simeq \frac{N}{M} \sum_{m=1}^{M} \nabla_{\phi} \log p\left(\tilde{y}_{m} \mid \tilde{x}_{m}, f\left(\phi, \epsilon_{m}\right)\right)$



### (b) Local Reparameterization  Trick (Kingma et al., 2015)

- sample separate weight matrices for each data-point inside mini-batch
- done efficiently by moving the noise from "weights" to  "activation"



## 3.3 Variational Dropout

$B=(A \odot \Xi) W, \text { with } \xi_{m i} \sim p(\xi)$  .... putting noise on INPUT



Bernoulli(Binary) Dropout 

- Hinton et al., 2012

-  $\xi_{m i} \sim$ Bernoulli $(1-p)$ 



Gaussian Dropout with continuous noise

- Srivastava et al, 2014

- $\xi_{m i} \sim \mathcal{N}(1, \alpha= \frac{p}{1-p})$

- continuous noise is better than discrete noise 

  ( multiplying the inputs by Gaussian noise = putting Gaussian noise on the weights )

- can be used to obtain posterior distribution over model's weight! (Wang \& Manning, 2013), (Kingma et al., 2015)

  ( $\xi_{i j} \sim \mathcal{N}(1, \alpha)$ = sampling $w_{ij}$ from $q\left(w_{i j} \mid \theta_{i j}, \alpha\right)=\mathcal{N}\left(w_{i j} \mid \theta_{i j}, \alpha \theta_{i j}^{2}\right) .$ )

  ( Then, $\begin{array}{c}
  w_{i j}=\theta_{i j} \xi_{i j}=\theta_{i j}\left(1+\sqrt{\alpha} \epsilon_{i j}\right) \sim \mathcal{N}\left(w_{i j} \mid \theta_{i j}, \alpha \theta_{i j}^{2}\right) \;\;\; \text{where} \;\;\;
  \epsilon_{i j} \sim \mathcal{N}(0,1)
  \end{array}$ )



Variational Dropout

- ( use reparam trick + draw single sample $W \sim q(W \mid \theta, \alpha)$ )

  $\rightarrow$ Gaussian dropout = stochastic optimization of exxpected log likelihood

- VD extends this technique!

  use $q(W \mid \theta, \alpha)$ as an approximate posterior with special prior, $p\left(\log \left|w_{i j}\right|\right)=\mathrm{const} \Leftrightarrow p\left(\left|w_{i j}\right|\right) \propto \frac{1}{\left|w_{i j}\right|}$

  

GD Training = VD Training ( when $\alpha$ is fixed )

However, VD provides a way to train dropout rate $\alpha$ by optimizing the ELBO



# 4. Sparse Variational Dropout

difficulties in training the model with large values of $\alpha$

$\rightarrow$ have considered the case of $\alpha \leq 1$ ( $\leftrightarrow$  $p\leq 0.5$ in binary dropout )



High dropout rate $\alpha_{ij} \rightarrow + \infty$ = $p=1$

( meaning : corresponding weight is always ignored \& can be removed )



## 4.1 Additive Noise Reparameterization

$$
\frac{\partial \mathcal{L}^{S G V B}}{\partial \theta_{i j}}=\frac{\partial \mathcal{L}^{S G V B}}{\partial w_{i j}} \cdot \frac{\partial w_{i j}}{\partial \theta_{i j}}=(1) \times (2)
$$
(2) is very noisy if $\alpha_{i j}$ is large.

$w_{i j}=\theta_{i j}\left(1+\sqrt{\alpha_{i j}} \cdot \epsilon_{i j}\right) $

$\frac{\partial w_{i j}}{\partial \theta_{i j}}=1+\sqrt{\alpha_{i j}} \cdot \epsilon_{i j} $, where $\epsilon_{i j} \sim \mathcal{N}(0,1)$



How to reduce variance when $\alpha_{i j}$ is large ?

replace multiplicative noise term $1+\sqrt{\alpha_{i j}} \cdot \epsilon_{i j}$ .... with $\sigma_{i j} \cdot \epsilon_{i j},$

( where $\sigma_{i j}^{2}=\alpha_{i j} \theta_{i j}^{2}$ )

$\begin{aligned}
w_{i j}=& \theta_{i j}\left(1+\sqrt{\alpha_{i j}} \cdot \epsilon_{i j}\right)\\&=\theta_{i j}+\sigma_{i j} \cdot \epsilon_{i j} 
\end{aligned}$

Thus, $\frac{\partial w_{i j}}{\partial \theta_{i j}}=1, \quad \epsilon_{i j} \sim \mathcal{N}(0,1)$
( has no injection noise! )





avoid the problem of large gradient variance!

can train the model within the full range of $\alpha_{i j} \in(0,+\infty)$



## 4.2. Approximation of the KL Divergence

full KL-divergence term in ELBO

$D_{K L}(q(W \mid \theta, \alpha) \| p(W))= \sum_{i j} D_{K L}\left(q\left(w_{i j} \mid \theta_{i j}, \alpha_{i j}\right) \| p\left(w_{i j}\right)\right)$



log-scale uniform prior distribution is an improper prior

$-D_{K L}\left(q\left(w_{i j} \mid \theta_{i j}, \alpha_{i j}\right) \| p\left(w_{i j}\right)\right)
= \frac{1}{2} \log \alpha_{i j}-\mathbb{E}_{\epsilon \sim \mathcal{N}\left(1, \alpha_{i j}\right)} $

Term above is intractable in VD

need to be sampled \& approximated

$\begin{array}{c}
-D_{K L}\left(q\left(w_{i j} \mid \theta_{i j}, \alpha_{i j}\right) \| p\left(w_{i j}\right)\right) \approx 
\left.\approx k_{1} \sigma\left(k_{2}+k_{3} \log \alpha_{i j}\right)\right)-0.5 \log \left(1+\alpha_{i j}^{-1}\right)+\mathrm{C} \\
k_{1}=0.63576 \quad k_{2}=1.87320 \quad k_{3}=1.48695
\end{array}$

