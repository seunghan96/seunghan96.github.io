---
ㄴtitle: \[interpretable\] (paper 4) Uncertainty-Aware Attention for Reliable Interpretation and Prediction
categories: [INTE,STUDY]
tags: [Interpretable Learning]
excerpt: Attention mechanism
---

# Uncertainty-Aware Attention for Reliable Interpretation and Prediction

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Contents

0. Abstract
2. Introduction
2. Approach 
   1. Stochastic Attention with input-adaptive Gaussian Noise
   2. Variational Inference

<br>

# 0. Abstract

Attention mechanism은 relevant feature에 집중할 수 있게끔 하는 엄청난 기능을 가지지만....

$$\rightarrow$$ may be **UNRELIABLE**

<br>

이를 극복하기 위해....***Input-dependent uncertainty***라는 개념을 소개함

( = generates attention for each feature with varying degrees of **noise based on the given input** )

$$\rightarrow$$ learn **larger variance** on instances, with high uncertainty

<br>

propse **UA ( Uncertainty-aware Attention )** mechanism using VI

<br>

# 1. Introduction

## Background

- **high reliability**를 얻는 것은 매우 중요해! 특히 safety 관련해서!

- Attention : find most **relevant features** for each input instance

  ( + allows **easy interpretation** )

- Attention의 한계점 : ***unreliable***

  $$\rightarrow$$ need a model that knows its own limitation!

  ( = 예측/판단을 내려도 될 정도로 safe한지를 모델 지 스스로 잘 알아야! )

<br>

## Proposal

allow attention to **output uncertainty on each input**

- 더 나아가서, leverage them when making final predictions

- 구체적으로, attention weight를 Gaussian으로 모델링 ( input dependent noise O )

<br>

## Contribution

- 1) Variational Attention 모델을 제안함
- 2) UA알고리즘이 accurate calibration of model uncertainty를 만듬
- 3) 6개의 real world 데이터에 테스스

<br>

# 2. Approach 

**STOCHASTIC attention** ( 최초 제안은 X )

- $$\mathbf{v}(\mathbf{x}) \in \mathbb{R}^{r \times i}$$ : concatenation of $$i$$ intermediate features
  -  each column of which $$\mathbf{v}_{j}(\mathbf{x})$$ is a length $$r$$ vector
  - 이 $$\mathbf{v}(\mathbf{x})$$ 로부터 random variables $$\left\{\mathbf{a}_{j}\right\}_{j=1}^{i}$$ 가 conditionally 생성된다
- $$\mathbf{c}(\mathbf{x})=\sum_{j=1}^{i} \mathbf{a}_{j} \odot \mathbf{v}_{j}(\mathbf{x})$$ : context vector ( $$\mathbf{c} \in \mathbb{R}^{r}$$ )
- $$\hat{\mathbf{y}}=f(\mathbf{c}(\mathbf{x}))$$ : final output

<br>

attention은 deterministic / stochastic 할 수 있다

- ex) stochastic attention :   $$\mathbf{a}_j$$ 는 **Bernoulli distribution에서 생성됨.**

  $$\rightarrow$$ maximize ELBO

- **stochastic attention > deterministic counterpart**, on image annotation task.

<br>

## 2-1. Stochastic Attention with input-adaptive Gaussian Noise

위의 Stochastic attention의 2가지 한계점

( stochastic attention을 directly하게 Bernoulli/Multinomial에서 뽑는다는 점에서 )

- **[ 한계점 1 ] Bernoulli의 variance는 allocation probability $$\mu$$와 DEPENDENT**

  - $$\mathbf{a} \sim \text{Bernoulli}(\mu)$$.
  - allocation probability $$\mu$$ : attention 연결 할지/말지
  - Bernoulli의 variance : $$\sigma^{2}=\mu(1-\mu)$$
  - 따라서, $$\mathbf{a}$$의 variance는 낮기 어렵다 ( if $$\mu$$가 0.5 부근 )

- **[ 해결책 1 ]**

  disentangle the **attention strength** a from the **attention uncertainty** 

  $$\rightarrow$$ so that the uncertainty could **vary** even with the same attention strength

<br>

- **[ 한계점 2 ] Vanilla stochastic attention models the noise independently of the input**

  - (구 방식) input과 무관하게 noise를 모델링함

- **[ 해결책 2 ]** 위의 두 한계점들을 극복하기 위해...

  "input과 **관련있게 noise를 모델링**함 ( $$\sigma(x)$$ )"

  - (구) $$p(\boldsymbol{\omega})=\mathcal{N}\left(\mathbf{0}, \tau^{-1} \mathbf{I}\right), \quad p_{\theta}(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\omega})=\mathcal{N}\left(\boldsymbol{\mu}(\mathbf{x}, \boldsymbol{\omega} ; \theta), \operatorname{diag}\left(\boldsymbol{\sigma}^{2}\right)\right)$$

  - (신) $$p(\boldsymbol{\omega})=\mathcal{N}\left(\mathbf{0}, \tau^{-1} \mathbf{I}\right), \quad p_{\theta}(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\omega})=\mathcal{N}\left(\boldsymbol{\mu}(\mathbf{x}, \boldsymbol{\omega} ; \theta), \operatorname{diag}\left(\boldsymbol{\sigma}^{2}(\mathbf{x}, \boldsymbol{\omega} ; \theta)\right)\right)$$

    ( 위 두 식에서, $$\mathbf{z}$$는 attention score before squashing... 즉 $$\mathrm{a}=\pi(\mathrm{z})$$ )

  - empirically shown that the **quality of uncertainty improves**

<br>

## 2-2. Variational Inference

( 위 2-1.에서 세운 방법을, VI를 통해 푼다 )

$$\mathbf{Z}$$ : set of latent variables $$\left\{\mathbf{z}^{(n)}\right\}_{n=1}^{N}$$ that stands for attention weight before squashing. 

Posterior $$p(\mathbf{Z}, \boldsymbol{\omega} \mid \mathcal{D})$$ is usually **computationally intractable** !

$$\rightarrow$$ use VI ( Variational Inference )

<br>

Variational Distribution : 

- $$q(\mathbf{Z}, \boldsymbol{\omega} \mid \mathcal{D})=q_{\mathbf{M}}(\boldsymbol{\omega} \mid \mathbf{X}, \mathbf{Y}) q(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\omega})$$.

- 1번째 term) MC Dropout 사용 ( variational parameter $$\mathbf{M}$$ )
- 2번째 term) 그냥 set $$q(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\omega})=p_{\theta}(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\omega})$$

<br>

이를 정리하면, 아래의 ELBO를 maximize하는 것과 동일하다.

- $$\begin{aligned}
  \log p(\mathbf{Y} \mid \mathbf{X}) \geq & \mathbb{E}_{\boldsymbol{\omega} \sim q_{\mathbf{M}}(\boldsymbol{\omega} \mid \mathbf{X}, \mathbf{Y}), \mathbf{Z} \sim p_{\theta}(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\omega})}[\log p(\mathbf{Y} \mid \mathbf{X}, \mathbf{Z}, \boldsymbol{\omega})] \\
  &-\mathrm{KL}\left[q_{\mathbf{M}}(\boldsymbol{\omega} \mid \mathbf{X}, \mathbf{Y}) \| p(\boldsymbol{\omega})\right]-\operatorname{KL}\left[q(\mathbf{Z} \mid \mathbf{X}, \mathbf{Y}, \boldsymbol{\omega}) \| p_{\theta}(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\omega})\right]
  \end{aligned}$$.

<br>

최종 maximize할 대상 : $$\mathcal{L}(\theta, \mathbf{M} ; \mathbf{X}, \mathbf{Y})=\sum \log p_{\theta}\left(\mathbf{y}^{(n)} \mid \tilde{\mathbf{z}}^{(n)}, \mathbf{x}^{(n)}\right)-\lambda\|\mathbf{M}\|^{2}$$.

- step 1) sample random weights with dropout masks $$\widetilde{\omega} \sim q_{\mathrm{M}}(\boldsymbol{\omega} \mid \mathbf{X}, \mathbf{Y})$$ 
- step 2) sample $$\mathbf{z}$$ such that $$\tilde{\mathbf{z}}=q(\mathbf{x}, \tilde{\varepsilon}, \widetilde{\omega}), \tilde{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

<br>

Testing new input $$\mathbf{x}^{*}$$ : MC-sampling 사용해서..

- $$p\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}\right)=\iint p\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}, \mathbf{z}\right) p\left(\mathbf{z} \mid \mathbf{x}^{*}, \boldsymbol{\omega}\right) p(\boldsymbol{\omega} \mid \mathbf{X}, \mathbf{Y}) \mathrm{d} \boldsymbol{\omega} \mathrm{d} \mathbf{z} \approx \frac{1}{S} \sum_{s=1}^{S} p\left(\mathbf{y}^{*} \mid \mathbf{x}^{*}, \tilde{\mathbf{z}}^{(s)}\right)$$/

<br>

### Uncertainty Calibration

ECE (Expected Calibration Error)가 보다 나음을 확인!

( = expected gap w.r.t the distribution of model confidence )

$$\mathrm{ECE}=\mathbb{E}_{\text {confidence }}[\mid p($$ correct $$\mid$$ confidence $$)-$$ confidence $$\mid]$$

![figure2](/assets/img/INTE/img8.png)