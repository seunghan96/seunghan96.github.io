---
title: Summary of Bayesian DL papers (21~30)
categories: [BNN]
tags: [Bayesian Machine Learning, Bayesian Deep Learning, Probabilistic Deep Learning, Uncertainty Estimation, Variational Inference]
excerpt: Bayesian ML/DL, Probabilistic DL, Uncertainty Estimation
---

# Summary of Bayesian DL papers [21~30]

I have summarized the **must read + advanced papers** of papers regarding....

- various methods using Variational Inference

- Bayesian Neural Network

- Probabilistic Deep Learning

- Uncertainty Estimation

  

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



## 21.Variational Inference using Implicit Distributions 

### ( Ferenc Huszar , 2017 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/21.Variational Inference using Implicit Distributions (2017).pdf' | /assets/pdf/BNN/paper/21.Variational Inference using Implicit Distributions (2017).pdf }}) )

Variational Inference = use $$q$$ to approximate $$p$$

- ex) MFVI : simple, fast but may be inaccurate!

Key Point : Expand the Variational Family $$q_{\theta}(z)$$

<br>

Implicit distribution : (1) easy to sample (2) hard to evaluate

$$\rightarrow$$ can make more expressive distribution :)

<br>

but, using Implicit Distribution is hard in Variational Inference

( $$\because$$ entropy term in ELBO is intractable )

$$\rightarrow$$ thus, use "Density Ratio Estimation"

<br>

Density Ratio Estimation

- by training a classifier $$D(z)$$
  - $$y=1$$ : sample from $$q_{\theta}(z)$$
  - $$y=0$$ : sample from $$p(z)$$
- Algorithm summary
  - ELBO : $$ \mathbb{E}_{q_{\theta}(z)}[\log p(x \mid z)]-\mathbb{E}_{q_{\theta}(z)}[\log D(z)-\log (1-D(z))]$$
  - step 1) follow gradient estimate of the ELBO w.r.t $$\theta$$ ( with reparam trick )
  - step 2) for each $$\theta,$$ fit $$D(z)$$ so that $$D(z) \approx D^{*}(z)$$

- limitations : 
  - unstable learning when discriminator does not catch up
  - overfits in high dimension 

summary : [Download]({{ '/assets/pdf/BNN/review/[review]21.Variational Inference using Implicit Distributions (2017).pdf' | /assets/pdf/BNN/review/[review]21.Variational Inference using Implicit Distributions (2017).pdf }})

<br>

<br>

## 22. Semi-Implicit Variational Inference 

### ( M Yin, 2018 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/22.Semi-Implicit Variational Inference (2018).pdf' | /assets/pdf/BNN/paper/22.Semi-Implicit Variational Inference (2018).pdf }}) )

using Implicit Distribution is hard in Variational Inference

( $$\because$$ entropy term in ELBO is intractable )

<br>

instead of Density Ratio Estimation, use

$$\rightarrow$$ "Semi-implicit Distribution" (SIVI)

<br>

SIVI : "optimize LOWER BOUND of ELBO"

summary : [Download]({{ '/assets/pdf/BNN/review/[review]22.Semi-Implicit Variational Inference (2018).pdf' | /assets/pdf/BNN/review/[review]22.Semi-Implicit Variational Inference (2018).pdf }})

<br>

<br>

## 23.Unbiased Implicit Variational Inference 

### ( Michalis K. Titsias, Francisco J. R. Ruiz, 2019 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/23.Unbiased Implicit Variational Inference (2019).pdf' | /assets/pdf/BNN/paper/23.Unbiased Implicit Variational Inference (2019).pdf }}) )

using Implicit Distribution is hard in Variational Inference

( $$\because$$ entropy term in ELBO is intractable )

<br>

instead of Density Ratio Estimation, use

$$\rightarrow$$ "Unbiased-implicit Distribution" (UIVI)

<br>

UIVI : "DIRECTLY optimize ELBO"

( better performance than SIVI )

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]23.Unbiased Implicit Variational Inference (2019).pdf' | /assets/pdf/BNN/review/[review]23.Unbiased Implicit Variational Inference (2019).pdf }})

<br>

<br>

## 24.A Contrastive Divergence for Combining Variational Inference and MCMC

### ( Francisco J. R. Ruiz 1 2 Michalis K. Titsias, 2019 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/24.A Contrastive Divergence for Combining Variational Inference and MCMC (2019).pdf' | /assets/pdf/BNN/paper/24.A Contrastive Divergence for Combining Variational Inference and MCMC (2019).pdf }}) )

challenges of MCMC in VI

- 1) intractable
- 2) objective depend weakly on $$\theta$$

$$\rightarrow$$ use an alternative divergence, "Variational Contrastive Divergence" (VCD)

<br>

$$\mathcal{L}_{\mathrm{VCD}}(\theta)=\underbrace{\mathrm{KL}\left(q_{\theta}^{(0)}(z) \| p(z \mid x)\right)-\mathrm{KL}\left(q_{\theta}(z) \| p(z \mid x)\right)}_{\geq 0}+\underbrace{\mathrm{KL}\left(q_{\theta}(z) \| q_{\theta}^{(0)}(z)\right)}_{\geq 0}$$.

can be also written as...

$$\mathcal{L}_{\mathrm{VCD}}(\theta)=-\mathbb{E}_{q_{\theta}^{(0)}(z)}\left[\log p(x, z)-\log q_{\theta}^{(0)}(z)\right]+\mathbb{E}_{q_{\theta}(z)}\left[\log p(x, z)-\log q_{\theta}^{(0)}(z)\right]$$.

<br>

problem #1 ) (intractability)

- solution: $$\log q_{\theta}^{(0)}(z)$$ cancels out 

problem #2 ) (weak dependence)

-  solution : $$\mathcal{L}_{\mathrm{VCD}}(\theta) \stackrel{t \rightarrow \infty}{\longrightarrow} \mathrm{KL}\left(q_{\theta}^{(0)}(z) \| p(z \mid x)\right)+\mathrm{KL}\left(p(z \mid x) \| q_{\theta}^{(0)}(z)\right)$$

<br>

Steps

- 1) Sample $$z_{0} \sim q_{\theta}^{(0)}(z)$$ (reparameterization)

- 2) Sample $$z \sim Q^{(t)}\left(z \mid z_{0}\right)$$ (run $$t$$ MCMC steps)

- 3) Estimate the gradient $$\nabla_{\theta} \mathcal{L}_{\mathrm{VCD}}(\theta)$$

- 4) Take gradient step w.r.t. $$\theta$$

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]24.A Contrastive Divergence for Combining Variational Inference and MCMC.pdf' | /assets/pdf/BNN/review/[review]24.A Contrastive Divergence for Combining Variational Inference and MCMC.pdf }})

<br>

<br>

## 25.Non-linear Independent Components Estimation (NICE)

### ( Laurent Dinh, et al, 2014 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/25.Non-linear Independent Components Estimation (NICE) (2014).pdf' | /assets/pdf/BNN/paper/25.Non-linear Independent Components Estimation (NICE) (2014).pdf }}) )

( need to know about "Variable Transformation & determinant of Jacobian" )

contribution of NICE :

- 1) computing the determinant of Jacobian & inverse Jacobian is trivial
- 2) still learn complex non-linear transformations ( with composition of simple blocks )

<br>

Coupling layer

- (1) bijective transformation 
  (2) triangular Jacobian ( makes it tractable!)

- additive coupling layer

  combining coupling layer

- allows "Rescaling"

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]25.Non-linear Independent Components Estimation (NICE) (2014).pdf' | /assets/pdf/BNN/review/[review]25.Non-linear Independent Components Estimation (NICE) (2014).pdf }})

<br>

<br>

## 26.Variational Inference with Normalizing Flows 

### ( Danilo Jimenez Rezende, Shakir Mohamed, 2016 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/26.Variational Inference with Normalizing Flows (2016).pdf' | /assets/pdf/BNN/paper/26.Variational Inference with Normalizing Flows (2016).pdf }}) )

limitations of variational methods : choice of posterior approximation are often limited

$$\rightarrow$$ richer approximation is needed!

<br>

"Amortized Variational Inference" = (1) + (2)

- (1) MC gradient estimation
- (2) Inference network

<br>

For successful VI...need 2 requirements

- 1) efficient computation of derivatives of expected log-liklihood in ELBO

  $$\rightarrow$$ by Amortized Variational Inference

- 2) rich approximating distribution

  $$\rightarrow$$ by "NORMALIZING FLOW"

<br>

Formula of NF ( Successive application )

- $$\mathbf{z}_{K}=f_{K} \circ \ldots \circ f_{2} \circ f_{1}\left(\mathbf{z}_{0}\right)$$

- $$\ln q_{K}\left(\mathbf{z}_{K}\right)=\ln q_{0}\left(\mathbf{z}_{0}\right)-\sum_{k=1}^{K} \ln \left|\operatorname{det} \frac{\partial f_{k}}{\partial \mathbf{z}_{k-1}}\right|$$

- $$\mathbb{E}_{q_{K}}[h(\mathbf{z})]=\mathbb{E}_{q_{0}}\left[h\left(f_{K} \circ f_{K-1} \circ \ldots \circ f_{1}\left(\mathbf{z}_{0}\right)\right)\right]$$

  $$\rightarrow$$ does not depend on $$q_{k}$$

<br>

for successful NF, we must

- 1) specify a class of invertible transformations
- 2) efficient mechanism for computing the determinant of Jacobian

$$\rightarrow$$ should have low-cost computation of the determinant ( or where Jacobian is not needed )

<br>

Invertible Linear-time Transformations

- some types of flows can be invertible + calculated in linear time
- ex) Planar Flows, Radial Flows 

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]26.Variational Inference with Normalizing Flows (2016).pdf' | /assets/pdf/BNN/review/[review]26.Variational Inference with Normalizing Flows (2016).pdf }})

<br>

<br>

## 27.Density Estimation using Real NVP

### ( Laurent Dinh, et al., 2017 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/27.Density Estimation using Real NVP (2017).pdf' | /assets/pdf/BNN/paper/27.Density Estimation using Real NVP (2017).pdf }}) )

real NVP 

- real-valued non-volume preserving transformation
- "Powerful, Invertible, Learnable" transformation

- tractable yet expressive approach to model high-dimensional data!

<br>

bijection : Coupling Layer

- $$y_{1: d} =x_{1: d}$$

  $$y_{d+1: D} =x_{d+1: D} \odot \exp \left(s\left(x_{1: d}\right)\right)+t\left(x_{1: d}\right)$$

- 1) easy calculation of Jacobian

  2) invertible

<br>

Masked convolution  ( with binary mask)

Combining coupling layers

Batch Normalizations

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]27.Density Estimation using Real NVP (2017).pdf' | /assets/pdf/BNN/review/[review]27.Density Estimation using Real NVP (2017).pdf }})

<br>

<br>

## 28.Glow_Generative Flow with Invertible 1x1 Convolutions 

### ( Diederik P. Kingma, Prafulla Dhariwal, 2018 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/28.Glow_Generative Flow with Invertible 1x1 Convolutions (2018).pdf' | /assets/pdf/BNN/paper/28.Glow_Generative Flow with Invertible 1x1 Convolutions (2018).pdf }}) )

Glow

- simple type of generative flow, using "invertible 1 x 1 convolution"
- significant improvement in log-likelihood on standard benchmarks

<br>

Generative Modeling have advanced with likelihood-based methods
Likelihood-based methods : three categories

- 1) Autoregressive models
- 2) VAEs
- 3) Flow-based generative models ( ex. NICE, RealNVP )

<br>

Proposed Generative Flow

- built on NICE and RealNVP
- consists of a series of steps of flows
- combined with multi-scale architecture

<br>

Architecture

- 1) actnorm (using scale & bias)
- 2) Invertible 1 x 1 convolution
- 3) Affine Coupling Layers

<br>

summary : [Download]({{ '/assets/pdf/BNN/review/[review]28.Glow ; Generative Flow with Invertible 1x1 Convolutions (2018).pdf' | /assets/pdf/BNN/review/[review]28.Glow ; Generative Flow with Invertible 1x1 Convolutions (2018).pdf }})

<br>

<br>

## 29.What Uncertainties Do We Need in Bayesian Deep Learning

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/29.What Uncertainties Do We Need in Bayesian Deep Learning(2017).pdf' | /assets/pdf/BNN/paper/29.What Uncertainties Do We Need in Bayesian Deep Learning(2017).pdf }}) )

<br>

<br>

## 30.Uncertainty quantification using Bayesian neural networks in classification_Application to ischemic stroke lesion segmentation

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/30.Uncertainty quantification using Bayesian neural networks in classification_Application to ischemic stroke lesion segmentation (2018).pdf' | /assets/pdf/BNN/paper/30.Uncertainty quantification using Bayesian neural networks in classification_Application to ischemic stroke lesion segmentation (2018).pdf }}) )