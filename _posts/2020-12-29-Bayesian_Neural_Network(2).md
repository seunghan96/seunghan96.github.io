---
title: Summary of Bayesian DL papers (2)
categories: [BNN]
tags: [Bayesian Machine Learning, Bayesian Deep Learning, Probabilistic Deep Learning, Uncertainty Estimation, Variational Inference]
excerpt: Bayesian ML/DL, Probabilistic DL, Uncertainty Estimation
---

# Summary of Bayesian DL papers [21~40]

I have summarized the **must read + advanced papers** of papers regarding....

- various methods using Variational Inference

- Bayesian Neural Network

- Probabilistic Deep Learning

- Uncertainty Estimation

  

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



#### 20. Uncertainty in Deep Learning (1)

#### Gal, Y. (2016)

( download paper here :  [Download]({{ '/assets/pdf/temp.pdf' | /assets/pdf/temp.pdf }}) )

- Importance of knowing what we don't know

- Model Uncertainty

  - 1) Aleatoric Uncertainty ( = Data Uncertainty )
    - noisy data
  - 2) Epistemic Uncertainty ( = Model Uncertainty )
    - uncertainty in model parameters
  - 3) Out of Distribution
    - the point lies outside the data distribution

- BNN (Bayesian Neural Network)

  - GP can be recovered with infinitely many weights
  - model uncertainty can be obtained by placing "distribution over weights"

- models which gives uncertainty usually do not scale well

  $\rightarrow$ need practical techniques (ex. SRT)

- SRT (Stochastic Regularization Techniques)

  - adapt the model output "stochastically" as a way of model regularization
  - predictive mean/variance \& random output

- summary : [Download]({{ '/assets/pdf/temp.pdf' | /assets/pdf/temp.pdf }})



#### 20. Uncertainty in Deep Learning (2)

- review of Variational Inference
  - ELBO = (1) + (2)
    - (1) encourage $q$ to explain the data well
    - (2) encourage $q$ to be close to the prior
  - replace "marginalization" to "optimization"
  - does not scale to large data \& complex models
- previous histories of BNN
- 3 Modern Approximate Inference
  - 1) Variational Inference ( above )
  - 2) Sampling based techniques ( HMC, Langevin method , SGLD ...)
  - 3) Ensemble methods ( produce point estimate many times )
- summary : [Download]({{ '/assets/pdf/temp.pdf' | /assets/pdf/temp.pdf }})



#### 20. Uncertainty in Deep Learning (3)

- Bayesian Deep Learning : based on 2 works

  - (1) MC estimation ( Graves, 2011 )
  - (2) VI ( Hinton and Van Camp, 1993 )

- BNN inference + SRTs

- Steps

  - [step 1] analyze variance of several stochastic estimators (used in VI)
  - [step 2] tie these derivations to SRTs
  - [step 3] propose practical techniques to obtain "model uncertainty"

- "expected log likelihood" in ELBO

  - problem 1) high computation 

    $\rightarrow$ solve by data sub-sampling (mini-batch optimization)

  - problem 2) intractable integral

    $\rightarrow$ MC integration

- MC estimators

  - 1) score function estimator ( = likelihood ratio estimator, Reinforce)
  - 2) path-wise derivative estimator ( = reparameterization trick )
  - 3) characteristic function estimator

- SRT (Stochastic Regularization Techniques)

  - regularize model through injection of stochastic noise
  - Dropout, multiplicative Gaussian Noise....

- alternative of Dropout...

  - 1) Additive Gaussian Noise
  - 2) Multiplicative Gaussian Noise
  - 3) Drop connect

  algorithm 1(VI) = algorithm 2(DO) ( KL Condition )

  - VI) minimize divergence 

  - DO) optimization of NN with Dropout

- Model uncertainty in BNN

- Uncertainty in Classification

  - variation ratios / predictive entropy / mutual information

- Bayesian CNN/RNN

- summary : [Download]({{ '/assets/pdf/temp.pdf' | /assets/pdf/temp.pdf }})





1. 

#### 21.Variational Inference using Implicit Distributions 



( download paper here :  [Download]({{ '/assets/pdf/temp.pdf' | /assets/pdf/temp.pdf }}) )

#### 22. Semi-Implicit Variational Inference 

23.Unbiased Implicit Variational Inference (2019)

24.A Contrastive Divergence for Combining Variational Inference and MCMC (2019)

25.Non-linear Independent Components Estimation (NICE) (2014)

26.Variational Inference with Normalizing Flows (2016)

27.Density Estimation using Real NVP (2017)

28.Glow_Generative Flow with Invertible 1x1 Convolutions (2018)

29.What Uncertainties Do We Need in Bayesian Deep Learning(2017)

30.Uncertainty quantification using Bayesian neural networks in classification_Application to ischemic stroke lesion segmentation (2018)

31.Uncertainty Estimations by Softplus normalization in Bayesian Convolutional Neural Networks with Variational Inference

32.MADE_Masked Autoencoder for Distribution Estimation (2015)

33.Functional Variational Bayesian Neural Networks

33.Improved Variational Inference with Inverse Autoregressive Flow (2016)

34.Function Space Particle Optimization For Bayesian Neural Networks

34.Masked Autoregressive Flow for Density Estimation (2017)

35.Gaussian Process Behaviour in Wide Deep Neural Networks

36.Practical Learning of Deep Gaussian Processes via Random Fourier Features

37.Mapping Gaussian Process Priors to Bayesian Neural Networks

38.VIME ; Variational Information Maximizing Exploration

39.Bayesian Optimization with Robust Bayesian Neural Networks

40.Bayesian GAN

41.Learning And Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks

42.Model Selection in Bayesian Neural Networks via Horseshoe Priors

43.Learning Structural Weight Uncertainty for Sequential Decision-Making

44.Task Agnostic Continual Learning Using Online Variational Bayes

45.Variational Continual Learning

46.A Scalable Laplace Approximation for Neural Networks

47.Online Structured Laplace Approximations For Overcoming Catastrophic Forgetting

48.Loss-Calibrated Approximate Inference in Bayesian Neural Networks

49.Dropout Inference in Bayesian Neural Networks with Alpha-divergences

50.Noisy Natural Gradient as Variational Inference

51.Bayesian Dark Knowledge

52.Variational Implicit Processes

53.Learning Structured Weight Uncertainty in Bayesian Neural Networks

54.Multiplicative Normalizing Flows for Variational Bayesian Neural Networks

55.Kernel Implicit Variational Inference

56.Bayesian Hypernetworks

57.Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam

58.Adversarial Distillation of Bayesian Neural Network Posteriors

59.SLANG_Fast Structured Covariance Approximations for Bayesian Deep Learing with Natural Gradient

60.Sparse Bayesisan Learning and the Relevance Vector Machine (2001)

61.Uncertainty Decomposition in Bayesian Neural Networks with Latent Variables

62.Dropout as a Bayesian Approximation_Representing Model Uncertainty in Deep Learning

63.Bayesian Compression for Deep Learning

64.Structured Variational Learning of Bayesian Neural Networks with Horseshoe Priors

65.Noise Contrastive Priors for Functional Uncertainty

66.Variance Networks ; When Expectation Does Not Meet Your Expectations

67.Sylvester Normalizing Flows for Variational Inference (2019)

68.The Description Length of Deep Learning Models

69.Deterministic Variational Inference For Robust Bayesian Neural Networks

70.Understanding Priors in Bayesian Neural Networks at the Unit Level