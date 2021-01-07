---
title: Summary of Bayesian DL papers (11~20)
categories: [BNN]
tags: [Bayesian Machine Learning, Bayesian Deep Learning, Probabilistic Deep Learning, Uncertainty Estimation, Variational Inference]
excerpt: Bayesian ML/DL, Probabilistic DL, Uncertainty Estimation
---

# Summary of Bayesian DL papers [11~20]

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

I have summarized the **must read + advanced papers** of papers regarding....

- various methods using Variational Inference

- Bayesian Neural Network

- Probabilistic Deep Learning

- Uncertainty Estimation



## 11. Deep Neural Networks as Gaussian Processes

### Lee, J., et al. (2017)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/11.Deep Neural Networks as Gaussian Processes.pdf' | /assets/pdf/BNN/paper/11.Deep Neural Networks as Gaussian Processes.pdf }}) )

- Neal (1994) : 1-layer NN = GP

  this paper : DNN = GP

- recursive, deterministic computation of kernel function

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]11.Deep Neural Networks as Gaussian Processes (2018).pdf' | /assets/pdf/BNN/review/[review]11.Deep Neural Networks as Gaussian Processes (2018).pdf }})

<br>

<br>

## 12. Representing Inferential Uncertainty in Deep Neural Networks through Sampling

### McClure, P., & Kriegeskorte, N. (2016)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf' | /assets/pdf/BNN/paper/12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf}}) )

- Bayesian model catches model uncertainty
- Bayesian DNN trained with..
  - 1) Bernoulli drop out
  - 2) Bernoulli drop connect
  - 3) Gaussian drop out
  - 4) Gaussian drop connect
  - **5) Spike-and Slab Dropout**
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf' | /assets/pdf/BNN/review/[review]12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf }})

<br>

<br>

## 13. Bayesian Uncertainty Estimation for Batch Normalized Deep Networks 

### Teye, M., Azizpour, H., & Smith, K. (2018)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf' | /assets/pdf/BNN/paper/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf }}) )

- BN (Batch Normalization) = approximate inference in Bayesian models

  $$\rightarrow$$ allow us to estimate "model uncertainty" under the "conventional architecture" !!

  ( Previous works : mostly required modification of architecture )

- BN

  - training) use mini batch ( estimated mean & var for each mini-batch )
- evaluation) use all the training data
  
- (1) Bayesian Modeling : VA (Variational Approximation)

  (2) DNN with Batch Normalization

  $$\rightarrow$$ result : (1) = (2)

- predictive uncertainty in Batch Normalized Deep Nets!

- network is trained just as a regular BN network!

  ( but, instead of replacing $$w=\{\mu_B, \sigma_B\})$$ with population values from $$D$$,

  update these params stochastically! )

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf' | /assets/pdf/BNN/review/[review]13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf }})

<br>

<br>

## 14. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

### Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/14.Simple and Scalable Predictive Uncertainty (2017).pdf' | /assets/pdf/BNN/paper/14.Simple and Scalable Predictive Uncertainty (2017).pdf }}) )

- propose an alternative to BNN 

  ( not a Bayesian Method )

- advantages

  - 1) simple to implement, 2) parallelizable. 3) very little hyperparameter tuning,

    4) yields high quality of predictive uncertainty estimates

- 2 evaluation measures

  - (1) calibration ( measure by proper scoring rules )
  - (2) generalization to unknown class ( OOD examples )

- two modifications

  - Ensembles

  - Adversarial training
( adversarial examples : close to the original train data, but misclassified by NN )
  
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]14.Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2017).pdf' | /assets/pdf/BNN/review/[review]14.Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2017).pdf }})

<br>

<br>

## 15. Fast Dropout Training

### Wang, S., & Manning, C. (2013)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/15.Fast Dropout Training (2013).pdf' | /assets/pdf/BNN/paper/15.Fast Dropout Training (2013).pdf }}) )

- Dropout : repeatedly sampling makes it slower

  $$\rightarrow$$ use Gaussian Approximation to make it faster!

- problems with dropout

  - 1) slow training
  - 2) loss of information

- with Gaussian Approximation $$Y \rightarrow S$$:

  - (1) faster ( without actually sampling )  

    (2) efficient ( use all data )

  - ( $$m$$ : number of dimension , $$K$$ : number of samples)

    original dropout ) sample from $$Y$$ .... $$O(mK)$$ times 

    with GA ) sample from $$S$$ .... $$O(K)$$ times 

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]15.Fast Dropout Training (2013).pdf' | /assets/pdf/BNN/review/[review]15.Fast Dropout Training (2013).pdf }})

<br>

<br>

## 16. Variational Dropout and Local Reparameterization Trick

### Kingma, D. P., Salimans, T., & Welling, M. (2015)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/16.Variational Dropout and Local Reparameterization Trick (2015).pdf' | /assets/pdf/BNN/paper/16.Variational Dropout and Local Reparameterization Trick (2015).pdf }}) )

- propose LRT for reducing variance of SGVB
  - LRT : Local Reparameterization Trick
  - SGVB : Stochastic Gradient Variational Bayes

- LRT :

  - translates uncertainty about global parameters into local noise 
    ( which is independent across mini-batch )
  - can be parallelized
  - has variance ( inversely proportional to the mini-batch size $$M$$ )

- connection with dropout

  - Gaussian dropout = SGVB with LRT

  - propose "Variational Dropout"

    ( = generalization of Gaussian dropout )

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]16.Variational Dropout and Local Reparameterization Trick (2015).pdf' | /assets/pdf/BNN/review/[review]16.Variational Dropout and Local Reparameterization Trick (2015).pdf }})

<br>

<br>

## 17. Dropout as a Bayesian Approximation : Representing Model Uncertainty in Deep Learning 

### Gal, Y., & Ghahramani, Z. (2016)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/17.Dropout as a Bayesian Approximation_Representing Model Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/paper/17.Dropout as a Bayesian Approximation_Representing Model Uncertainty in Deep Learning (2016).pdf }}) )

- Model Uncertainty with Dropout NNs

  ( Dropout in NN = approximate Bayesian inference in deep Gaussian )

- Dropout

  - can be interpreted as "Bayesian Approximation" of GP
  - avoid overfitting
  - approximately integrates over the models' weights

- obtaining model uncertainty

  - [step 1] sample $$T$$ set of vectors
  - [step 2] find $$W$$
  - [step 3] MC Dropout ( estimate mean, variance )

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]17.Dropout as Bayesian Approximation (2016).pdf' | /assets/pdf/BNN/review/[review]17.Dropout as Bayesian Approximation (2016).pdf }})

<br>

<br>

## 18. Variational Dropout Sparsifies Deep Neural Networks

### Molchanov, D., Ashukha, A., & Vetrov, D. (2017)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf' | /assets/pdf/BNN/paper/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf }}) )

- Key point

  - 1) Sparse Variational Dropout

    $$\rightarrow$$ extend VD(Variational Dropout) to the case where "dropout rates are unbounded"

  - 2) reduce the variance of the gradient estimator

    $$\rightarrow$$ leads to faster convergence

- Instead of injecting noise...use "Sparsity"

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf' | /assets/pdf/BNN/review/[review]18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf }})

<br>

<br>

## 19. Relevance Vector Machine Explained

### Fletcher, T. (2010)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/19.Relevance Vector Machine Explained (2010).pdf' | /assets/pdf/BNN/paper/19.Relevance Vector Machine Explained (2010).pdf }}) )

- problems with SVM (Support Vector Machine)

  - not a probabilistic prediction
  - only binary decision
  - have to tune the hyperparameter $$C$$

- SVM vs RVM

  - Sparsity : RVM > SVM

  - Generalization : RVM > SVM

  - Need to estimate hyperparamter : only SVM

  - Training time : RVM > SVM 

    ( can be solved with sparsity )

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]19.Relevance Vector Machine Explained (2010).pdf' | /assets/pdf/BNN/review/[review]19.Relevance Vector Machine Explained (2010).pdf }})

<br>

<br>

## 20. Uncertainty in Deep Learning (1)

### Gal, Y. (2016)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/20.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/paper/20.Uncertainty in Deep Learning (2016).pdf }}) )

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

  $$\rightarrow$$ need practical techniques (ex. SRT)

- SRT (Stochastic Regularization Techniques)

  - adapt the model output "stochastically" as a way of model regularization
  - predictive mean/variance \& random output

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]20-1.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/[review]20-1.Uncertainty in Deep Learning (2016).pdf }})

<br>

<br>

## 20. Uncertainty in Deep Learning (2)

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
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]20-2.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/[review]20-2.Uncertainty in Deep Learning (2016).pdf }})

<br>

<br>

## 20. Uncertainty in Deep Learning (3)

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

    $$\rightarrow$$ solve by data sub-sampling (mini-batch optimization)

  - problem 2) intractable integral

    $$\rightarrow$$ MC integration

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

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]20-3.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/[review]20-3.Uncertainty in Deep Learning (2016).pdf }})
