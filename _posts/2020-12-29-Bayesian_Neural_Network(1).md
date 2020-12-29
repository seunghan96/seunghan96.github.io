---
title: Summary of Bayesian DL papers (1)
categories: [BNN]
tags: [Bayesian Machine Learning, Bayesian Deep Learning, Probabilistic Deep Learning, Uncertainty Estimation, Variational Inference]
excerpt: Bayesian ML/DL, Probabilistic DL, Uncertainty Estimation
---

# Summary of Bayesian DL papers [1~20]

I have summarized the **must read + advanced papers** of papers regarding....

- various methods using Variational Inference

- Bayesian Neural Network

- Probabilistic Deep Learning

- Uncertainty Estimation

  

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## 01. A Practical Bayesian Framework for Backpropagation Networks

### MacKay, D. J. (1992)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/01.A Practical Bayesian Framework for Backpropagation Networks.pdf' | /assets/pdf/BNN/paper/01.A Practical Bayesian Framework for Backpropagation Networks.pdf }}) )

- how Bayesian Framework can be applied to Neural Network
- loss function = train loss + regularizer
  - train loss : for good fit
  - regularizer : for generalization
- summary : [Download]({{ '/assets/pdf/BNN/review/1.A Practical Bayesian Framework for Backpropagation Networks(1992).pdf' | /assets/pdf/BNN/review/1.A Practical Bayesian Framework for Backpropagation Networks(1992).pdf }})





## 02. Bayesian Learning for Neural Network (1)

### Neal, R. M. (2012) 	

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/02.Bayesian Learning for Neural Network.pdf' | /assets/pdf/BNN/paper/02.Bayesian Learning for Neural Network.pdf }}) )

- Bayesian view in NN : find predictive distribution by "Integration", rather than "maximization"
- BNN : not only single guess! also "UNCERTAINTY"
- MAP(Maximum a posteriori probability) : act as PENALIZED likelihood ( penalty term by "prior" )
- BNN = automatic Occam's Razor
- ARD (Automatic Relevance Determination model) 
  -  limit the number of input variables "automatically"
  - each input variable has its own hyperparameter, that controls the weight 
- review of MCMC
  - MC approximation : unbiased estimate
  - MH algorithm, Gibbs...
  - Hybrid Monte Carlo : auxiliary variable, 'momentum'

- summary : [Download]({{ '/assets/pdf/BNN/review/2-1.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/2-1.Bayesian Learning For Neural Network(1995).pdf }})





## 02. Bayesian Learning for Neural Network (2)

- priors of weight? obscure in NN
- Infinte network = 'non-parametric model'
  - "network with 1 hidden layers with  INFINITE number of unit is an universal approximator"
  - "converges to GP(Gaussian Process)"

- summary : [Download]({{ '/assets/pdf/BNN/review/2-2.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/2-2.Bayesian Learning For Neural Network(1995).pdf }})





## 02. Bayesian Learning for Neural Network (3)

- Hamiltonian Monte Carlo
  - draw auxiliary momentum variable
  - calculate derivative
  - leap frog integrator
  - Metropolis acceptance step
- HMC is the most promising mC method
- summary : [Download]({{ '/assets/pdf/BNN/review/2-3.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/2-3.Bayesian Learning For Neural Network(1995).pdf }})





## 03. Keeping Neural Networks Simple by Minimizing the Description Length of the Weights

### Hinton, G. E., & Van Camp, D. (1993)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/03.Keeping Neural Networks Simple by Minimmizing the Description Length of the Weights.pdf' | /assets/pdf/BNN/paper/03.Keeping Neural Networks Simple by Minimmizing the Description Length of the Weights.pdf }}) )

- MDL (Minimum Description Length) Principle

  - NN generalize weights if "LESS information" is in the weights

  - loss(cost) = 1) + 2)

    - loss 1) cost for describing the model ( = weight penalty )
    - loss 2) describing the misfit between model & data  ( = train loss )

  - bits back argument

    - step 1) sender collapse the weights drawn from $Q(w)$

    - step 2) sender sends each weight for $Q(w)$ and sends the data misfit
    - step 3) receiver recovers the exact same posterior $Q(w)$ with correct output & misfits
    - step 4) calculate true expected description length for a noisy weight

  - start of Variational Inference...?

- summary : [Download]({{ '/assets/pdf/BNN/review/3.Keeping Neural Networks Simple by Minimizing the Description Length of the Weights(1993).pdf' | /assets/pdf/BNN/review/3.Keeping Neural Networks Simple by Minimizing the Description Length of the Weights(1993).pdf }})





## 04. Practical Variational Inference for Neural Networks

### Graves, A. (2011)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/04.Practical Variational Inference for Neural Networks.pdf' | /assets/pdf/BNN/paper/04.Practical Variational Inference for Neural Networks.pdf }}) )

- introduce "Stochastic Variational method" 

- Key point

  - 1) instead of analytical solutions, use numerical integration
  - 2) stochastic method for VI with a diagonal Gaussian posterior

- takes a view of MDL

  ( ELBO (Variational Free Energy)can be viewd with MDL principle! )

  - ELBO : entropy loss + complexity loss
  - MDL : cost of transmitting the model + cost of transmitting the prior

- Diagonal Gaussian posterior

  - each weight requires a separate mean \& variance
  - cannot compute derivative of loss function(-ELBO) directly.... use MC integration

- summary : [Download]({{ '/assets/pdf/BNN/review/4.Practical Variational Inference for Neural Networks(2011).pdf' | /assets/pdf/BNN/review/4.Practical Variational Inference for Neural Networks(2011).pdf }})





## 05. Ensemble Learning in Bayesian Neural Networks

### Barber, D., & Bishop, C. M. (1998)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/05.Ensemble Learning in Bayesian Neural Networks.pdf' | /assets/pdf/BNN/paper/05.Ensemble Learning in Bayesian Neural Networks.pdf }}) )

- Bayesian for NN : 3 approaches

  - 1) Gaussian Approximation
    - knwon as Laplace's method
    - centered at the mode of $p(w\mid D)$
  - 2) MCMC
    - generate samples from the posterior
    - computationally expensive
    - ex) HMC
  - 3) Ensemble Learning
    - unlike Laplace's method, fitted globally

- Ensemble Learning

  - use ELBO / Variational Free Energy

  - minimize KL divergence 

  - choice of $Q$ ( approximating distribution )

    - should be close to true posterior
    - analytically tractable integration

  - original)  diagonal covariance  ( Hinton and van Camp, 1993 )

    proposed) full covariance

- summary : [Download]({{ '/assets/pdf/BNN/review/5.Ensemble Learning in Bayesian Neural Networks(1998).pdf' | /assets/pdf/BNN/review/5.Ensemble Learning in Bayesian Neural Networks(1998).pdf }})





## 06. Weight Uncertainty in Neural Networks

### Barber, D., & Bishop, C. M. (1998)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/06.Weight Uncertainty in Neural Networks.pdf' | /assets/pdf/BNN/paper/06.Weight Uncertainty in Neural Networks.pdf }}) )

- Bayes by Backprop

  - regularize weight by minimizing a compression cost
  - comparable performance to dropout
  - uncertainty can be used to improve generalization
  - exploration-exploitation in RL

- BBB : instead of single networks, train "ensembles of networks"

  ( each network has its weights drawn from a distribution )

- previous works

  - early stopping, weight decay, dropout ....

- summary : [Download]({{ '/assets/pdf/BNN/review/6.Weight Uncertainty in Neural Networks(2015).pdf' |/assets/pdf/BNN/review/6.Weight Uncertainty in Neural Networks(2015).pdf  }})





## 07. Expectation Propagation for Approximate Bayesian Inference

### Minka, T. P. (2013)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/07.Expectation Propagation for Approximate Bayesian Inference(2001).pdf' | /assets/pdf/BNN/paper/07.Expectation Propagation for Approximate Bayesian Inference(2001).pdf }}) )

- Expectation Propagation 

  - EP = ADF + Loopy belief propagation

    ( ADF = online Bayesian Learning, moment matching, weak marginalization... )

  - one-pass, sequential method for computing approximate posterior

- novel interpretation of ADF

  - original ADF) approximate posterior that  includes each observation term $t_i$
  - new interpretation) using an exact posterior with $\tilde{t_i}$ (=ratio of new \& old posterior )

- summary : [Download]({{ '/assets/pdf/BNN/review/7.Expecation Propagation for Approximate Bayesian Inference(2001).pdf' | /assets/pdf/BNN/review/7.Expecation Propagation for Approximate Bayesian Inference(2001).pdf }})





## 08. Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks

### Hernández-Lobato, J. M., & Adams, R. (2015, June)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/08.Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks.pdf' | /assets/pdf/BNN/paper/08.Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks.pdf }}) )

- Probabilistic NN

- disadvantage of backpropagation

  - problem 1) have to tune large numbers of hyperparameters

  - problem 2) lack of calibrated probabilistic predictions

  - problem 3) tendency to overfit

    $\rightarrow$ solve by Bayesian Approach ! with PBP

- PBP ( Probabilistic Backpropagation )

  - scalable method for learning BNN

  - [step 1] forward propagataion of probabilities

    [step 2] backward computation of gradients

  - provides accurate estimates of the posterior variance

- PBP solves problem 1)~3) by

  - 1) automatically infer ( by marginalizing out of the posterior )
  - 2) account for uncertainty
  - 3) average over the parameter values

- summary : [Download]({{ '/assets/pdf/BNN/review/8.Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks(2015).pdf' | /assets/pdf/BNN/review/8.Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks(2015).pdf }})





## 09. Priors For Infinite Networks 

### Neal, R. M. (1994)

( part of  **2.Bayesian Learning for Neural Network (2)** )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/09.Priors For Infinte Networks (1994).pdf' | /assets/pdf/BNN/paper/09.Priors For Infinte Networks (1994) }}) )

- infinite network = non-parametric model
- Priors over functions reach reasonable limits, as the number of hidden units in the network goes infinity!
- summary : [Download]({{ '/assets/pdf/BNN/review/9.Priors For Infinite Networks (1994).pdf' | /assets/pdf/BNN/review/9.Priors For Infinite Networks (1994).pdf }})





## 10. Computing with Infinite Networks

### Williams, C. K. (1997)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/10.Comuting with Infinite Networks (1997).pdf' | /assets/pdf/BNN/paper/10.Comuting with Infinite Networks (1997).pdf }}) )

- when number of hidden units $H \rightarrow \infty$, it is same as GP (Neal, 1994)

- Neal (1994) : Infinite NN=GP , but does not give the covariance function

  this paper : 

  - for certain weight priors (Gaussian) and transfer functions (Sigmoidal, Gaussian) in NN

    $\rightarrow$ the covariance function of GP can be calculated analytically!

- review note 수정요함 (시공간자료분석)
- summary : [Download]({{ '/assets/pdf/BNN/review/10.Computing with Infinite Networks (1997).pdf' | /assets/pdf/BNN/review/10.Computing with Infinite Networks (1997).pdf }})





## 11. Deep Neural Networks as Gaussian Processes

### Lee, J., et al. (2017)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/11.Deep Neural Networks as Gaussian Processes.pdf' | /assets/pdf/BNN/paper/11.Deep Neural Networks as Gaussian Processes.pdf }}) )

- Neal (1994) : 1-layer NN = GP

  this paper : DNN = GP

- recursive, deterministic computation of kernel function

- summary : [Download]({{ '/assets/pdf/BNN/review/11.Deep Neural Networks as Gaussian Processes (2018).pdf' | /assets/pdf/BNN/review/11.Deep Neural Networks as Gaussian Processes (2018).pdf }})





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
- summary : [Download]({{ '/assets/pdf/BNN/review/12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf' | /assets/pdf/BNN/review/12.Representing Inferential Uncertainty in Deep Neural Networks through Sampling (2017).pdf }})





## 13. Bayesian Uncertainty Estimation for Batch Normalized Deep Networks 

### Teye, M., Azizpour, H., & Smith, K. (2018)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf' | /assets/pdf/BNN/paper/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf }}) )

- BN (Batch Normalization) = approximate inference in Bayesian models

  $\rightarrow$ allow us to estimate "model uncertainty" under the "conventional architecture" !!

  ( Previous works : mostly required modification of architecture )

- BN

  - training) use mini batch ( estimated mean \& var for each mini-batch )

    evaluation) use all the training data

  - train with mini-batch optimization... minimizing 

- (1) Bayesian Modeling : VA (Variational Approximation)

  (2) DNN with Batch Normalization

  $\rightarrow$ (1) = (2)

- predictive uncertainty in Batch Normalized Deep Nets!

- network is trained just as a regular BN network!

  ( but, instead of replacing $w=\{\mu_B, \sigma_B\})$ with population values from $D$,

  update these params stochastically! )

- summary : [Download]({{ '/assets/pdf/BNN/review/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf' | /assets/pdf/BNN/review/13.Bayesian Uncertainty Estimation for Batch Normalized Deep Networks (2018).pdf }})





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

    ( adversarial examples : close to the original training examples, but are misclassified by NN )

- summary : [Download]({{ '/assets/pdf/BNN/review/14.Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2017).pdf' | /assets/pdf/BNN/review/14.Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2017).pdf }})





## 15. Fast Dropout Training

### Wang, S., & Manning, C. (2013)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/15.Fast Dropout Training (2013).pdf' | /assets/pdf/BNN/paper/15.Fast Dropout Training (2013).pdf }}) )

- Dropout : repeatedly sampling makes it slower

  $\rightarrow$ use Gaussian Approximation to make it faster!

- problems with dropout

  - 1) slow training
  - 2) loss of information

- with Gaussian Approximation $Y \rightarrow S$:

  - faster(without actually sampling) + efficient(use all data)

  - ( $m$ : number of dimension , $K$ : number of samples)

    original dropout ) sample from $Y$ .... $O(mK)$ times 

    with GA ) sample from $S$ .... $O(mK)$ times 

- summary 수정요함
- summary : [Download]({{ '/assets/pdf/BNN/review/15.Fast Dropout Training (2013).pdf' | /assets/pdf/BNN/review/15.Fast Dropout Training (2013).pdf }})





## 16. Variational Dropout and Local Reparameterization Trick

### Kingma, D. P., Salimans, T., & Welling, M. (2015)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/16.Variational Dropout and Local Reparameterization Trick (2015).pdf' | /assets/pdf/BNN/paper/16.Variational Dropout and Local Reparameterization Trick (2015).pdf }}) )

- propose LRT for reducing variance of SGVB
  - LRT : Local Reparameterization Trick
  - SGVB : Stochastic Gradient Variational Bayes

- LRT :

  - translates uncertainty about global parameters into local noise ( which is independent across mini-batch )
  - can be parallelized
  - has variance ( inversely proportional to the mini-batch size $M$ )

- connection with dropout

  - Gaussian dropout = SGVB with LRT

  - propose "Variational Dropout"

    ( = generalization of Gaussian dropout )

- summary : [Download]({{ '/assets/pdf/BNN/review/16.Variational Dropout and Local Reparameterization Trick (2015).pdf' | /assets/pdf/BNN/review/16.Variational Dropout and Local Reparameterization Trick (2015).pdf }})





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

  - [step 1] sample $T$ set of vectors
  - [step 2] find $W$
  - [step 3] MC Dropout ( estimate mean, variance )

- summary : [Download]({{ '/assets/pdf/BNN/review/17.Dropout as Bayesian Approximation (2016).pdf' | /assets/pdf/BNN/review/17.Dropout as Bayesian Approximation (2016).pdf }})





## 18. Variational Dropout Sparsifies Deep Neural Networks

### Molchanov, D., Ashukha, A., & Vetrov, D. (2017)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf' | /assets/pdf/BNN/paper/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf }}) )

- Key point

  - 1) Sparse Variational Dropout

    $\rightarrow$ extend VD(Variational Dropout) to the case where "dropout rates are unbounded"

  - 2) reduce the variance of the gradient estimator

    $\rightarrow$ leads to faster convergence

- Instead of injecting noise...use "Sparsity"

- summary : [Download]({{ '/assets/pdf/BNN/review/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf' | /assets/pdf/BNN/review/18.Variational Dropout Sparsifies Deep Neural Networks (2017).pdf }})





## 19. Relevance Vector Machine Explained

### Fletcher, T. (2010)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/19.Relevance Vector Machine Explained (2010).pdf' | /assets/pdf/BNN/paper/19.Relevance Vector Machine Explained (2010).pdf }}) )

- problems with SVM (Support Vector Machine)

  - not a probabilistic prediction
  - only binary decision
  - have to tune the hyperparameter $C$

- SVM vs RVM

  - Sparsity : RVM > SVM

  - Generalization : RVM > SVM

  - Need to estimate hyperparamter : only SVM

  - Training time : RVM > SVM 

    ( can be solved with sparsity )

- summary : [Download]({{ '/assets/pdf/BNN/review/19.Relevance Vector Machine Explained (2010).pdf' | /assets/pdf/BNN/review/19.Relevance Vector Machine Explained (2010).pdf }})





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

  $\rightarrow$ need practical techniques (ex. SRT)

- SRT (Stochastic Regularization Techniques)

  - adapt the model output "stochastically" as a way of model regularization
  - predictive mean/variance \& random output

- summary : [Download]({{ '/assets/pdf/BNN/review/20-1.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/20-1.Uncertainty in Deep Learning (2016).pdf }})





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
- summary : [Download]({{ '/assets/pdf/BNN/review/20-2.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/20-2.Uncertainty in Deep Learning (2016).pdf }})





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

- summary : [Download]({{ '/assets/pdf/BNN/review/20-3.Uncertainty in Deep Learning (2016).pdf' | /assets/pdf/BNN/review/20-3.Uncertainty in Deep Learning (2016).pdf }})
