---
title: Summary of Bayesian DL papers (1~10)
categories: [BNN]
tags: [Bayesian Machine Learning, Bayesian Deep Learning, Probabilistic Deep Learning, Uncertainty Estimation, Variational Inference]
excerpt: Bayesian ML/DL, Probabilistic DL, Uncertainty Estimation
---

# Summary of Bayesian DL papers [1~10]

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

I have summarized the **must read + advanced papers** of papers regarding....

- various methods using Variational Inference

- Bayesian Neural Network

- Probabilistic Deep Learning

- Uncertainty Estimation




## 01. A Practical Bayesian Framework for Backpropagation Networks 

### MacKay, D. J. (1992)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/01.A Practical Bayesian Framework for Backpropagation Networks.pdf' | /assets/pdf/BNN/paper/01.A Practical Bayesian Framework for Backpropagation Networks.pdf }}) )

- how Bayesian Framework can be applied to Neural Network
- loss function = train loss + regularizer
  - train loss : for good fit
  - regularizer : for generalization
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]1.A Practical Bayesian Framework for Backpropagation Networks(1992).pdf' | /assets/pdf/BNN/review/[review]1.A Practical Bayesian Framework for Backpropagation Networks(1992).pdf }})

<br>

<br>

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

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]2-1.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/[review]2-1.Bayesian Learning For Neural Network(1995).pdf }})

<br>

<br>

## 02. Bayesian Learning for Neural Network (2)

- priors of weight? obscure in NN
- Infinte network = 'non-parametric model'
  - "network with 1 hidden layers with  INFINITE number of unit is an universal approximator"
  - "converges to GP(Gaussian Process)"

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]2-2.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/[review]2-2.Bayesian Learning For Neural Network(1995).pdf }})

<br>

<br>

## 02. Bayesian Learning for Neural Network (3)

- Hamiltonian Monte Carlo (HMC)
  - draw auxiliary momentum variable
  - calculate derivative
  - leap frog integrator
  - Metropolis acceptance step
- HMC is the most promising MC method
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]2-3.Bayesian Learning For Neural Network(1995).pdf' | /assets/pdf/BNN/review/[review]2-3.Bayesian Learning For Neural Network(1995).pdf }})

<br>

<br>

## 03. Keeping Neural Networks Simple by Minimizing the Description Length of the Weights

### Hinton, G. E., & Van Camp, D. (1993)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/03.Keeping Neural Networks Simple by Minimmizing the Description Length of the Weights.pdf' | /assets/pdf/BNN/paper/03.Keeping Neural Networks Simple by Minimmizing the Description Length of the Weights.pdf }}) )

- MDL (Minimum Description Length) Principle

  - NN generalize weights if "LESS information" is in the weights

  - loss(cost) = 1) + 2)

    - loss 1) cost for describing the model ( = weight penalty )
    - loss 2) describing the misfit between model & data  ( = train loss )

  - bits back argument

    - step 1) sender collapse the weights drawn from $$Q(w)$$

    - step 2) sender sends each weight for $$Q(w)$$ and sends the data misfit
    - step 3) receiver recovers the exact same posterior $$Q(w)$$ with correct output & misfits
    - step 4) calculate true expected description length for a noisy weight

  - start of Variational Inference...?

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]3.Keeping Neural Networks Simple by Minimizing the Description Length of the Weights(1993).pdf' | /assets/pdf/BNN/review/[review]3.Keeping Neural Networks Simple by Minimizing the Description Length of the Weights(1993).pdf }})

<br>

<br>

## 04. Practical Variational Inference for Neural Networks

### Graves, A. (2011)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/04.Practical Variational Inference for Neural Networks.pdf' | /assets/pdf/BNN/paper/04.Practical Variational Inference for Neural Networks.pdf }}) )

- introduce "Stochastic Variational method" 

- Key point

  - 1) instead of analytical solutions, use numerical integration
  - 2) stochastic method for VI with a diagonal Gaussian posterior

- takes a view of MDL

  ( ELBO (Variational Free Energy)can be viewed with MDL principle! )

  - ELBO : entropy loss + complexity loss
  - MDL : cost of transmitting the model + cost of transmitting the prior

- Diagonal Gaussian posterior

  - each weight requires a separate mean & variance
  - cannot compute derivative of loss function(-ELBO) directly.... use MC integration

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]4.Practical Variational Inference for Neural Networks(2011).pdf' | /assets/pdf/BNN/review/[review]4.Practical Variational Inference for Neural Networks(2011).pdf }})

<br>

<br>

## 05. Ensemble Learning in Bayesian Neural Networks

### Charles Blundell, et.al ( 2015 )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/05.Ensemble Learning in Bayesian Neural Networks.pdf' | /assets/pdf/BNN/paper/05.Ensemble Learning in Bayesian Neural Networks.pdf }}) )

- Bayesian for NN : 3 approaches

  - 1) Gaussian Approximation
    - knwon as Laplace's method
      - centered at the mode of $$p(w\mid D)$$
  - 2) MCMC
    - generate samples from the posterior
    - computationally expensive
    - ex) HMC
  - 3) Ensemble Learning
    - unlike Laplace's method, fitted globally

- Ensemble Learning

  - use ELBO / Variational Free Energy

  - minimize KL divergence 

  - choice of $$Q$$ ( approximating distribution )

    - should be close to true posterior
    - analytically tractable integration

  - original)  diagonal covariance  ( Hinton and van Camp, 1993 )

    proposed) full covariance

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]5.Ensemble Learning in Bayesian Neural Networks(1998).pdf' | /assets/pdf/BNN/review/[review]5.Ensemble Learning in Bayesian Neural Networks(1998).pdf }})

<br>

<br>

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

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]6.Weight Uncertainty in Neural Networks(2015).pdf' |/assets/pdf/BNN/review/[review]6.Weight Uncertainty in Neural Networks(2015).pdf  }})

<br>

<br>

## 07. Expectation Propagation for Approximate Bayesian Inference

### Minka, T. P. (2013)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/07.Expectation Propagation for Approximate Bayesian Inference(2001).pdf' | /assets/pdf/BNN/paper/07.Expectation Propagation for Approximate Bayesian Inference(2001).pdf }}) )

- Expectation Propagation 

  - EP = ADF + Loopy belief propagation

    ( ADF = online Bayesian Learning, moment matching, weak marginalization... )

  - one-pass, sequential method for computing approximate posterior

- novel interpretation of ADF

  - original ADF) approximate posterior that  includes each observation term $t_i$
  - new interpretation) using an exact posterior with $$\tilde{t_i}$$ (=ratio of new \& old posterior )

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]7.Expecation Propagation for Approximate Bayesian Inference(2001).pdf' | /assets/pdf/BNN/review/[review]7.Expecation Propagation for Approximate Bayesian Inference(2001).pdf }})

<br>

<br>

## 08. Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks

### Hernández-Lobato, J. M., & Adams, R. (2015, June)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/08.Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks.pdf' | /assets/pdf/BNN/paper/08.Probabilistic Backpropagation for Scalable Learning for Bayesian Neural Networks.pdf }}) )

- Probabilistic NN

- disadvantage of backpropagation

  - problem 1) have to tune large numbers of hyperparameters

  - problem 2) lack of calibrated probabilistic predictions

  - problem 3) tendency to overfit

    $$\rightarrow$$ solve by Bayesian Approach ! with PBP

- PBP ( Probabilistic Backpropagation )

  - scalable method for learning BNN

  - [step 1] forward propagataion of probabilities

    [step 2] backward computation of gradients

  - provides accurate estimates of the posterior variance

- PBP solves problem 1)~3) by

  - 1) automatically infer ( by marginalizing out of the posterior )
  - 2) account for uncertainty
  - 3) average over the parameter values

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]8.Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks(2015).pdf' | /assets/pdf/BNN/review/[review]8.Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks(2015).pdf }})

<br>

<br>

## 09. Priors For Infinite Networks 

### Neal, R. M. (1994)

( part of  **2.Bayesian Learning for Neural Network (2)** )

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/09.Priors For Infinte Networks (1994).pdf' | /assets/pdf/BNN/paper/09.Priors For Infinte Networks (1994) }}) )

- infinite network = non-parametric model
- Priors over functions reach reasonable limits, as the number of hidden units in the network goes infinity!
- summary : [Download]({{ '/assets/pdf/BNN/review/[review]9.Priors For Infinite Networks (1994).pdf' | /assets/pdf/BNN/review/[review]9.Priors For Infinite Networks (1994).pdf }})

<br>

<br>

## 10. Computing with Infinite Networks

### Williams, C. K. (1997)

( download paper here :  [Download]({{ '/assets/pdf/BNN/paper/10.Comuting with Infinite Networks (1997).pdf' | /assets/pdf/BNN/paper/10.Comuting with Infinite Networks (1997).pdf }}) )

- when number of hidden units $$H \rightarrow \infty$$, it is same as GP (Neal, 1994)

- Neal (1994) : Infinite NN=GP , but does not give the covariance function

  this paper : 

  - for certain weight priors (Gaussian) and transfer functions (Sigmoidal, Gaussian) in NN

    $$\rightarrow$$ the covariance function of GP can be calculated analytically!

- summary : [Download]({{ '/assets/pdf/BNN/review/[review]10.Computing with Infinite Networks (1997).pdf' | /assets/pdf/BNN/review/[review]10.Computing with Infinite Networks (1997).pdf }})


