## [ Paper review 20 ]

# Uncertainty in Deep Learning - Chapter 1

### ( Yarin Gal, 2016 )



## [ Contents ]

1. Introduction : The importance of Knowing What We Don't Know
   1. Deep Learning
   2. Model Uncertainty
   3. Model Uncertainty and AI safety
   4. Applications of Model Uncertainty
   5. Model Uncertainty in Deep Learning
   6. Thesis structure
   
   

# 1. Introduction

## The importance of Knowing What We Don't Know

Probabilistic view : offers confidence bounds

Knowing Uncertainty is the fundamental concern in Bayesian ML

but most DL models are deterministic functions

$\rightarrow$ we can get uncertainty information from existing DL models for free!



## 1.1 Deep Learning

x $\&$ y need not be linear. ( non-linear function $f(x)$ )

linear basis function regression

- non-linear transformations $\phi_k(x)$ : our basis function

- compose a feature vector with $\phi_k(x)$

  $\Phi(\mathrm{x})=\left[\phi_{1}(\mathrm{x}), \ldots, \phi_{K}(\mathrm{x})\right] $



Parametrised basis function

- relax constraint

- define our basis function to be $\phi_k^{w_k, b_k}$

- $\left\langle\mathbf{w}_{k}, \mathbf{x}\right\rangle+b_{k}$

  - example ) $\phi_{k}(\cdot)=\sin (\cdot)$ then,  $\phi_{k}^{\mathrm{w}_{k}, b_{k}}(\mathrm{x})=\sin \left(\left\langle\mathrm{w}_{k}, \mathrm{x}\right\rangle+b_{k}\right)$

    $\mathrm{f}(\mathrm{x})=\Phi^{\mathrm{W}_{1}, \mathrm{~b}_{1}}(\mathrm{x}) \mathrm{W}_{2}+\mathrm{b}_{2}$ , where $\Phi^{\mathbf{W}_{1}, \mathbf{b}_{1}}(\mathbf{x})=\phi\left(\mathbf{W}_{1} \mathbf{x}+\mathbf{b}_{1}\right), \mathbf{W}_{1}$

    

### Feed-forward NN

( in case of single hidden layer )

$\widehat{\mathbf{y}}=\sigma\left(\mathbf{x} \mathbf{W}_{1}+\mathbf{b}\right) \mathbf{W}_{2}$



Loss function

- Regression : Euclidean Loss

  $E^{\mathbf{W}_{1}, \mathbf{W}_{2}, \mathbf{b}}(\mathbf{X}, \mathbf{Y})=\frac{1}{2 N} \sum_{i=1}^{N}\left\|\mathbf{y}_{i}-\widehat{\mathbf{y}}_{i}\right\|^{2}$

- Classification : softmax loss

  $\hat{p}_{d}=\exp \left(\hat{y}_{d}\right) /\left(\sum_{d^{\prime}} \exp \left(\hat{y}_{d^{\prime}}\right)\right) $

  $E^{\mathrm{W}_{1}, \mathrm{~W}_{2}, \mathrm{~b}}(\mathrm{X}, \mathrm{Y})=-\frac{1}{N} \sum_{i=1}^{N} \log \left(\widehat{p}_{i, d_{i}}\right)$

  where $d_{i} \in\{1,2, \ldots, D\}$ is the observed class for input $i$.

  

Weight decay ($\lambda_i$ )

- to prevent overfit
- $\mathcal{L}\left(\mathbf{W}_{1}, \mathbf{W}_{2}, \mathbf{b}\right):=E^{\mathbf{W}_{1}, \mathbf{W}_{2}, \mathbf{b}}(\mathbf{X}, \mathbf{Y})+\lambda_{1}\left\|\mathbf{W}_{1}\right\|^{2}+\lambda_{2}\left\|\mathbf{W}_{2}\right\|^{2}+\lambda_{3}\|\mathbf{b}\|^{2}$



### CNN \& RNN

CNN : 생략

RNN

- input sequence $\mathbf{x}=\left[\mathbf{x}_{1}, \ldots, \mathbf{x}_{T}\right]$ of length $T$

- hidden state $\mathbf{h}_{t}$ for time step $t$

- $\hat{\mathbf{y}}=\mathbf{f}_{\mathbf{y}}\left(\mathbf{h}_{T}\right)=\mathbf{h}_{T} \mathbf{W}_{\mathbf{y}}+\mathbf{b}_{\mathbf{y}}$

  where the hidden state is $\mathbf{h}_{t}=\mathbf{f}_{\mathbf{h}}\left(\mathbf{x}_{t}, \mathbf{h}_{t-1}\right)=\sigma\left(\mathbf{x}_{t} \mathbf{W}_{\mathbf{h}}+\mathbf{h}_{t-1} \mathbf{U}_{\mathbf{h}}+\mathbf{b}_{\mathbf{h}}\right)$



## 1.2 Model uncertainty

1) Out of Distribution

- the point lies outside of the data distribution

2) Aleatoric uncertainty

- noisy data

3) Model Uncertainty (=Epistemic uncertainty)

- uncertainty in "model parameters"
- structure uncertainty



## 1.3 Model uncertainty and AI safety

Need uncertainty for safety!



## 1.4 Applications of model uncertainty

Beside AI safety....

ex) to learn from "small amount" of data

- choosing what data to learn  from $\rightarrow$ in Active Learning
- exploring an agent's environment efficiently $\rightarrow$ in RL

 

# 1.5 Model uncertainty in Deep Learning

probability vector in softmax IS NOT model confidence



Modern DL models : do not capture confidence

- (probability vector in softmax IS NOT model confidence)
- but still closely related to probabilistic models like GP



Bayesian Neural Network

- GP can be recovered in limit of infinitely many weights ( Neal, 1995 )

- For a finite number of weights, 

  model uncertainty can still be obtained by placing "distribution over weights"

- Have been resurrected under different names with "variational techniques"



Models which gives us uncertainty..

- usually do not scale well to complex model \& big data

- thus, require us to develop new models for which we already have well performing tools

  ( we need practical techniques! such as SRT )



Stochastic Regularization Techniques (SRTs)

- for model regularization

- successful within DL

  ( we can take almost any network trained with an SRT )

- adapt the model output "stochastically" as a way of model regularization

- ex) Dropout, Multiplicative Gaussian Noise, Drop Connect, ...



How does SRT work?

- predictive mean \& predictive variance

- simulate a network with input $x^{*}$

  $\rightarrow$ random output

- repeat this many times

  $\begin{aligned}
  \mathbb{E}\left[\mathbf{y}^{*}\right] & \approx \frac{1}{T} \sum_{t=1}^{T} \widehat{\mathbf{y}}_{t}^{*}\left(\mathrm{x}^{*}\right) \\
  \operatorname{Var}\left[\mathbf{y}^{*}\right] & \approx \tau^{-1} \mathbf{I}_{D}+\frac{1}{T} \sum_{t=1}^{T} \widehat{\mathbf{y}}_{t}^{*}\left(\mathrm{x}^{*}\right)^{T} \widehat{\mathbf{y}}_{t}^{*}\left(\mathrm{x}^{*}\right)-\mathbb{E}\left[\mathbf{y}^{*}\right]^{T} \mathbb{E}\left[\mathbf{y}^{*}\right]
  \end{aligned}$

  ( this is practical with large models and big data! )



## 1.6 Thesis Structure

The code for the experiments presented in this work is available at

 https://github.com/yaringal