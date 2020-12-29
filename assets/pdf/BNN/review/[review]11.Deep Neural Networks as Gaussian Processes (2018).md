## [ Paper review 11 ]

# Deep Neural Networks as Gaussian Processes

### ( Jaehoon Lee, et.al, 2018 )



## [ Contents ]

0. Abstract
1. Introduction
2. Probabilistic Neural Network Models
3. Probabilistic Backpropagation



# 0. Abstract

when $H \rightarrow \infty$ : single layer NN with a prior = GP  (Neal, 1994)

contribution: "show infinitely wide deep networks = GP "

- 1) trained NN accuracy approaches that of the corresponding GP
- 2) GP uncertainty is strongly correlated with trained network prediction error



# 1. Introduction

DNN \& GP

- DNN = flexible parametric models nowadays
- GP = traditional non-parametric tool
- ( limit of $\infty$ width ) the CLT implies that the function computed by NN = function drawn from GP ( Neal, 1994 )
- this substitution enables exact Bayesian inference for regression using NN ( Williams, 1997 )



### 1.1 Related Work

GP context

- infinite network = GP (Neal, 1994)

- GP prior for exact Bayesian Inference in Regression  (Williams, 1997)
- building deep GP \& observe degenerate form of kernels (Duvenaud et al, 2014)
- constructing kernels equivalent to infinitely wide DNN (Hazan \& Jaakola, 2015)



outside GP context

- derives compositional kernels for polynomial rectified nonlinearities (Cho \& Saul, 2009)
- extends the construnction of compositional kernels to NN (Daniely et al, 2016)



### 1.2 Summary of Contributions

begin by specifying the form of GP ( which corresponds to deep, infinitely wide NN ) = NNGP in terms of recursive, deterministic computation of the kernel function

Then, develop computationally efficient method to compute covariance function 



## 2. Deep, Infinitely Wide NN are drawn from GPs

### 2.1 Notation

$L$ : # of hidden layer

$N_L$ : width of layer $L$

$\phi$ : pointwise non-linearity

 $x \in \mathbb{R}^{d_{\text {in }}}$ : input

 $x_{i}^{l}$  :  $i$th component of the activations in $l$th layer, post-nonlinearity  ( = post activation ) 

 $z_{i}^{l}$  :  $i$th component of the activations in $l$th layer, post-affine transformation ( = pre activation )

$z^{L} \in \mathbb{R}^{d_{\text {out }}}$ : output ( = post-affine transformation )

$W_{i j}^{l}, b_{i}^{l}$ : weight and bias ( zero mean, and covariance with $\sigma_{w}^{2} / N_{l}$ and  $\sigma_{b}^{2}$ each)

$\mathcal{G} \mathcal{P}(\mu, K)$ : GP with mean, covariance $\mu(\cdot), K(\cdot, \cdot),$ respectively.



### 2.2 Review of GP and 1-layer NN

The $i$ th component of the network output, $z_{i}^{1},$ is computed as,
$$
z_{i}^{1}(x)=b_{i}^{1}+\sum_{j=1}^{N_{1}} W_{i j}^{1} x_{j}^{1}(x), \quad x_{j}^{1}(x)=\phi\left(b_{j}^{0}+\sum_{k=1}^{d_{i n}} W_{j k}^{0} x_{k}\right)
$$

- $x_k$ : pre-activation
- $x_{i}^{l}(x)$ : post-activation
- $z_{i}^{l}(x)$ : pre-activation



by CLT, as $N_{1} \rightarrow \infty$

- $z_{i}^{1}(x)$ is Gaussian distributied

- any finite collection of $\left\{z_{i}^{1}\left(x^{\alpha=1}\right), \ldots, z_{i}^{1}\left(x^{\alpha=k}\right)\right\}$ will have a joint MVN ( = GP )

$\therefore$ $z_{i}^{1} \sim \mathcal{G} \mathcal{P}\left(\mu^{1}, K^{1}\right)$

- mean : $\mu^{1}(x)=\mathbb{E}\left[z_{i}^{1}(x)\right]=0$
- covariance : $K^{1}\left(x, x^{\prime}\right) \equiv \mathbb{E}\left[z_{i}^{1}(x) z_{i}^{1}\left(x^{\prime}\right)\right]=\sigma_{b}^{2}+\sigma_{w}^{2} \mathbb{E}\left[x_{i}^{1}(x) x_{i}^{1}\left(x^{\prime}\right)\right] \equiv \sigma_{b}^{2}+\sigma_{w}^{2} C\left(x, x^{\prime}\right)$



### 2.3 GP and DNN

previous sections(works) can be extended to DEEPER layers

( $N_1 \rightarrow \infty$, $N_2 \rightarrow \infty$, $N_3 \rightarrow \infty$ ..... )



Suppose that $z_j^{l-1}$ is GP. After $l-1$ steps..

$z_{i}^{l}(x)=b_{i}^{l}+\sum_{j=1}^{N_{l}} W_{i j}^{l} x_{j}^{l}(x), \quad x_{j}^{l}(x)=\phi\left(z_{j}^{l-1}(x)\right)$

- $z_i^l(x)$ is a sum of i.i.d random terms
- Thus, CLT works! $\left\{z_{i}^{1}\left(x^{\alpha=1}\right), \ldots, z_{i}^{1}\left(x^{\alpha=k}\right)\right\}$ follows MVN
- Therefore, $z_{i}^{l} \sim \mathcal{G} \mathcal{P}\left(0, K^{l}\right)$



$z_{i}^{l} \sim \mathcal{G} \mathcal{P}\left(0, K^{l}\right)$

- mean : 0

- covariance : 

  $\begin{aligned}K^{l}\left(x, x^{\prime}\right) &\equiv \mathbb{E}\left[z_{i}^{l}(x) z_{i}^{l}\left(x^{\prime}\right)\right]\\
  &=\sigma_{b}^{2}+\sigma_{w}^{2} \mathbb{E}_{z_{i}^{l-1} \sim \mathcal{G} \mathcal{P}\left(0, K^{l-1}\right)}\left[\phi\left(z_{i}^{l-1}(x)\right) \phi\left(z_{i}^{l-1}\left(x^{\prime}\right)\right)\right]\\
  &=\sigma_{b}^{2}+\sigma_{w}^{2} F_{\phi}\left(K^{l-1}\left(x, x^{\prime}\right), K^{l-1}(x, x), K^{l-1}\left(x^{\prime}, x^{\prime}\right)\right)\end{aligned}$

  ( RECURSIVE relationship between $K^l$ and $K^{l-1}$ via deterministic function $F$, whose form depends only on the non-linearity $\phi$  $\rightarrow$ iterative series! )



For the base case $K^0$,

- weight: $W_{i j}^{0} \sim \mathcal{N}\left(0, \sigma_{w}^{2} / d_{\mathrm{in}}\right)$ \& bias : $b_{j}^{0} \sim \mathcal{N}\left(0, \sigma_{b}^{2}\right)$

- $K^{0}\left(x, x^{\prime}\right)=\mathbb{E}\left[z_{j}^{0}(x) z_{j}^{0}\left(x^{\prime}\right)\right]=\sigma_{b}^{2}+\sigma_{w}^{2}\left(\frac{x \cdot x^{\prime}}{d_{\text {in }}}\right)$



### 2.4 Bayesian Training for NN, using GP priors

How GP prior over functions can be used to do Bayesian Inference (Rasmussen \& Williams, 2006 )

- data : $\mathcal{D}=\left\{\left(x^{1}, t^{1}\right), \ldots,\left(x^{n}, t^{n}\right)\right\}$

- distribution over functions : $z(x)$

  (  $z \equiv\left(z^{1}, \ldots, z^{n}\right)$ on the training inputs $x \equiv\left(x^{1}, \ldots, x^{n}\right)$ )

- targets on training set : $\mathbf{t}$

- goal : make prediction at test point $x^*$, using a distribution over functions $z(x)$

  $\begin{aligned}P\left(z^{*} \mid \mathcal{D}, x^{*}\right)&=\int  P\left(z^{*} \mid z, x, x^{*}\right) P(z \mid \mathcal{D})d z\\&=\frac{1}{P(\mathbf{t})} \int P\left(z^{*}, z \mid x^{*}, x\right) P(\mathbf{t} \mid z)d z\end{aligned}$

  

$z^{*}, z \mid x^{*}, x \sim \mathcal{N}(0, \mathbf{K})$, where $\mathbf{K}=\left[\begin{array}{ll}
K_{\mathcal{D}, \mathcal{D}} & K_{x^{*}, \mathcal{D}}^{T} \\
K_{x^{*}, \mathcal{D}} & K_{x^{*}, x^{*}}
\end{array}\right]$

- $K_{\mathcal{D}, \mathcal{D}}$ is an $n \times n$ matrix whose $(i, j)$ th element is $K\left(x^{i}, x^{j}\right)$ with $x^{i}, x^{j} \in \mathcal{D}$ 
- the $i$ th element of $K_{x^{*}, \mathcal{D}}$ is $K\left(x^{*}, x^{i}\right), x^{i} \in \mathcal{D}$. 



$P\left(z^{*} \mid \mathcal{D}, x^{*}\right)$ = $z^{*} \mid \mathcal{D}, x^{*} \sim \mathcal{N}(\bar{\mu}, \bar{K})$

- mean : $\bar{\mu} =K_{x^{*}, \mathcal{D}}\left(K_{\mathcal{D}, \mathcal{D}}+\sigma_{\epsilon}^{2} \mathbb{I}_{n}\right)^{-1} \boldsymbol{t}$
- covariance : $\bar{K} =K_{x^{*}, x^{*}}-K_{x^{*}, \mathcal{D}}\left(K_{\mathcal{D}, \mathcal{D}}+\sigma_{\epsilon}^{2} \mathbb{I}_{n}\right)^{-1} K_{x^{*}, \mathcal{D}}^{T}$



form of the covariance function used is determined by the choice of GP prior

( NN : depth, nonlinearity, and weight and bias variances )



### 2.5 Efficient Implementation of the GP Kernel

constructing covariance matrix $K^L$ 

= computing Gaussian integral $\sigma_{b}^{2}+\sigma_{w}^{2} \mathbb{E}_{z_{i}^{l-1} \sim \mathcal{G} \mathcal{P}\left(0, K^{l-1}\right)}\left[\phi\left(z_{i}^{l-1}(x)\right) \phi\left(z_{i}^{l-1}\left(x^{\prime}\right)\right)\right]$  for all train \&test pairs

( recursively for all layers )



for some nonlinearities..

- RELU : integration can be done "analytically"
- kernel corresponding to arbitrary nonlinearities : must be done "numerically"



Simple way : compute integrals independently for each pair of data points \& each layer

​	$\rightarrow$ $\mathcal{O}\left(n_{g}^{2} L\left(n_{\text {train }}^{2}+n_{\text {train }} n_{\text {test }}\right)\right)$

Pre-process all the inputs to have identical norm

​	$\rightarrow$ $\mathcal{O}\left(n_{g}^{2} n_{v} n_{c}+L\left(n_{\text {train }}^{2}+n_{\text {train }} n_{\text {test }}\right)\right)$



#### STEP

[step 1] Generate

- pre-activations $u=\left[-u_{\max }, \cdots, u_{\max }\right] $    ......  $n_g$ elements
- variances $s=\left[0, \cdots, s_{\max }\right]$ ......  $n_v$ elements
- correlations $c=(-1, \cdots, 1)$ ......  $n_c$ elements



[step 2] Populate a matrix $F$

- involves numerically approximating Gaussian integral

  ( in terms of marginal variances $s$ and $c$ )

![[image-20201206153033891]](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201206153033891.png)



![image-20201206153241877](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201206153241877.png)



This computational recipe allows us to compute the covariance matrix for the NNGP corresponding
to any well-behaved nonlinearity $\phi$

