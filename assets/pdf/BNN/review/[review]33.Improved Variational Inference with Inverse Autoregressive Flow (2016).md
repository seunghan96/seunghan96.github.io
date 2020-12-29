## [ Paper review 33 ]

# Improved Variational Inference with Inverse Autoregressive Flow

### ( Kingma, et al. 2016 )



## [ Contents ]

1. Abstract
2. Introduction
3. Variational inference and Learning
   1. Requirements for computational tractability
   2. Normalizing Flow
4. Inverse Autoregressive Transformations
5. Inverse Autoregressive Flow (IAF)

6. Summary



# 1. Abstract

NF ( Normalizing Flow )

- strategy for flexible VI



 "Inverse Autoregressive Flow (IAF)"

- new type of NF, which scales well to high-dimensional latent spaces

- consists of cahin of invertible transformations

  ( each transformation is based on a "Autoregressive NN")

  

# 2. Introduction

SVI : scalable posterior inference, using stochastic gradient update

VAE : NN with inference network \& generative network

IAF : scales well to high-dimensional latent space



IAF ( Inverse Autoregressive Flow )

- lie in Gaussian autoregressive functions

  ( normally used for density estimation )

  	- input : variable with some specific ordering 
  	- output : mean and std for each element

  ex) RNN, MADE, PixelCNN, WaveNet

- such functions can....

  - be turned into "invertible" nonlinear transformations
  - with a "simple Jacobian determinant"

  $\rightarrow$ flexibliity + known determinant = Normalizing Flow



demonstrate this method by "improving the inference network" of deep VAE



# 3. Variational inference and Learning

notation

- $x$ : observed variable
- $z$  : latent variable
- $p(x,z)$ : joint pdf $\rightarrow$ "generative model"



Maximize ELBO

- $\log p(\mathrm{x}) \geq \mathbb{E}_{q(\mathrm{z} \mid \mathrm{x})}[\log p(\mathrm{x}, \mathrm{z})-\log q(\mathrm{z} \mid \mathrm{x})]=\mathcal{L}(\mathrm{x} ; \boldsymbol{\theta})\log p(\mathbf{x})-D_{K L}(q(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z} \mid \mathbf{x}))$

- re-parameterization trick in $q(\mathbf{z} \mid \mathbf{x})$



Models with multiple latent variable

- factorize $q(\mathbf{z} \mid \mathbf{x})$

  ( factorize into partial inference models with some ordering )

- $q\left(\mathbf{z}_{a}, \mathbf{z}_{b} \mid \mathbf{x}\right)=q\left(\mathbf{z}_{a} \mid \mathbf{x}\right) q\left(\mathbf{z}_{b} \mid \mathbf{z}_{a}, \mathbf{x}\right)$



## 3-1. Requirements for computational tractability

have to efficiently optimize ELBO!

Computationally efficient to...

- 1) compute and differentiate $q(z\mid x)$

- 2) sample from it

  ( since both these operations need to be performed for each datapoint in a minibatch at every iteration of optimization)



example) diagonal posterior

- $q(\mathbf{z} \mid \mathbf{x}) \sim \mathcal{N}\left(\boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\sigma}^{2}(\mathbf{x})\right),$
- but not much flexible.... 



## 3-2. Normalizing Flow

NF 

- in the context of SGVI ( Stochastic Gradient Variational Inference)

- build flexible posterior distribution using "iterative procedure"



$\mathrm{z}_{0} \sim q\left(\mathrm{z}_{0} \mid \mathrm{x}\right), \quad \mathrm{z}_{t}=\mathrm{f}_{t}\left(\mathrm{z}_{t-1}, \mathrm{x}\right) \quad \forall t=1 \ldots T$

$\log q\left(\mathbf{z}_{T} \mid \mathbf{x}\right)=\log q\left(\mathbf{z}_{0} \mid \mathbf{x}\right)-\sum_{t=1}^{T} \log \operatorname{det}\left|\frac{d \mathbf{z}_{t}}{d \mathbf{z}_{t-1}}\right|$

$\mathbf{f}_{t}\left(\mathbf{z}_{t-1}\right)=\mathbf{z}_{t-1}+\mathbf{u} h\left(\mathbf{w}^{T} \mathbf{z}_{t-1}+b\right)$

- $\mathbf{u} h\left(\mathbf{w}^{T} \mathbf{z}_{t-1}+b\right)$ can be interpreted as a MLP with "bottleneck hidden layer" with a single unit

  $\rightarrow$ problem : long chain of transform is needed in high-dimension



# 4. Inverse Autoregressive Transformations

for NF that scales well to high-dimensional space...

- consider Gaussian version of autoregressive AE

  ( ex. MADE, PixeCNN )



Notation

- $[\boldsymbol{\mu}(\mathbf{y}), \boldsymbol{\sigma}(\mathbf{y})]$ :  function of the vector $\mathrm{y},$ to the vectors $\mu$ and $\sigma $

- $\left[\mu_{i}\left(\mathbf{y}_{1: i-1}\right), \sigma_{i}\left(\mathbf{y}_{1: i-1}\right)\right]$ : predicted mean and standard deviation of the $i$ -th element of $\mathrm{y}$ 

  ( Due to the autoregressive structure, Jacobian is lower triangular with zeros on the diagonal:

   $\partial\left[\boldsymbol{\mu}_{i}, \boldsymbol{\sigma}_{i}\right] / \partial \mathbf{y}_{j}=[0,0]$ for $j \geq i$   )

- $\epsilon \sim \mathcal{N}(0, \mathrm{I})$ : sample from noise vector 
- $\mathrm{y}$
  - $y_{0}=\mu_{0}+\sigma_{0} \odot \epsilon_{0}$
  - $y_{i}=\mu_{i}\left(\mathrm{y}_{1: i-1}\right)+\sigma_{i}\left(\mathrm{y}_{1: i-1}\right) \cdot \epsilon_{i}$     where $i>0$



Computation involved in this transformation : proportional to dimension $D$ 

- but inverse transformation is interesting in NF!

  $\epsilon_{i}=\frac{y_{i}-\mu_{i}\left(\mathbf{y}_{1: i-1}\right)}{\sigma_{i}\left(\mathbf{y}_{1: i-1}\right)}$



2 key observations

- 1) inverse transformation can be parallelized : $\boldsymbol{\epsilon}=(\mathbf{y}-\boldsymbol{\mu}(\mathbf{y})) / \boldsymbol{\sigma}(\mathbf{y})$

  ( individual element $\epsilon_i$ do not depend on each other! )

- 2) inverse autoregressive operation has a simple Jacobian determinant

  due to the autoregressive structure ($\partial\left[\mu_{i}, \sigma_{i}\right] / \partial y_{j}=[0,0]$ for $j \geq i$ )

  *  $\partial \epsilon_{i} / \partial y_{j}=0$ for $j>i$
  * $\partial \epsilon_{i} / \partial y_{i}=\sigma_{i}$ ( simple diagonal )

  Thus, $\log \operatorname{det}\left|\frac{d \boldsymbol{\epsilon}}{d \mathbf{y}}\right|=\sum_{i=1}^{D}-\log \sigma_{i}(\mathbf{y})$

  

# 5. Inverse Autoregressive Flow (IAF)

based on $\boldsymbol{\epsilon}=(\mathbf{y}-\boldsymbol{\mu}(\mathbf{y})) / \boldsymbol{\sigma}(\mathbf{y})$  ( Inverse Autoregressive Transformations )

![image-20201227170536352](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201227170536352.png)



![image-20201227170544562](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201227170544562.png)



chain of $T$ : $\mathbf{z}_{t}=\boldsymbol{\mu}_{t}+\boldsymbol{\sigma}_{t} \odot \mathbf{z}_{t-1}$



autoregressive w.r.t. $\mathrm{z}_{t-1}$

- Jacobians $\frac{d \mu_{t}}{d \mathbf{z}_{t-1}}$ and $\frac{d \sigma_{t}}{d \mathbf{z}_{t-1}}$ are triangular with zeros on the diagonal. 

  Thus, $\frac{d \mathbf{z}_{t}}{d \mathbf{z}_{t-1}}$ is triangular with $\sigma_{t}$ on the diagonal, with determinant $\prod_{i=1}^{D} \sigma_{t, i}$

  

$\log q\left(\mathbf{z}_{T} \mid \mathbf{x}\right)=-\sum_{i=1}^{D}\left(\frac{1}{2} \epsilon_{i}^{2}+\frac{1}{2} \log (2 \pi)+\sum_{t=0}^{T} \log \sigma_{t, i}\right)$



Autoregressive Neural Network

$\begin{aligned}
\left[\mathbf{m}_{t}, \mathbf{s}_{t}\right] & \leftarrow \text { AutoregressiveNN }[t]\left(\mathbf{z}_{t}, \mathbf{h} ; \boldsymbol{\theta}\right) \\
\boldsymbol{\sigma}_{t} &=\operatorname{sigmoid}\left(\mathbf{s}_{t}\right) \\
\mathbf{z}_{t} &=\boldsymbol{\sigma}_{t} \odot \mathbf{z}_{t-1}+\left(1-\boldsymbol{\sigma}_{t}\right) \odot \mathbf{m}_{t}
\end{aligned}$

Autoregressive NN form a rich family of nonlinear transformations for IAF!



# 6. Summary

( by Coursera )



### Inverse Autoregressive Flow (IAF)
The inverse autoregressive flow reverses the dependencies to make the forward pass parallelisable, but the inverse pass sequential. 


It uses the same equations:

$$ x_i = \mu_i + \exp(\sigma_i) z_i \quad \quad i=1, \ldots, D$$

but has the scale and shift functions depend on the $z_i$ instead of the $x_i$:

$$ \mu_i = f_{\mu_i}(z_1, \ldots, z_{i-1}) \quad \quad \sigma_i = f_{\sigma_i}(z_1, \ldots, z_{i-1}).$$

Note that now the forward equation (determining $\mathbf{x}$ from $\mathbf{z}$) can be parallelised, but the reverse transformations require determining $z_1$, followed by $z_2$, etc. and must hence be solved in sequence.