## [ Paper review 27 ]

# Dennsity Estimation using Real NVP

### ( Laurent Dinh, et al., 2017 )



## [ Contents ]

1. Abstract
2. Introduction
3. Related Works
4. Model definition
   1. Change of variable formula
   2. Coupling layers
   3. Properties
   4. Masked Convolutions
   5. Combining coupling layers
   6. Multi-scale Architecture
   7. Batch Normalization



# 1. Abstract

real NVP = real-valued Non-Volumne Preserving transformation

- powerful, stably invertible, learnable transformations
- resulting in
  - exact log-likelihood computation
  - exact and efficient sampling
  - exact and efficient inference of latent variables
  - interpretable latent space



# 2. Introduction

representation learning

- have recently advanced thanks to supervised learning techniques
- unsupervised learning can also help!



one principle approach to unsupervised learning : "generative probabilistic modeling"

$\rightarrow$ but problem in high dimension



real NVP : tractable yet expressive approach to model high-dimensional data!



# 3. Related Works

lots of works (based on "generative probabilistic modeling" ) have focused on maximum likelihood

- probabilistic undirected graphs

  ( ex. RBM, DBM $\rightarrow$ due to intractability, used approximation like Mean Field Inference and MCMC )

- directed graphical models



# 4. Model definition

in this paper...

- learn highly nonlinear models in high-dimensional continuous spaces throguh ML
- use more flexible class of architectures ( using the change of variable formula )



## 4.1 change of variable formula

$\begin{aligned}
p_{X}(x) &=p_{Z}(f(x))\left|\operatorname{det}\left(\frac{\partial f(x)}{\partial x^{T}}\right)\right| \\
\log \left(p_{X}(x)\right) &=\log \left(p_{Z}(f(x))\right)+\log \left(\left|\operatorname{det}\left(\frac{\partial f(x)}{\partial x^{T}}\right)\right|\right)
\end{aligned}$



## 4.2 Coupling layers

computing the Jacobian with high-dimensional domain \& codomain : very expensive!

$\rightarrow$ "triangular matrix" ( both tractable and flexible )

$\begin{aligned}
y_{1: d} &=x_{1: d} \\
y_{d+1: D} &=x_{d+1: D} \odot \exp \left(s\left(x_{1: d}\right)\right)+t\left(x_{1: d}\right)
\end{aligned}$



## 4.3 Properties

Jacobian of transformation :

- $\frac{\partial y}{\partial x^{T}}=\left[\begin{array}{cc}
  \mathbb{I}_{d} & 0 \\
  \frac{\partial y_{d+1: D}}{\partial x_{1: d}^{T}} & \operatorname{diag}\left(\exp \left[s\left(x_{1: d}\right)\right]\right)
  \end{array}\right]$



we can efficiently compute its determinant as ...$\exp \left[\sum_{j} s\left(x_{1: d}\right)_{j}\right] .$

computing the inverse is no more complex than forward propagation!

( = meaning that sampling is as efficient as inference )



$\begin{aligned}
\left\{\begin{array}{l}
y_{1: d} &= x_{1: d} \\
y_{d+1: D} & =x_{d+1: D} \odot \exp \left(s\left(x_{1: d}\right)\right)+t\left(x_{1: d}\right)
\end{array}\right.\\
\Leftrightarrow\left\{\begin{array}{l}
x_{1: d}=y_{1: d} \\
x_{d+1: D}=\left(y_{d+1: D}-t\left(y_{1: d}\right)\right) \odot \exp \left(-s\left(y_{1: d}\right)\right)
\end{array}\right.
\end{aligned}$

  

## 4.4 Masked Convolution

$y=b \odot x+(1-b) \odot(x \odot \exp (s(b \odot x))+t(b \odot x))$

- partitioning using "binary mask $b$ "
  - 1 for the first half
  - 0 for the second half
- $s(\cdot)$ and $t(\cdot)$ are rectified CNN

![image-20201220194413344](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201220194413344.png)



## 4.5 Combining coupling layers

forward transformation leaves some components unchanged...

$\rightarrow$ can be overcome by "composing coupling layers"! still tractable

$\begin{aligned}
\frac{\partial\left(f_{b} \circ f_{a}\right)}{\partial x_{a}^{T}}\left(x_{a}\right) &=\frac{\partial f_{a}}{\partial x_{a}^{T}}\left(x_{a}\right) \cdot \frac{\partial f_{b}}{\partial x_{b}^{T}}\left(x_{b}=f_{a}\left(x_{a}\right)\right) \\
\operatorname{det}(A \cdot B) &=\operatorname{det}(A) \operatorname{det}(B)
\end{aligned}$

inverse : $\left(f_{b} \circ f_{a}\right)^{-1}=f_{a}^{-1} \circ f_{b}^{-1}$



## 4.6 Multi-scale Architecture

by using "squeezing operation"

for each channel...

- $s \times s \times c \rightarrow \frac{s}{2} \times \frac{s}{2} \times 4c$
- effectively trading "spatial size" for "number of channels"



Sequence of "coupling-squeezing-coupling"

$\begin{aligned}
h^{(0)} &=x \\
\left(z^{(i+1)}, h^{(i+1)}\right) &=f^{(i+1)}\left(h^{(i)}\right) \\
z^{(L)} &=f^{(L)}\left(h^{(L-1)}\right) \\
z &=\left(z^{(1)}, \ldots, z^{(L)}\right)
\end{aligned}$

![image-20201220194743551](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201220194743551.png)





## 4.7 Batch Normalization

use deep Resnet \& BN & WN

$x \mapsto \frac{x-\tilde{\mu}}{\sqrt{\tilde{\sigma}^{2}+\epsilon}} $ , has a Jacobian matrix $\left(\prod_{i}\left(\tilde{\sigma}_{i}^{2}+\epsilon\right)\right)^{-\frac{1}{2}}$



