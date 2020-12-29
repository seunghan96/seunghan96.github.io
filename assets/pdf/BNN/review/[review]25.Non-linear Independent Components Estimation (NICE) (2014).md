## [ Paper review 25 ]

# Non-linear Independent Components Estimation (NICE)

### ( Laurent Dinh, et al, 2014 )



## [ Contents ]

1. Abstract
2. Introduction
   1. Variable transformation
   2. Key Point
3. Learning Bijective Transformations of Continuous Probabilites
4. Architecture
   1. Triangular structure
   2. Coupling layer
   3. Allowing rescaling
   4. Prior distributions



# 1. Abstract

propose NICE

- for modeling complex high-dimensional densities
- based on the idea that "good representation = distribution that is easy to model"



Key point

- 1) computing the determinant of Jacobian & inverse Jacobian is trivial
- 2) still learn complex non-linear transformations ( with composition of simple blocks )



# 2. Introduction

## 2.1 Variable transformation

$p_{X}(x)=p_{H}(f(x))\left|\operatorname{det} \frac{\partial f(x)}{\partial x}\right|$

- $ \frac{\partial f(x)}{\partial x}$ : Jacobian matrix of function $f$ at $x$



transformation $f$'s properties

- 1) easy determinant of Jacobian
- 2) easy inverse



## 2.2 Key point

split $x$ into 2 blocks $(x_1,x_2)$

$\begin{array}{l}
y_{1}=x_{1} \\
y_{2}=x_{2}+m\left(x_{1}\right)
\end{array}$

- $m$ : arbitrarily complex function



inverse :

$\begin{array}{l}
x_{1}=y_{1} \\
x_{2}=y_{2}-m\left(y_{1}\right)
\end{array}$



# 3. Learning Bijective Transformations of Continuous Probabilities

$\log \left(p_{X}(x)\right)=\log \left(p_{H}(f(x))\right)+\log \left(\left|\operatorname{det}\left(\frac{\partial f(x)}{\partial x}\right)\right|\right)$

- $p_{H}(h)$ : prior distribution

  ( ex. isotropic Gaussian )

  ( does not need to be constant, could also be learned )



if prior is factorial.... we obtain the following "NICE criterion"

$\log \left(p_{X}(x)\right)=\sum_{d=1}^{D} \log \left(p_{H_{d}}\left(f_{d}(x)\right)\right)+\log \left(\left|\operatorname{det}\left(\frac{\partial f(x)}{\partial x}\right)\right|\right)$, where $f(x)=\left(f_{d}(x)\right)_{d \leq D}$



Auto-encoders

- $f$ : encoder
- $f^{-1}$ : decoder



# 4. Architecture

## 4.1 Triangular Structure

obtain a family of bijections

- 1) whose Jacobian determinant is tractable
- 2) whose computation is straight forward



Jacobian determinant is the product of its layer's Jacobian determinants

$f=f_{L} \circ \ldots \circ f_{2} \circ f_{1}$



affine transformations

-  inverse & determinant when using diagonal matrices



$M=LU$

- $M$ : square matrices 
- $L$, $U$ : upper and lower triangular matrices



HOW?

- method 1) build a NN with traingular weights..

  $\rightarrow$ constrained.....

- method 2) consider a family of functions with "triangular Jacobians"



## 4.2 Coupling Layer

(1) bijective transformation

(2) triangular Jacobian

(1)+(2) = "tractable Jacobian determinant"



### General Coupling layer

![image-20201220164437424](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201220164437424.png)



$\begin{array}{l}
y_{I_{1}}=x_{I_{1}} \\
y_{I_{2}}=g\left(x_{I_{2}} ; m\left(x_{I_{1}}\right)\right)
\end{array}$

thus, $\frac{\partial y}{\partial x}=\left[\begin{array}{cc}
I_{d} & 0 \\
\frac{\partial y_{I_{2}}}{\partial x_{I_{1}}} & \frac{\partial y_{I_{2}}}{\partial x_{I_{2}}}
\end{array}\right]$,   and    $\operatorname{det} \frac{\partial y}{\partial x}=\operatorname{det} \frac{\partial y_{I_{2}}}{\partial x_{I_{2}}}$



inverse

$\begin{array}{l}
x_{I_{1}}=y_{I_{1}} \\
x_{I_{2}}=g^{-1}\left(y_{I_{2}} ; m\left(y_{I_{1}}\right)\right)
\end{array}$



### Additive Coupling Layer

$g\left(x_{I_{2}} ; m\left(x_{I_{1}}\right)\right) = x_{I_{2}}+m\left(x_{I_{1}}\right)$



That is...

$$\begin{array}{l}
y_{I_{2}}=x_{I_{2}}+m\left(x_{I_{1}}\right) \\
x_{I_{2}}=y_{I_{2}}-m\left(y_{I_{1}}\right)
\end{array}$$

thus, $\frac{\partial y}{\partial x}=\left[\begin{array}{cc}
I_{d} & 0 \\
\frac{\partial y_{I_{2}}}{\partial x_{I_{1}}} & \frac{\partial y_{I_{2}}}{\partial x_{I_{2}}}
\end{array}\right]$,   and    $\operatorname{det} \frac{\partial y}{\partial x}=\operatorname{det} \frac{\partial y_{I_{2}}}{\partial x_{I_{2}}}=1$



### Combining Coupling Layers



## 4.3 Allowing Rescaling

each additive coupling layers has unit Jacobian determinant ( = volume preserving )

$\rightarrow$ lets include "diagonal scaling matrix $S$"



allows the learner to give more weight on some dimension!

( low $S_{ii}$, less important latent variable $z_i$ )



Then, NICE criterion :

- $\log \left(p_{X}(x)\right)=\sum_{i=1}^{D}\left[\log \left(p_{H_{i}}\left(f_{i}(x)\right)\right)+\log \left(\left|S_{i i}\right|\right)\right]$



## 4.4 Prior distributions

factorized distributions :  $p_{H}(h)=\prod_{d=1}^{D} p_{H_{d}}\left(h_{d}\right)$

- Gaussian :

  $\log \left(p_{H_{d}}\right)=-\frac{1}{2}\left(h_{d}^{2}+\log (2 \pi)\right)$

- Logistic :

  $\log \left(p_{H_{d}}\right)=-\log \left(1+\exp \left(h_{d}\right)\right)-\log \left(1+\exp \left(-h_{d}\right)\right)$