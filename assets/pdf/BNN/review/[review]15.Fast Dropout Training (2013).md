## [ Paper review 15 ]

# Fast Dropout Training

### ( Sida I. Wang, Christopher D. Manning, 2013 )



## [ Contents ]

0. Abstract
1. Introduction
2. Fast approximations to dropout
   3. The implied objective function
   2. The Gaussian approximation
   3. Gradient computation by sampling from Gaussian
   4. A closed- form approximation



# 0. Abstract

- Dropout ( Hinton et al., 2012 )

  but, repeatedly sampling makes much slower!

- This paper shows how to do fast dropout training!

  "by sampling from, or integrating a GAUSSIAN APPROXIMATION" ( instead of doing MC optimization ) (by CLT)



# 1. Introduction

Dropout 

- prevent feature co-adaptation $\rightarrow$ regularization
- can be seen as "averaging over many NN with shared weights"



Problem with Dropout

- makes training SLOWER!

- loss of information

  ( drop out rate of $p$ : proportion of data still not seen after $n$ passes is $p^n$ )



This paper suggests "benefit of dropout training without actually sampling", thereby using ALL data efficiently

$\rightarrow$ use Gaussian Approximation



# 2. Fast approximations to dropout

## 2.1 The implied objective function

example ) Logistic Regression with dropout

- $m$ dimension data

- $z_{i} \sim$ Bernoulli $\left(p_{i}\right)$ , where $i=1 \ldots m $  
- SGD update : $\Delta w=\left(y-\sigma\left(w^{T} D_{z} x\right)\right) D_{z} x$
  - $D_{z}=\operatorname{diag}(z) \in \mathbb{R}^{m \times m}$
  - $\sigma(x)=1 /\left(1+e^{-x}\right)$
- MC approximation : $\Delta \bar{w}=E_{z ; z_{i} \sim \text { Bernoulli }\left(p_{i}\right)}\left[\left(y-\sigma\left(w^{T} D_{z} x\right)\right) D_{z} x\right]$



Objective function of gradient above :

- $y \sim \operatorname{Bernoulli}\left(\sigma\left(w^{T} D_{z} x\right)\right)$

  $\begin{array}{l}
  L(w)&=E_{z}\left[\log \left(p\left(y \mid D_{z} x ; w\right)\right]\right. \\
  &=E_{z}\left[y \log \left(\sigma\left(w^{T} D_{z} x\right)\right)+(1-y) \log \left(1-\sigma\left(w^{T} D_{z} x\right)\right)\right]
  \end{array}$

- complexity : $O(2^m m)$
- can be reduced to $O(m)$. .... HOW?



## 2.2 The Gaussian approximation

( now, let $Y(z)=w^{T} D_{z} x=\sum_{i}^{m} w_{i} x_{i} z_{i}$ ..... weighted sum of Bernoulli r.v )

$Y$ can be approximated by Normal distribution ( as $m \rightarrow \infty$ )

let $Y \stackrel{d}{\rightarrow} S$

$S=E_{z}[Y(z)]+\sqrt{\operatorname{Var}[Y(z)]} \epsilon=\mu_{S}+\sigma_{S} \epsilon$

- $\epsilon \sim \mathcal{N}(0,1)$ 
- $E_{z}[Y(z)]=\sum_{i=1}^{m} p_{i} w_{i} x_{i},$
- $\operatorname{Var}[Y(z)]=\sum_{i=1}^{m} p_{i}\left(1-p_{i}\right)\left(w_{i} x_{i}\right)^{2}$



## 2.3 Gradient computation by sampling from Gaussian

BEFORE) sample from $Y(z)$ directly

- time : $O(m)$
- d



AFTER) sample from $S$ 

- especially good for high dimensional case
- time : $O(1)$ ( $m$ times faster ! )



$L(w)=E_{z}\left[y \log \left(\sigma\left(w^{T} D_{z} x\right)\right)+(1-y) \log \left(1-\sigma\left(w^{T} D_{z} x\right)\right)\right]$

$\nabla L(w)=E_{z}\left[(y-\sigma(Y(z))) D_{z} x\right]$

- $f(Y(z))=y-\sigma(Y(z))$
- $g(z)=D_{z} x$ 



$\begin{aligned}
\nabla L(w)=& E_{z}\left[(y-\sigma(Y(z))) D_{z} x\right] \\
=& E_{z}\left[f(Y(z)) x_{i} z_{i}\right] \\
=& \sum_{z_{i} \in\{0,1\}} p\left(z_{i}\right) z_{i} x_{i} E_{z_{-i} \mid z_{i}}[f(Y(z))] \\
=& p\left(z_{i}=1\right) x_{i} E_{z_{-i} \mid z_{i}=1}[f(Y(z))] \\
\approx& p_{i} x_{i}\left(\begin{array}{c}
E_{S \sim \mathcal{N}\left(\mu_{S}, \sigma_{S}^{2}\right)}[f(S)]
+\left.\Delta \mu_{i} \frac{\partial E_{S \sim \mathcal{N}\left(\mu, \sigma_{S}^{2}\right)}[f(S)]}{\partial \mu}\right|_{\mu=\mu_{S}} 
+\left.\Delta \sigma_{i}^{2} \frac{\partial E_{S \sim \mathcal{N}\left(\mu_{S}, \sigma^{2}\right)}[f(S)]}{\partial \sigma^{2}}\right|_{\sigma^{2}=\sigma_{S}^{2}}
\end{array}\right) \\
=&p_{i} x_{i}\left(\alpha\left(\mu_{S}, \sigma_{S}^{2}\right)+\Delta \mu_{i} \beta\left(\mu_{S}, \sigma_{S}^{2}\right)+\Delta \sigma_{i}^{2} \gamma\left(\mu_{S}, \sigma_{S}^{2}\right)\right)
\end{aligned}$

- $\Delta \mu_{i}=\left(1-p_{i}\right) x_{i} w_{i}$ 
- $\Delta \sigma_{i}^{2}=-p_{i}\left(1-p_{i}\right) x_{i}^{2} w_{i}^{2}$ 



$\alpha, \beta, \gamma$ can be computed by drawing $K$ samples from $S$ $\rightarrow$ takes $O(K)$ time

( $\leftrightarrow$ if sample from $Y(z)$ , takes $O(mK)$ time! )

- $\alpha$ only need to be computed ONE per training case

- $\beta\left(\mu, \sigma^{2}\right)=\frac{\partial \alpha\left(\mu, \sigma^{2}\right)}{\partial \mu}$

  $\gamma\left(\mu, \sigma^{2}\right)=\frac{\partial \alpha\left(\mu, \sigma^{2}\right)}{\partial \sigma^{2}}$

- $\alpha\left(\mu_{S}, \sigma_{S}^{2}\right) = E_{S \sim \mathcal{N}\left(\mu_{S}, \sigma_{S}^{2}\right)}[f(S)]$  

  $\alpha\left(\mu, \sigma^{2}\right)=y-E_{S \sim \mathcal{N}(0,1)}\left[\frac{1}{1+e^{-\mu-\sigma_{S} S}}\right]$



$\begin{array}{l}
L(w)&=E_{z}\left[\log \left(p\left(y \mid D_{z} x ; w\right)\right]\right. \\
&=E_{z}\left[y \log \left(\sigma\left(w^{T} D_{z} x\right)\right)+(1-y) \log \left(1-\sigma\left(w^{T} D_{z} x\right)\right)\right]\\
&\approx E_{S \sim \mathcal{N}\left(\mu_{S}, \sigma_{S}\right)}[y \log (\sigma(S))+(1-y) \log (1-\sigma(S))]
\end{array}$



## 2.4 A closed- form approximation

$\Phi(x) $ : CDF of Gaussian ( $= \frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} e^{-t^{2} / 2} d t $)

$\sigma(x)$ : sigmoid (logistic) function

Since $\sigma(x) \approx \Phi(\sqrt{\pi / 8} x),$

- $\int_{-\infty}^{\infty} \Phi(\lambda x) \mathcal{N}(x \mid \mu, s) d x=\Phi\left(\frac{\mu}{\sqrt{\lambda^{-2}+s^{2}}}\right)$ 
- $\int_{-\infty}^{\infty} \sigma(x) \mathcal{N}\left(x \mid \mu, s^{2}\right) d x \approx \sigma\left(\frac{\mu}{\sqrt{1+\pi s^{2} / 8}}\right)$



Apply the above to our case ...

$\begin{array}{l}
E_{X \sim \mathcal{N}\left(\mu, s^{2}\right)}[\log (\sigma(X))]&=\int_{-\infty}^{\infty} \log (\sigma(x)) \mathcal{N}\left(x \mid \mu, s^{2}\right) d x \\
&\approx \sqrt{1+\pi s^{2} / 8} \log \sigma\left(\frac{\mu}{\sqrt{1+\pi s^{2} / 8}}\right)
\end{array}$





