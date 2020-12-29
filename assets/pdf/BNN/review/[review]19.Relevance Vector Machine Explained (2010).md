## [ Paper review 19 ]

# Relevance Vector Machine Explained

### ( Tristan Fletcher, 2010 )



## [ Contents ]

1. Introduction

2. RVM for Regression

3. Analysis of Sparsity

4. RVM for Classification



# 1. Introduction

SVM 

- Not a probabilistic prediction
- Only Binary decision
- have to tune hyperparameter $C$



RVM is more sparse, and can solve three problems above.



# 2. RVM for Regression

RVM = Linear Model + Modified prior for sparse solution



## 2.1 Model setup

1) conditional distribution : $p(t \mid x, w, \beta)=N(t \mid y(x), 1 / \beta)$

2) prior :

- ( LM )     $p\left(w_{i}\right)=N\left(0,1 / \alpha\right)$
- ( RVM )   $p\left(w_{i}\right)=N\left(0,1 / \alpha_{i}\right)$

3) posterior :  $p(w \mid t, X, \alpha, \beta)=N(w \mid m, \Sigma)$ , where

- $m=\beta \Sigma \Phi^{T} t$
- $\Sigma=\left(A+\beta \Phi^{T} \Phi\right)^{-1}$   ( where $A = diag(a_i)$  )



## 2.2 Maximize Marginal Likelihood

- find optimal $\alpha$ and $\beta$ by maximizing marginal likelihood, $p(t \mid X, \alpha, \beta)$

 $\begin{aligned}p(t \mid X, \alpha, \beta)&=\int p(t \mid X, w, \beta) p(w \mid \alpha) d w \\ &=\int N\left(t \mid w^{\top} \phi(x), \frac{1}{\beta}\right) N\left(w \mid 0, A^{-1}\right) d w \\ &=N(\boldsymbol{t} \mid 0, C) \end{aligned}$

​	where $C=\beta^{-1} I+\boldsymbol{\Phi} \mathrm{A}^{-1} \boldsymbol{\Phi}^{T}$



#### Woodbury identity

$$
\left(A+B D^{-1} C\right)^{-1}=A^{-1}-A^{-1} B\left(D+C A^{-1} B\right)^{-1} C A^{-1}
$$
In our case, $A=\beta^{-1} I, B=\Phi, D=A, C=\Phi^{T}$



If we solve...

- $\frac{\partial}{\partial \alpha} p(t \mid X, \alpha, \beta)=0$
- $\frac{\partial}{\partial \beta} p(t \mid X, \alpha, \beta)=0$



Solution :

$\begin{array}{l}
\alpha_{i}^{\text {new}}=\frac{r_{i}}{m_{i}^{2}}=\left(1-\alpha_{i} \Sigma_{i i}\right) / m_{i}^{2}: \text { implicit } \\
\left(\beta^{\text {Nex}}\right)^{-1}=\|t-\boldsymbol{\Phi} m\|^{2} /\left(N-\sum_{i} r_{i}\right)
\end{array}$

- can not solve $\alpha_{i}^{\text {new}}$ directly...

- step 1) initialize $\alpha_0$ and $\beta_0$

  step 2) find posterior

  step 3) update $\alpha$ and $\beta$

  step 4) repeat step 2 \& 3



#### Relevance Vector

- data(vector) with non-zero weight

- $\alpha_i \approx \infty$  , $w_i = 0$



#### RVM vs SVM

1) Sparsity : RVM > SVM

2) Generalization : RVM > SVM

3) Need to estimate hyperparameter : only SVM

4) Training Time : RVM>>SVM



# 3. Analysis of Sparsity

Alternative way to train RVM, due to long training time.

[ Log Marginal Likelihood ($L(\alpha) $ ) ]

$L(\alpha) = L(\alpha_i) + \lambda(\alpha_i)$



$L(\alpha) = \ln (p(\boldsymbol{t} \mid \boldsymbol{X}, \boldsymbol{\alpha}, \beta))=\ln (N(\boldsymbol{t} \mid 0, C))$

where $C=\beta^{-1} I+\sum_{j \neq i} \alpha_{j}^{-1} \phi_{j} \phi_{j}^{T}+\alpha_{i}^{-1} \phi_{i} \phi_{i}^{T} = C_{-i}+\alpha_{i}^{-1} \phi_{i} \phi_{i}^{T} $



Solution :

$\begin{array}{l}
\lambda\left(\boldsymbol{\alpha}_{i}\right)=\frac{1}{2}\left\{\ln \left(\left|\mathbf{1}+\alpha_{i}^{-\mathbf{1}} \phi_{i}^{T} \boldsymbol{C}_{-i}^{-\mathbf{1}} \phi_{i}\right|\right)-\boldsymbol{t}^{T}\left(\frac{C_{-i}^{-1} \phi_{i} \phi_{i}^{T} C_{-i}^{-1}}{\alpha_{i}+\phi_{i}^{T} c_{-i}^{-1} \phi_{i}}\right) \boldsymbol{t}\right\}
\end{array}$



$s_{i}=\phi_{i}^{T} C_{-i}^{-1} \phi_{i}$

- sparsity of  $\phi_{i}$
- overlap between $\phi_i$ and $\phi_j$



$q_{i}=\phi_{i}^{T} C_{-i}^{-1} t$

- quality of  $\phi_{i}$
- $C_{-i}^{-1} t$ : prediction error $\rightarrow q_i : $ information about $\phi_i$



$s_i > q_i$ $\rightarrow $ $\phi_i=0$

$s_i < q_i$ $\rightarrow $ $\phi_i \neq 0$



$\frac{\partial}{\partial \alpha} L(\alpha) = \frac{\partial}{\partial \alpha} \lambda(\alpha) =\frac{\alpha_i^{-1}s_i^2 + s_{i}-q_i^2}{(\alpha_i + s_i)^2}0 $

- if $s_i \leq q_i^2$ $\rightarrow$ $\alpha_i = \frac{s_i^2}{s_i - q_i^2}$
- if $s_i > q_i^2$ $\rightarrow$ $\alpha_i = \infty$



# 4. RVM for Classification

Binary case

- $y(x)=\sigma\left(w^{T} \varphi(x)\right) $



Multi-class case

- $y_{K}(x)=\frac{\exp \left(w_{K}^{T} \varphi(x)\right)}{\sum_{j} \exp \left(w_{j}^{T} \varphi(x)\right)}$



Hard to calculate marginal likelihood, so use Laplace Approximation