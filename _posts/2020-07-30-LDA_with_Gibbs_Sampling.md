---
title: LDA with Collapsed Gibbs Sampling
categories: [ML,STAT]
tags: [LDA, Gibbs sampling]
excerpt: LDA with Collapsed Gibbs Sampling
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# LDA with Collapsed Gibbs Sampling

## 1. LDA

​	LDA는 토픽 모델링(Topic Modeling)의 방법 중 하나로, 주어진 문서(document)에 적절한 주제(topic)를 파악하는 방법이다. 기본적으로, **"문서는 Topic들의 혼합으로 구성있고, 각각의 Topic은 확률 분포에 기반하여 단어를 생성"** 한다는 가정을 하고 있다. 쉽게 말해, 다음과 같은 생각으로 문서가 작성되는 것이다.

*"나는 문서 작성을 위해서, 이러한 주제들을 넣을 것이고,*
*이런 주제들을 이루기 위해서 이런 단어들을 넣을 것이야!"*

즉, LDA의 핵심은 주어진 데이터에 대해서 '문서가 생성되는 과정을 역추적'하는 것이라고 볼 수 있다!





## 2. Parameters

<img src="https://shuyo.files.wordpress.com/2011/05/smoothed_lda.png" width="700" />

https://shuyo.files.wordpress.com/2011/05/smoothed_lda.png



LDA를 설명하는 변수들은 위의 Bayesian Network 그림으로 간단하게 설명할 수 있다. 

( N : 문서 내 **단어의 개수** )

( M : 총 **문서의 개수** )

- $$\theta$$ : document-topic distribution ( 각 문서가 가지고 있는 **주제들에 대한 확률 분포** )
- $$\Z$$ : latent variable ( 잠재 변수로써, 각 단어의 주제를 의미한다 )
- $$W$$ : 단어 ( 우리가 관찰할 수 있는 evidence이다 )
- $$\phi$$ :  word-topic distribution ( 주제들이 가지고 있는 **단어들에 대한 확률 분포** )
- $$\alpha$$ & $$\beta$$ :  각각 $$\theta$$ & $$\phi$$에 대한 prior distribution **( Dirichlet Distribution )**
  - $$\theta_i \sim Dir(\alpha)$$
  - $$\phi_k \sim Dir(\beta)$$

이 모델의 핵심은, 가장 최적의 $$Z$$를 할당하는 것에 있다!



## 3. Collapsed Gibbs Sampling in LDA

앞어 말했듯, 이 모델에서는 가장 적절한 주제($$Z$$)를 할당하는 것이 중요하다, 그러기 위해 **Gibbs Sampling**을 사용할 것이다 ( Gibbs Sampling에 대한 자세한 설명은 생략한다 )



우선, 우리는 prior $$\alpha$$ 와 $$\beta$$가 주어졌을 때의 확률을 다음과 같이 factorize하여 표현할 수 있다.

( 여기서 $$i$$는 몇 번째 주제인지, $$j$$는 몇 번째 문서인지, $$l$$은 몇 번째 단어인지를 의미한다 )

$$P(W,Z,\theta, \phi ; \alpha, \beta) = \prod_{i=1}^{K}P(\phi_i ; \beta) \prod_{j=1}^{M}P(\theta_j ; \alpha) \prod_{l=1}^{N}P(Z_{j,l} \mid \theta_j) P(W_{j,l} \mid \phi_{Z_{j,l}})$$



위 식에서, 우리는 우리의 관심 대상인 $$W$$(단어) 와 $$\Z$$(주제)와, prior인 $$\alpha$$,$$\beta$$를 제외하고 나머지를 collapse할 것이다.  ( 즉, $$\theta$$ 와 $$\phi$$를 없앨 것이다 ) 이를 정리하면 다음과 같다.

$$\begin{align*} P(W,Z ; \alpha, \beta) &= \int_{\theta} \int_{\phi} P(W,Z,\theta, \phi ; \alpha, \beta)d\phi d\theta\\
&=\int_{\theta} \int_{\phi}  \prod_{i=1}^{K}P(\phi_i ; \beta) \prod_{j=1}^{M}P(\theta_j ; \alpha) \prod_{l=1}^{N}P(Z_{j,l} \mid \theta_j) P(W_{j,l} \mid \phi_{Z_{j,l}})d\phi d\theta\\  
&= \int_{\phi} \prod_{i=1}^{K}P(\phi_i ; \beta) \prod_{j=1}^{M} \prod_{l=1}^{N} P(W_{j,l} \mid \phi_{Z_{j,l}})d\phi \;\; \times  \int_{\theta} \prod_{j=1}^{M}P(\theta_j ; \alpha) \prod_{l=1}^{N}P(Z_{j,l} \mid \theta_j) d\theta \\ &= (1) \times (2)
\end{align*}$$



**우선 (1)을 먼저 정리해보자.**

$$\begin{align*}&\int_{\phi} \prod_{i=1}^{K}P(\phi_i ; \beta) \prod_{j=1}^{M} \prod_{l=1}^{N} P(W_{j,l} \mid \phi_{Z_{j,l}})d\phi \\&= \prod_{i=1}^{K}\int_{\phi_i} P(\phi_i ; \beta) \prod_{j=1}^{M} \prod_{l=1}^{N} P(W_{j,l} \mid \phi_{Z_{j,l}})d\phi \\ 
&= \prod_{i=1}^{K}\int_{\phi_i} \frac{\Gamma(\sum_{v=1}^{V})}{\prod_{v=1}^V \Gamma(\beta_v)} \prod_{v=1}^V \phi_{i,v}^{\beta_v-1}\prod_{j=1}^{M} \prod_{l=1}^{N} P(W_{j,l} \mid \phi_{Z_{j,l}})d\phi  \\
&=  \prod_{i=1}^{K}\int_{\phi_i} \frac{\Gamma(\sum_{v=1}^{V})}{\prod_{v=1}^V \Gamma(\beta_v)} \prod_{v=1}^V \phi_{i,v}^{\beta_v-1}\prod_{v=1}^{V}\phi_{i,v}^{n_{(.),v}^i}d\phi_i \\
&= \prod_{i=1}^{K}\int_{\phi_i} \frac{\Gamma(\sum_{v=1}^{V})}{\prod_{v=1}^V \Gamma(\beta_v)} \prod_{v=1}^V \phi_{i,v}^{n_{(.),v}^i+\beta_v-1}d\phi_i 
\end{align*}$$

( 위 식에서 새로 도입한 $$n_{j,r}^i$$는, $$j$$번째 문서의 $$r$$종류의 단어에 할당된 $$i$$번째 주제와 관련 된 단어 수를 의미한다. $$v$$는 모든 단어 ( 사전에 등재된 모든 단어 )의 개수라고 보면 된다.  )



위 식을 이어서 정리하면

$$\begin{align*}&= \prod_{i=1}^{K}
\frac{\prod_{v=1}^{V}\Gamma(n_{(.),v}^i+\beta_v)\Gamma(\sum_{v=1}^{V}\beta_v)}{\prod_{v=1}^{V}\Gamma(\beta_v)\Gamma(\sum_{v=1}^{V}n_{(.),v}^i+\beta_v)}
\int_{\phi_i} \frac{\Gamma(\sum_{v=1}^{V} n_{(.),v}^i+\beta_v )}{\prod_{v=1}^V \Gamma(n_{(.),v}^i+\beta_v)} \prod_{v=1}^V \phi_{i,v}^{n_{(.),v}^i+\beta_v-1}d\phi_i \\
&= \prod_{i=1}^{K}\frac{\prod_{v=1}^{V}\Gamma(n_{(.),v}^i+\beta_v)\Gamma(\sum_{v=1}^{V}\beta_v)}{\prod_{v=1}^{V}\Gamma(\beta_v)\Gamma(\sum_{v=1}^{V}n_{(.),v}^i+\beta_v)} \times 1\\
&= \prod_{i=1}^{K}\frac{\prod_{v=1}^{V}\Gamma(n_{(.),v}^i+\beta_v)\Gamma(\sum_{v=1}^{V}\beta_v)}{\prod_{v=1}^{V}\Gamma(\beta_v)\Gamma(\sum_{v=1}^{V}n_{(.),v}^i+\beta_v)}
\end{align*}$$



**(2)도 위와 같은 방식으로 정리하면 다음과 같은 결과가 나온다 (생략)**

(2) = $$\prod_{j=1}^{M}\frac{\prod_{i=1}^{K}\Gamma(n_{j,(.)}^i+\alpha_k)\Gamma(\sum_{i=1}^{K}\alpha_v)}{\prod_{j=1}^{M}\Gamma(\alpha_i)\Gamma(\sum_{i=1}^{K}n_{j,(.)}^i+\alpha_k)}$$



위에서 (1)과 (2)를 각각 정리하였다. 이를 다시 대입하면....

$$\begin{align*} P(W,Z ; \alpha, \beta) &=  (1) \times (2) \\
&= \prod_{i=1}^{K}\frac{\prod_{v=1}^{V}\Gamma(n_{(.),v}^i+\beta_v)\Gamma(\sum_{v=1}^{V}\beta_v)}{\prod_{v=1}^{V}\Gamma(\beta_v)\Gamma(\sum_{v=1}^{V}n_{(.),v}^i+\beta_v)} \times \prod_{j=1}^{M}\frac{\prod_{i=1}^{K}\Gamma(n_{j,(.)}^i+\alpha_k)\Gamma(\sum_{i=1}^{K}\alpha_v)}{\prod_{j=1}^{M}\Gamma(\alpha_i)\Gamma(\sum_{i=1}^{K}n_{j,(.)}^i+\alpha_k)} \end{align*}$$



이제 위의 정리된 식에서, 우리는 Gibbs Sampling을 시행할 것이다. 나머지 $$Z$$의 요소들과, $$W,\alpha,\beta$$를 모두 고정시킨 채로, $$Z$$의 하나의 element( $$Z_{(m,l)}$$ , $$m$$번째 문서의 $$l$$번째 단어의 주제 $$Z$$ ) 를 sampling할 것이다. 해당 식은 아래와 같이 표현할 수 있다.

$$\begin{align*} 
P(Z_{(m,l)} = k \mid Z_{-(m,l)},W ; \alpha, \beta) &= \frac{P(Z_{(m,l)}=k, Z_{-(m,l)},W ; \alpha, \beta)}{P(Z_{-(m,l)},W ; \alpha, \beta)}\\
&\propto P(Z_{(m,l)}=k, Z_{-(m,l)},W ; \alpha, \beta) \\
&= \prod_{i=1}^{K}\frac{\Gamma(n_{(.),v}^i+\beta_v)}{\Gamma(\sum_{r=1}^Vn_{(.),r}^i + \beta_r)}\times \frac{\prod_{i=1}^K \Gamma(n_{m,(.)}^i+\alpha_i)}{\Gamma(\sum_{i=1}^K n_{m,(.)}^i + \alpha_i)} \\
&\propto \prod_{i=1}^{K}\frac{\Gamma(n_{(.),v}^i+\beta_v)}{\Gamma(\sum_{r=1}^Vn_{(.),r}^i + \beta_r)} \times \prod_{i=1}^K \Gamma(n_{m,(.)}^i+\alpha_i)\\\end{align*}$$



위 식을 이어서 정리하면..

$$\begin{align*}
&\propto \prod_{i=1,i \neq k}^{K}\frac{\Gamma(n_{(.),v}^{i,-(m,n)}+\beta_v)}{\Gamma(\sum_{r=1}^Vn_{(.),r}^{i,-(m,n)} + \beta_r)} \times \prod_{i=1,i\neq k}^K \Gamma(n_{m,(.)}^{i,-(m,n)}+\alpha_i) \\ 
&\times \frac{\Gamma(n_{(.),v)}^{k,-(m,n)}+\beta_v)}{\Gamma(\sum_{r=1}^V n_{(.),r}^{k,-(m,n)}+\beta_v)}\Gamma(n_{m,(.)}^{k,-(m,n)}+\alpha_k) \times \frac{n_{(.),v}^{k,-(m,n)}+\beta_k}{\sum_{r=1}^Vn_{(.),r}^{k,-(m,n)}+\beta_r}(n_{m,(.)}^{k,(m,n)}+\alpha_k)\\
&\propto \prod_{i=1}^{K}\frac{\Gamma(n_{(.),v}^{i,-(m,n)}+\beta_v)}{\Gamma(\sum_{r=1}^Vn_{(.),r}^{i,-(m,n)} + \beta_r)} \times \prod_{i=1}^K \Gamma(n_{m,(.)}^{i,-(m,n)}+\alpha_i) \times \frac{n_{(.),v}^{k,-(m,n)}+\beta_v} {\sum_{r=1}^V n_{(.),r}^{k,-(m,n)}+\beta_r}(n_{m,(.)}^{k,-(m,n)}+\alpha_k) \\
&\propto \frac{n_{(.),v}^{k,-(m,n)}+\beta_v} {\sum_{r=1}^V n_{(.),r}^{k,-(m,n)}+\beta_r}(n_{m,(.)}^{k,-(m,n)}+\alpha_k) \\
\end{align*}$$



따라서, $$P(Z_{(m,l)} = k \mid Z_{-(m,l)},W ; \alpha, \beta) \propto \frac{n_{(.),v}^{k,-(m,n)}+\beta_v} {\sum_{r=1}^V n_{(.),r}^{k,-(m,n)}+\beta_r}(n_{m,(.)}^{k,-(m,n)}+\alpha_k)$$ 이다!



## 4. Summary

지금까지의 LDA 과정을 요약하면 다음과 같다

- input : Text Corpus, $$\alpha, \beta%%
- 알고리즘:
  - 1 ) 임의로 Z를 initialize한다
  - 2 ) $$n_{j,r}^i$$를 계산한다 ( $$j$$번째 문서의 $$r$$종류의 단어에 할당된 $$i$$번째 주제와 관련 된 단어 수 )
  - 3 ) 수렴할 때 까지...
    - for $$m = 1$$ ... 
      - for $$l=1$$ ...
        - 다음 식으로 부터 $$k$$를 sample하고 $$P(Z_{(m,l)} = k \mid Z_{-(m,l)},W ; \alpha, \beta) \propto \frac{n_{(.),v}^{k,-(m,n)}+\beta_v} {\sum_{r=1}^V n_{(.),r}^{k,-(m,n)}+\beta_r}(n_{m,(.)}^{k,-(m,n)}+\alpha_k)$$
        - $$n_{j,r}^{i}$$를 조정한다
  - 4 ) $$\theta, \phi$$를 estimate한다



