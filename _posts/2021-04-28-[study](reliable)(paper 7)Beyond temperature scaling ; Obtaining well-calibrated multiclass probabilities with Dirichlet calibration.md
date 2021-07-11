---
title: \[reliable\] (paper 7) Beyond temperature scaling \: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: 
---

# Beyond temperature scaling : Obtaining well-calibrated multiclass probabilities with Dirichlet calibration

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Evaluation of Calibration & Temperature Scaling
   1. Multiclass-calibrated ( = calibrated )
   2. Classwise-calibrated
   3. Confidence-calibrated
2. Dirichlet Calibration
   1. Dirichlet calibration map family
   2. Interpretability
   3. Relationship to other families
   4. Fitting and ODIR regularization

<br>

# 0. Abstract

대부분의 multi-class classifier들은 **UNcalibrated** $$\rightarrow$$ OVERCONFIDENCE!

NN에서 자주 사용되는 방법으로, **"temperature scaling"**은 calibration을 improve시킨다!

<br>

이 논문에서는, **MULTI-class calibration** method를 제안함

- **(Dirichlet distribution에서 나온)** 그 어떠한 model class에 적용 가능
- Binary-class의 경우인 **beta-calibration의 general 버전**

- 아래의 지표들을 통해 성능 비교
  - **confidence ECE, classwise ECE, log loss, Brier score**

<br>

# 1. Evaluation of Calibration & Temperature Scaling

### Notation 소개

- probabilistic classifier : $$\hat{\mathbf{p}}: \mathscr{X} \rightarrow \Delta_{k}$$
  - output : class probabilities of $$k$$ classes
  - $$\Delta_{k}=\left\{\left(q_{1}, \ldots, q_{k}\right) \in[0,1]^{k} \mid \sum_{i=1}^{k} q_{i}=1\right\}$$.
- output : $$\hat{\mathbf{p}}(\mathbf{x})=\left(\hat{p}_{1}(\mathbf{x}), \ldots, \hat{p}_{k}(\mathbf{x})\right)$$

<br>

## 1-1) Multiclass-calibrated ( = calibrated )

$$P(Y=i \mid \hat{\mathbf{p}}(X)=\mathbf{q})=q_{i} \quad \text { for } i=1, \ldots, k$$.

- $$\mathbf{q}=\left(q_{1}, \ldots, q_{k}\right) \in \Delta_{k}$$.

- (key point) class 고려 X

<br>

## 1-2) Classwise-calibrated

$$P\left(Y=i \mid \hat{p}_{i}(X)=q_{i}\right)=q_{i}$$.

- (key point) **class 고려 O**

<br>

## 1-3) Confidence-calibrated

$$P(Y=\operatorname{argmax}(\hat{\mathbf{p}}(X)) \mid \max (\hat{\mathbf{p}}(X))=c)=c$$.

- for any $$c \in[0,1]$$

<br>

For practical evaluation, 위의 1-1) ~ 1-3)의 개념이 완화될 필요가 있다!

$$\rightarrow$$ **equal-width binning** ( + viz using **reliability diagram** )

<br>

## Temperature Scaling

$$S_{i}(x ; T)=\frac{\exp \left(f_{i}(x) / T\right)}{\sum_{j=1}^{N} \exp \left(f_{j}(x) / T\right)}$$.

- $$T$$ : temperature scaling parameter

  ( training 과정 중에는 $$T=1$$ )

- softmax score : $$S_{\hat{y}}(x ; T)=\max _{i} S_{i}(x ; T)$$

$$\rightarrow$$ calibration에 도움을 준다!

<br>

## Reliability Diagram

![figure2](/assets/img/reli/img13.png)

위의 그림을 통해 알 수 있는점?

- temperature-scaled 모델이

  - confidence-calibrated (O)
  - **classwise-calibrated (X)**

  하나의 tuneable parameter (= $$T$$ )만으로는, different class에 따른 고려를 못한다!

- [Figure 1-d]와 같이, classwise-calibration 수행하면 좋은 결과나옴!

  ( **classwise-ECE**를 measure로써 제안함 )

<br>

$$\text { classwise-ECE }=\frac{1}{k} \sum_{j=1}^{k} \sum_{i=1}^{m} \frac{ \mid B_{i, j} \mid }{n} \mid y_{j}\left(B_{i, j}\right)-\hat{p}_{j}\left(B_{i, j}\right) \mid $$.

- $$k,m,n$$ : classes / bins / instances의 수
- $$ \mid B_{i, j} \mid $$ : size of bin
- $$\hat{p}_{j}\left(B_{i, j}\right)$$ : average prediction of class $$j$$ probability in bin $$B_{i,j}$$
- $$y_{j}\left(B_{i, j}\right)$$ :  actual proportion of class $$j$$ in bin $$B_{i,j}$$
- **class-j-ECE** : contribution of single class $$j$$ to classwise-ECE

<br>

(기존 방법(class 고려 X)의 한계점들)

- input 무관하게, class별로 고르게 predict하면 좋은 calibrated probabilities 나오게 됨!

- 따라서, 새로운 evaluation metric이 필요하다!

  ( error rate말고도, proper losses 들을 사용해서 )

- ***proper losses***?

  - probabilistic predictions를 evaluate한다
  - **calibration loss** & **refinement loss**로 decompose 된다
  - (주로) objective functions in post-hoc calibration으로 사용됨
  - ex) Brier score, log-loss

- proper loss는 같은 **calibration map**으로 minimized 된다

  ( = **canonical calibration function** of $$\hat{\mathbf{p}})$$ 

  $$\mu(\mathbf{q})=(P(Y=1 \mid \hat{\mathbf{p}}(X)=\mathbf{q}), \ldots, P(Y=k \mid \hat{\mathbf{p}}(X)=\mathbf{q}))$$.

<br>

**Dirichlet calibration의 목적도 마찬가지로, 위의 canonical calibration map $$\mu$$를 estimate하는 것이다!**

<br>

# 2. Dirichlet Calibration

( Beta calibration : 2 class )

( Dirichlet calibration : multi-class )

<br>

## 2-1) Dirichlet calibration map family

distribution of prediction vectors $$\hat{\mathbf{p}}(\mathbf{x})$$ ( = $$k$$ class distribution )이 **Dirichlet distribution**이라고 가정한다.

$$\hat{\mathbf{p}}(X) \mid Y=j \sim \operatorname{Dir}\left(\alpha^{(j)}\right)$$.

- $$\alpha^{(j)}=\left(\alpha_{1}^{(j)}, \ldots, \alpha_{k}^{(j)}\right) \in(0, \infty)^{k}$$.

<br>

Canonical calibration function : $$P(Y \mid \hat{\mathbf{p}}(X))$$

- 위 function을 3가지로 표현할 수 있다.

<br>

### (1) Generative parameterization

- $$\hat{\mu}_{\text {DirGen }}(\mathbf{q} ; \alpha, \pi)=\left(\pi_{1} f_{1}(\mathbf{q}), \ldots, \pi_{k} f_{k}(\mathbf{q})\right) / z$$.

  ( where $$z=\sum_{j=1}^{k} \pi_{j} f_{j}(\mathbf{q})$$ is the normalizer )

<br>

### (2) Linear parameterization

- $$\hat{\mu}_{\text {DirLin }}(\mathbf{q} ; \mathbf{W}, \mathbf{b})=\sigma(\mathbf{W} \ln \mathbf{q}+\mathbf{b})$$.

  ( $$\mathbf{W} \in \mathbb{R}^{k \times k}$$ 는 $$k \times k$$ parameter 행렬 )

  ( $$\mathbf{b} \in \mathbb{R}^{k}$$ 는 length $$k$$의 parameter 벡터 ) 

<br>

### (3) Canonical parameterization

- $$\hat{\mu}_{D i r}(\mathbf{q} ; \mathbf{A}, \mathbf{c})=\sigma\left(\mathbf{A} \ln \frac{\mathbf{q}}{1 / k}+\ln \mathbf{c}\right)$$.

  ( $$\mathbf{A} \in[0, \infty)^{k \times k}$$ 는 $$k\times k$$ 행렬 )

  ( $$\mathbf{c} \in \mathbb{R}^{k}$$ 는 length $$k$$의 probability 벡터 ) 

<br>

### 위의 (1)~(3) 요약

- benefit of **(2) Linear parameterization**?

  $$\rightarrow$$ NN에 additional layer 추가함으로써 구현 가능하다

- **(3) Canonical parameterization**

  $$\rightarrow$$ **ANY function in Dirichlet** calibration map family can be **represented by a single pair of matrix $$\mathbf{A}$$ & vector $$\mathbf{c}$$**

<br>

## 2-2) Interpretability

Canonical parameterization은 interpretable하다!

- 우선, linear parameterization $$\rightarrow$$ Canonical parameterization으로의 변환은 쉽다

  - $$a_{i j}=w_{i j}-\min _{i} w_{i j}$$.
  - $$\mathbf{c}=\sigma(\mathbf{W} \ln \mathbf{u}+\mathbf{b})$$.
  - $$\mathbf{u}=(1 / k, \ldots, 1 / k) .$$.

- 해석 : $$A$$ 행렬의 요소  $$a_{i j}$$ 의 증가 

  = Increase the **calibrated probability of class $$i$$ **, with effect size depending on the **uncalibrated probability of class $$j .$$**

<br>

### Example

![figure2](/assets/img/reli/img14.png)

[Figure 2-b & 2-c]

- element $$a_{3,9}=0.63$$  : **increases class 2 probability** whenever **class 8 has high predicted probability**, 
  **modifying decision boundaries** 

[Figure 2-a]

-  $$3+1$$ interpretation points in an example for $$k=3$$, where each arrow visualises the result of calibration (end of arrow) at a particular point (beginning of arrow)
- Dirichlet calibration map $$\hat{\mu}_{\text {Dir }}(\mathbf{q} ; \mathbf{A}, \mathbf{c})$$ 를 통해서..
  - (before) $$\left(\varepsilon, \frac{1-\varepsilon}{k-1}, \ldots, \frac{1-\varepsilon}{k-1}\right), \ldots,\left(\frac{1-\varepsilon}{k-1}, \ldots, \frac{1-\varepsilon}{k-1}, \varepsilon\right), \text { and }\left(\frac{1}{k}, \ldots, \frac{1}{k}\right)$$
  - (after)  $$\left(\varepsilon^{a_{11}}, \ldots, \varepsilon^{a_{k 1}}\right) / z_{1}, \ldots,\left(\varepsilon^{a_{1 k}}, \ldots, \varepsilon^{a_{k k}}\right) / z_{k}, \text { and }\left(c_{1}, \ldots, c_{k}\right)$$

<br>

## 2-3) Relationship to other families

### Temperature scaling maps

- 이들 모두 Dirichlet family에 속한다

- $$\hat{\mu}_{\text {Temps }}(\mathbf{q} ; t)=\hat{\mu}_{\text {DirLin }}\left(\mathbf{q} ; \frac{1}{t} \mathbf{I}, \mathbf{0}\right)$$.

<br>

### Matrix Scaling Family

- $$\hat{\mu}_{M a t}(\mathbf{z} ; \mathbf{W}, \mathbf{b})=\sigma(\mathbf{W} \mathbf{z}+\mathbf{b})$$.

<br>

## 2-4) Fitting and ODIR regularization

***Any calibration model with tens of thousands parameters will OVERFIT to small validation set***

따라서, **novel ODIR ( Off-Diagonal and Intercept Regularization )** 을 제안한다!

- overfitting 해소에 도움

- temperature scaling보다 outperform

- loss function :

  $$L=\frac{1}{n} \sum_{i=1}^{n} \operatorname{logloss}\left(\hat{\mu}_{\text {DirLin }}\left(\hat{\mathbf{p}}\left(\mathbf{x}_{i}\right) ; \mathbf{W}, \mathbf{b}\right), y_{i}\right)+\lambda \cdot\left(\frac{1}{k(k-1)} \sum_{i \neq j} w_{i j}^{2}\right)+\mu \cdot\left(\frac{1}{k} \sum_{j} b_{j}^{2}\right)$$.

  -  $$w_{i j}, b_{j}$$  : $$\mathbf{W}$$ and $$\mathbf{b}$$의 element
  - $$\lambda, \mu$$ : hyper-parameters tunable ( validation 데이터에 대해 C.V 할때 )

- 위 loss function 해석 :

  - **(Diagonal 부분)** allowed to freely follow the biases of classes
  - **(Intercept 부분)** regularized separately from the off-diagonal elements, due to having different scales