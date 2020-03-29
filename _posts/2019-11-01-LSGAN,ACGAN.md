---
title: LSGAN (Least Squares GAN) & ACGAN (Auxiliary Class GAN)
categories: [DL,GAN]
tags: [Deep Learning, WGAN]
excerpt: LSGAN & ACGAN
---

# LSGAN & ACGAN
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## 1. LSGAN ( Least Squares GAN )

이전 절에서는 **WGAN**에 대해서 배웠다. WGAN이 좋았던 이유는, 두 분포 ($$p_{data}$$ 와 $$p_g$$) 사이의 중첩되는 부분이 없을 때에도, Loss Function으로 Wasserstein Distance를 사용함으로써 이미지를 생성할 수 있다는 안정성 측면의 장점이 있었다. 하지만, 그렇다고 이 모델이 반드시 좋다고 할 수는 없다.

지금까지 배웠던 GAN의 여러 모델들 (GAN, CGAN, WGAN 등)에서는 정답이냐 아니냐"에만 신경을 썼지, 그것이 "얼마나 잘해서" 정답이고, "얼마나 못해서" 오답인지를 반영하지는 못했다. () 이미 결정 경계에서 진짜 혹은 가짜로 분류된 이상, 그 경사는 손실되었다 ) 따라서 이 모델들 같은 경우에는, 생성된 fake image가 Discriminator만 속인 이상, 보다 더 낫게 진짜처럼 보이려는 노력을 하지 않게 된다. 그래서 나오게 된 개념이 **LSGAN**이다.

<br>

### Loss Function of LSGAN

LSGAN의 Loss Function이 기존의 GAN의 것과 어떻게 다른지를 확인해보면 이해할 수 있을 것이다.

<br>

**GAN**

- Discriminator의 Loss Function : $$L^{(D)} = - E_{x \sim P_{data}}logD(x) - E_z log(1-D(G(z))) $$
- Generator의 Loss Function : $$L^{(G)} = - E_zlogD(G(z))$$

<br>

**LSGAN**

- Discriminator의 Loss Function : $$L^{(D)} = E_{x \sim P_{data}}(D(x)-1)^2 + E_zD(G(z))^2$$
- Generator의 Loss Function : $$L^{(G)} = E_z(D(G(z))-1)^2$$



Loss가 모두 MSE로 대체된 것을 확인할 수 있다. 이러한 Loss Function을 사용함으로써, 판별기가 판독을 한 이후, 그 정답/오답의 '정도'또한 반영하여 품질의 개선에 기여할 수 있다는 것을 알 수 있다.

<br>

<br>

## 2. ACGAN ( Auxiliary Class GAN )

ACGAN은, CGAN과 마찬가지로 Generator의 입력으로 class의 label값을 받는다. 하지만 차이점은, CGAN에서는 Discriminator의 입력으로도 class의 label값을 받고, 생성해낸 이미지가 진짜일 확률값을 출력하지만, ACGAN에서는 Discriminator의 입력은 **이미지**이고, 출력값은 **해당 이미지가 진짜이면서 해당 class에 속할 확률**이다. 다음 그림을 통해 쉽게 이해할 수 있다.



<img src="data:image/png;base64,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" width="900" />

https://www.researchgate.net/figure/GAN-conditional-GAN-CGAN-and-auxiliary-classifier-GAN-ACGAN-architectures-where-x_fig1_328494719

<br>

ACGAN에서 "AC"는 Auxiliary Class를 뜻하는 것으로,  보조 클래스를 의미한다. 위 그림에서도 알 수 있듯, ACGAN의 Discriminator은 CGAN과 마찬가지로 "해당 fake image가 진짜/가짜일 확률"을 반환할 뿐만 아니라, 보조적으로 **생성한 fake image가 각 class에 속할 확률**을 계산하여 이 또한 모델의 평가에 있어서 반영한다. 우선 두 모델 (CGAN과 ACGAN)의 Loss Function을 살펴보자.

<br>

### Loss Function of ACGAN

**CGAN**

- Discriminator의 Loss Function : $$L^{(D)} = - E_{x \sim P_{data}}logD(x\mid y) - E_z log(1-D(G(z\mid y))) $$
- Generator의 Loss Function : $$L^{(G)} = - E_zlogD(G(z\mid y))$$

<br>

**ACGAN**

- Discriminator의 Loss Function : 
  $$L^{(D)} = - E_{x \sim P_{data}}logD(x\mid y) - E_z log(1-D(G(z\mid y))) - E_{x\sim P_{data}}logp(c\mid x)-E_z logp(c\mid G(z\mid y))$$
- Generator의 Loss Function : $$L^{(G)} = - E_zlogD(G(z\mid y)) - E_zlogp(c\mid G(z\mid y))$$

<br>

ACGAN의 Loss Function을 보면, 기존 CGAN의 Loss Function에 다음과 같은 추가적인 손실이 붙은 것을 확인할 수 있다. 

**[ Discriminator의 Loss Function 에 추가로 붙은 부분 ]**

- $$- E_{x\sim P_{data}}logp(c\mid x)$$ : real image가 주어졌을 때, **해당 class에 속할 확률**
- $$ -E_z logp(c\mid G(z\mid y))$$ : fake image가 주어졌을 때, **해당 class에 속할 확률**



**[ Generator의 Loss Function 에 추가로 붙은 부분 ]**

- $$- E_zlogp(c\mid G(z\mid y))$$ : 생성해낸 fake image가, **해당 class에 속할 확률**



위와 같이, 부가적인 분류기 손실함수가 추가되었다는 점을 제외하고는 CGAN과 동일하다.