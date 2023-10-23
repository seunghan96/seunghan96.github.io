---
title: \[multimodal\] Neural FE of signal data - (2) SincNet 
categories: [MULT,TS,AUDIO]
tags: [Multimodal Learning]
excerpt: Signal Data, Wav2Vec, SincNet, PASE
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Neural Feature Extraction of signal data ]

기존의 (NN을 사용하지 않은) Feature Extraction 방법은 주로 "지식/공식"에 기반한 deterministic한 형태의 추출 방법이었다면, 이번에 다룰 Neural Featrue Extraction은 "특정 목적을 수행하기 위해 적절한" feature를 뽑아내기 위한 방법이다. (즉, task에 따라 같은 data에서도 feature가 다르게 뽑힐 수 있는 non-deterministic한 방법이다 )

Neural Feature Extraction의 대표적인 2가지 방법은 아래와 같다.

- 1) Wav2Vec
- 2) SincNet ( + PASE )

이번 포스트에서는 **SincNet** 에 대해서 다룰 것이다.

<br>

# 1. SincNet 

## (1) 구성 

- 첫 번째 layer : sinc function ( $$\sin(x)/x$$ )
- 그 이후 layer : 일반적인 NN과 유사

- Task : 화자가 누구인지 맞추는 task

<img src= "https://i.imgur.com/n1EXsWV.png" width="250" />.

<br>

## (2) Time 도메인에서의 Convolution 

$$y[n]=x[n] * h[n]=\sum_{l=0}^{L-1} x[l] \cdot h[n-1]y[n]=x[n] * h[n]=\sum_{l=0}^{L-1} x[l] \cdot h[n-1]$$

- $$x[n]$$ : Time 도메인에서 $$n$$번째 raw wave sample
- $$h[n]$$ : Convolution Filter의 $$n$$ 번째 값
- $$y[n]$$ : output의 $$n$$번째 값

<br>

직관적인 이해

- $$y$$는 $$x$$와 필터 $$h$$의 연관성이 높을수록 커짐
- 필터 = 주파수 Input을 증폭/감쇄하는 역할
- **Time 도메인에서의 Convolution = Frequency도메인에서의 Multiplication **

<br>

## (3) Bandpass Filter ( Sinc Function )

목표 : 발화자 인식!

 $$\rightarrow$$ 발화자 인식에 있어서 중요한 주파수 영역(band)만을 남기길 원함 ( via **Bandpass Filter** )

Frequency 도메인에서, 이러한 역할을 하는 함수는 아래 그림과 같은 Rectangular function  

- $$\operatorname{rect}(t)=\Pi(t)=\left\{\begin{array}{ll}
  0, & \text { if }|t|>\frac{1}{2} \\
  \frac{1}{2}, & \text { if }|t|=\frac{1}{2} \\
  1, & \text { if }|t|<\frac{1}{2}
  \end{array}\right.$$.



<img src= "https://i.imgur.com/FgzqVBY.jpg" width="450" />.

<br>

Time 도메인에서는,이에 적합한 함수가 **Sinc Function **

- $$f(x) = \sin(x)/x$$

- WHY?

  Frequency 도메인에서의 **Rectangular function**으로 곱셈 연산

  = Time 도메인에서 **Sinc Function**으로의 컨볼루션 연산

- Sinc function을 Fourier Transform하면 Rectangular function

  (+ 그 역도 성립)

  - $$\int_{-\infty}^{\infty} \operatorname{sinc}(t) e^{-i 2 \pi f t} d t=\operatorname{rect}(f)$$.
  - $$\int_{-\infty}^{\infty} \operatorname{rect}(t) \cdot e^{-i 2 \pi f t} d t=\frac{\sin (\pi f)}{\pi f}=\operatorname{sinc}(\pi f)$$.

<br>

## (4) Window ( Hamming Window )

**Lobe & Side Lobe Effect**

- **Lobe** = 봉우리 ( 아래 그림 참조 )
  - Main Lobe  : 가장 높은 봉우리
  - Side Lobe : 그 외의 봉우리들
- **Side Lobe Effect** : Filter에 Side Lobe들이 많으면, main으로 잡아내고자 하는 것 외의 다른 주파수 영역대 정보도 잡아냄 (noise로 작용)
- Filter의 길이(=$$L$$)가 길수록, Side Lobe Effect $$\uparrow$$

<img src= "https://www.researchgate.net/profile/Sumathi-Mahadevan/publication/263793974/figure/fig2/AS:421469973696513@1477497680345/Window-spectrum-dipicting-of-mainlobe-and-side-lobe-distribution.png" width="350" />.



**Sinc Function 자르기**

- sinc function이 fourier transform을 거쳐서 rectangular function이 되기 위해선, $$L$$이 무한해야함 ( $$\int_{-\infty}^{\infty} \operatorname{sinc}(t) e^{-i 2 \pi f t} d t=\operatorname{rect}(f)$$ )

- 하지만, 실제로 그럴 순 없기 때문에, 적당한 길이로 sinc function을 잘라야함

- 아래 그림 참조 )

  <img src= "https://i.imgur.com/dPiXPQ6.png" width="350" />.

- (이상적) 맨 위 그림 ( $$L\rightarrow \infty$$, rectangular function )

  (현실)  $$L < \infty$$ ... main 부분의 일부를 캐치 X, noise도 일부 껴있음

- 이를 해결하기 위해 고안된 것이 **Window**

  ( sinc function을 자르지 않고, window를 사용해서 smoothing )

<br>

**Hamming Window**

- $$w[n]=0.54-0.46 \cdot \cos \left(\frac{2 \pi n}{L}\right)$$.

  <img src= "https://i.imgur.com/tHPxKTg.png" width="350" />.

<br>

요약 : 이상적인 Filter는 $$L \rightarrow$$, 하지만 현실적으로 그럴 수 없다. 따라서 불필요한 정보는 일부 포함될수밖에 없고, 중요한 정보도 일부 손실될 수 밖에 없다. 이를 보완하고자 제안된 것이 (Hamming) Window를 사용한 smoothing이다.

<br>

## (5) SincNet

$$y[n]=x[n] * g[n, \theta]$$

- $$x[n]$$ : Time 도메인에서의 $$n$$번째 샘플 입력
- $$g$$ : Convolution Filter
  - 이상적) Rectangular Function 형태 in Frequency 도메인

<br>

**이상적인 BandPass Filter**

- (Frequency 도메인) $$G\left[f, f_{1}, f_{2}\right]=\operatorname{rect}\left(\frac{f}{2 f_{2}}\right)-\operatorname{rect}\left(\frac{f}{2 f_{1}}\right)$$

- (Time 도메인) $$g\left[n, f_{1}, f_{2}\right]=2 f_{2} \operatorname{sinc}\left(2 \pi f_{2} n\right)-2 f_{1} \operatorname{sinc}\left(2 \pi f_{1} n\right)$$.

  여기에 window를 적용하면...

  $$g_{w}\left[n, f_{1}, f_{2}\right]=g_{w}\left[n, f_{1}, f_{2}\right] \cdot w[n]$$ ,

  where $$w[n]=0.54-0.46 \cdot \cos \left(\frac{2 \pi n}{L}\right)$$.

<br>

# Reference

- https://ratsgo.github.io/speechbook/docs/neuralfe/sincnet

- [Ravanelli, M., & Bengio, Y. (2018). Speech and speaker recognition from raw waveform with sincnet. arXiv preprint arXiv:1812.05920.](https://arxiv.org/pdf/1812.05920)

