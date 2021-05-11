---
title: \[multimodal\] FE of signal data - (2) MFCC
categories: [STUDY]
tags: [Multimodal Learning]
excerpt: Signal Data, Fourier Transform, MFCC
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Feature Extraction of signal data ]

Signal data에서 feature를 뽑아내는 대표적인 2가지 방법은 아래와 같다.

- 1) Fourier Transform
- 2) Mel-Frequency Cepstral Coefficients (MFCC)

이번 포스트에서는 **Mel-Frequency Cepstral Coefficients (MFCC)**에 대해서 다룰 것이다.

<br>

# 1. Introduction

MFCC ?

- signal 데이터에서 "중요한 정보"만 남긴 feature
- 음성 인식 시스템에서 자주 사용되는 feature
- NN기반 feature extraction 방법과 달리, "공식"에 기반한 방법

<br>

# 2. Mel-Frequency Cepstral Coefficients (MFCC)

- Step1 ) 입력 signal을 짧은 구간으로 나눔 ( called **"frame"** )
  <br>
- Step2 )**각 frame에 Fourier Transform** 실시
  - Fourier Transform : "time" $$\rightarrow$$ "frequency"
  - 모든 frame에 Fourier Transform을 한 것을 **"Spectrum"**이라고 부름
  <br>
  
- Step 3) **Spectrum**에 Mel Filter Bank 필터 적용 ( called **"Mel Spectrum"** )
  - Mel Filter Bank : 사람의 말 소리 인식에 민감한 frequency는 세밀하게, 그렇지 않은 부분은 덜 촘촘히 분석하는 필터
    <br>
- Step 4) **MFCC** 생성
  - **MFCC** : log(Mel Spectrum)에 Inverse Fourier Transform한 것
    - 다시 time 도메인으로 컴백!
  - 인간의 말 소리에서 중요한 특징들이 추출된 것들!
  - NN에 의한 feature extraction도 대두되고 있지만, 여전히 많이 사용되고 있는 feature!

<br>

<img src= "https://i.imgur.com/Pn5LGTk.png" width="550" />

<br>

# 3. 성능 개선을 위해서...

## (1) Pre-emphasis

사람의 목소리를 위의 과정처럼 spectrum으로 변환 시,

일반적으로 "LOW frequency" > "HIGH frequency" ( 특히 모음에서 )

<br>

성능 개선 via "HIGH frequency"의 에너지 $$\uparrow$$  ! 이를 **Pre-emphasis**라고 함

공식 : $$\mathbf{y}_t = \mathbf{x}_t - \alpha \mathbf{x}_{t-1}$$

- $$\alpha$$는 주로 0.95나 0.97 사용

<br>

Pre-emphasis의 효과

- 1) 전체 frequency 영역에서 고른 에너지 분포
- 2) Fourier Transform 시 발생할 수 있는 numerical problem 예방
- 3) Signal-to-Noise Ratio ( 잡음 비율 ) 개선

<br>

## (2) Framing

signal은 매우 빠르게 변화함 (non-stationary) 

$$\rightarrow$$ 정확히 파악하기 어려울 수도!

따라서, signal을 짧은 시간 단위 (주로 25ms) 로 나눔 (위의 step 1). 이를 **"framing"**이라 함

<br>

## (3) Windowing

위에서 말한 것 처럼, signal을 매우 작은 시간 단위로 framing한다. 이렇게 해서 생긴 여러 frame들에 어떠한 함수를 적용할 때, smoothing하는 방법이다.

**ex) Hamming Window**
<br>

### Hamming Window

$$w[n]=0.54-0.46 \cos \left(\frac{2 \pi n}{N-1}\right)$$.

- frame의 중간에 있는 값들 : 그대로 ( $$\times 1$$ )
  - $$0.54-0.46(-1)=1$$ 이므로
- frame 양 끝의 값들 : 작은 값이 곱해짐
  - $$0.54-0.46(1)=0.12$$ 이므로
    <br>

<img src= "https://i.imgur.com/tHPxKTg.png" width="450" />.

<br>

## (4) Post Processing

MFCCs를 생성한 이후, 성능 향상 위해 Lift 혹은 Mean Normalization을 수행하기도 함

<br>

# 4. 기타

## (1) Magnitude

이전 포스트에서 봤듯이, time 도메인의 데이터가 (Discrete) Fourier Transform을 거치고 나면 실수 part와 허수 part로 구성된 frequency 도메인의 데이터가 나온다. ( $$X[k] = a + b\times j$$ )

이 frequency의 (1) 진폭(magnitude)와 (2) 위상(phase)는 아래와 같이 나타낼 수 있다.

- 진폭 : frequency의 "크기"
- 위상 : frequency의 "위치"

<img src= "https://i.imgur.com/HTD1vC2.png" width="450" />.

<br>

## (2) Power Spectrum

공식 : $$\text { Power }=\frac{\mid X[k]\mid^{2}}{N}$$

<br>

## (3) Filter Banks

사람의 목소리 인식은 LOW frequency 영역대가 HIGH frequency 영역대보다 민감하다.

따라서, LOW frequency 영역대를 보다 자세히 볼 필요가 있다. 

그러기 위해 사용하는 것이 **"Filter Banks"**

여기서 사용하는 필터는 **Mel Scale(멜 스케일)**

- 헤르츠 ($$Hz$$) $$\rightarrow$$ 멜(m)

- 공식 )

  $$\begin{aligned}
  m &=2595 \log _{10}\left(1+\frac{f}{700}\right) \\
  f &=700\left(10^{m / 2595}-1\right)
  \end{aligned}$$.

<br>

## (4) Log - Mel Spectrum

Log를 씌우는 이유? 

$$\rightarrow$$ "사람의 목소리는 log scale!"

용어

- (log 이전) **Mel Spectrum (멜 스펙트럼)**
- (log 이후) **Log-Mel Spectrum (로그 멜 스펙트럼)**

<br>

## (5) MFCCs

(Log-) Mel Spectrum은 feature 간의 상관관계 존재!

( $$\because$$ 여러 차원에서 서로 영향을 주고 받음 )

<br>
이러한 상관관계를 해소하기 위해, 다시 **INVERSE Fourier Transform** 적용

이것을 "**Mel-frequency Cepstral Coefficients(MFCCs)**"라고 함

<br>

# Reference

https://ratsgo.github.io/speechbook/docs/fe/mfcc#magnitude

