---
title: Shift Equivariance
categories: [TS, ML]
tags: []
excerpt:

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# **Shift Equivariance**

## 1. 개념

Shift Equivariance는 **입력 TS을 시간적으로 shift(이동)** 시켰을 때,

**모델이 추출하는 feature도 동일하게 shift되는 성질**

> **입력 x(t) → x(t − τ)** 로 이동하면

> **출력 f(x)(t) → f(x)(t − τ)** 로 똑같이 이동

“Pattern의 **위치**가 아니라 **형태**가 중요”한 경우 매우 중요한 속성!

<br>

## 2. 중요성

***현실의 TS 데이터는 대부분 misaligned 되어있다***

- ECG: QRS peak가 사람마다 약간씩 시간 위치가 다름
- Gesture: 같은 동작을 해도 시작 타이밍이 다름
- Sensor: 이벤트가 언제 발생했는지가 일정하지 않음

<br>

Shift-equivariant 모델을 사용하면,

- Pattern이 언제 나타났는지 → **덜 중요**
- Pattern의 모양 자체 → **더 중요**

<br>

## 3. Ex: ECG (R-peak detection)

- Signal A: QRS peak가 **t=1.2초**에 나타남
- Signal B: QRS peak가 **t=1.4초**에 나타남 (0.2초 shift)

<br>

두 신호는 **같은 심장 Pattern**을 가지고 있지만 **시간 위치만 다름**.

Shift-equivariant 모델:

- $$\text{feature}_A(t) = \text{feature}_B(t + 0.2)$$

<br>

즉, **peak 위치만 달라지고 Pattern 특성은 유지됨** 

→ classification에 영향 없음!

<br>

## **4. 어디에서 Shift Equivariance가 나타나는가?**

## (1) CNN

CNN은 구조적으로 **shift-equivariant**

- **Filter가 sliding** 되기 때문에! 
- Input shifting → output도 shifting

$$\therefore$$ 이 때문에 CNN은 time-series에서 자주 사용됨.

<br>

CNN의 한계점

- Local pattern 중심

- Long-range dependency 처리에 약함

  → TSCMamba에서는 이를 보완하기 위해 ROCKET, MLP, Mamba 등을 조합.

<br>

## (2) CWT + Shift Equivariance

CWT (Continuous Wavelet Transform)

- Real-valued mother wavelet 사용 시 **"shift equivariant"** 

DFT (Discrete Fourier Transform)

- Shift equivariant가 아님

- Global transform이기 때문
- Shift가 들어가면 **진폭/위상 모두 변함** → Pattern 왜곡

<br>

## **6. TSCMamba에서의 Shift Equivariance 활용**

Modules

- [CWT] shift-equivariant **spectral features** 확보
- [ROCKET/CNN] shift-equivariant **local features** 확보
- [MLP] shift-invariant **global features** 확보

$$\rightarrow$$ Multi-view fusion으로 서로 보완

<br>

Summary

- (1) **Shift-equivariant local + spectral features**
- (2) **Shift-invariant global features**