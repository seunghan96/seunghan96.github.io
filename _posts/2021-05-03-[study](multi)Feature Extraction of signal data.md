---
title: \[multimodal\] Feature Extraction of signal data - (1) Fourier Transform
categories: [STUDY]
tags: [Multimodal Learning]
excerpt: Signal Data, Fourier Transform, MFCC
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Feature Extraction of signal data

Signal data에서 feature를 뽑아내는 대표적인 2가지 방법은 아래와 같다.

- 1) Fourier Transform
- 2) Mel-Frequency Cepstral Coefficients (MFCC)

이번 포스트에서는 우선 **Fourier Transform**에 대해서 다룰 것이다.

<br>

# 1. Before Fourier Transform....

***Fourier Transform : "시간(time)의 영역에서 주파수(frequency)"의 영역으로 변환***

( 푸리에 변환은 time(시간) 과 frequency(주파수)의 관점을 전환할 수 있게 해주는 중요한 것이다! )

<br>

( 보다 직관적인 이해를 위해.... )

ex) **속도**를 "시간"과 "거리"의 관점에서 바라봄

- 100m 달리기를 몇초에 달려? $$\frac{? \text{seconds}}{100\text{ m}}= \frac{ \text{time}}{\text{distance}}$$
- 자동차가 얼마나 빨리 달려? $$\frac{? \text{km}}{1 \text{ hour}} = \frac{ \text{distance}}{\text{time}}$$

위 예시의 두 가지 경우 모두 "속도"에 대해서 이야기하지만, 
하나는 "시간"의 관점에서, 하나는 "거리"의 관점에서 바라보았다.

이를 푸리에 변환과 연관시켜 생각해보면 아래와 같다.

- (time 영역) cycle이 얼마나 빨리 일어나?
  
  -  $$\frac{ \text{time}}{\text{cycle}}$$= Period
- (frequency 영역) 1초 동안 cycle이 얼마나 자주 일어나?
  -  $$\frac{ \text{cycle}}{\text{time (1s)}}$$= Hz

  <br>

푸리에 변환을 통해서 알고자 하는 것을 쉽게 비유해서 설명하자면, 

*여러 과일이 섞여있는 스무디의 "과일의 구성 성분을 알려주는 것"*이라고 보면 된다. 

( ex 스무디 A = 딸기 30% + 바나나 50% + 사과 20% )

<br>

# 2. Fourier Transform

( 앞으로 다룰 Fourier Transforms는 전부 **DISCRETE** Fourier Transform )

- **time**에 대한 함수 & **frequency**에 대한 함수를 연결해줌
- 특정 domain ( time or frequency )의 signal $$\rightarrow$$  특정 domain ( time or frequency )의 signal
- 용어 : **Fourier Transform $$\leftrightarrow$$ Inverse Fourier Transform**
- 공식 :
  - Fourier Transform : $$X_k =\sum_{n=0}^{N-1}x_n \cdot e^{\frac{-i 2\pi k n}{N}}$$
  - Inverse Fourier Transform : $$x_n =\frac{1}{N}\sum_{k=0}^{N-1}X_k \cdot e^{\frac{-i 2\pi k n}{N}}$$

<br>

<img src= "https://i1.wp.com/aavos.eu/wp-content/uploads/2017/11/Fourier-transform.gif?fit=900%2C522&ssl=1" width="450" />.

<br>

### Euler's formula (오일러 공식)

- 삼각함수 & 지수함수에 대한 관계
- 공식 : $$\exp(i \theta)= \text{cos} \theta + i\cdot \text{sin} \theta$$
  - 실수 part) $$\text{cos} \theta$$
  - 허수 part) $$i\cdot \text{sin} \theta$$
- 오일러 등식 ( 위의 식에 $$\theta=\pi$$ 대입  )
  - $$\exp(i \theta)+1=0$$.
- **복소 지수 함수** : $$cis(\theta) =\exp(i \theta)= \text{cos} \theta + i\cdot \text{sin} \theta$$
- 의미?
  - $$\exp(i \theta)$$ : **복소 평면 상** 반지름이 1인 원



<img src= "https://i.imgur.com/iVBkQVd.png" width="250" />.

<br>

### Fourier Transform 공식 들여다보기

(Discrete) Fourier Transform : $$X_k =\sum_{n=0}^{N-1}x_n \cdot e^{\frac{-i 2\pi k n}{N}}$$

( 가정 : time $$\rightarrow$$ frequency )

- $$n$$ : time index

- $$x_n$$ : time 도메인에서의 $$n$$번째 샘플

- $$X_k$$ : frequency 도메인에서의 $$k$$번째 Fourier Transform 결과

- $$\frac{k}{N}$$ : 각 속도 (angular velocity)

  ( 즉, 단위원이 얼마나 빠르게 회전하는지 )

(해석) $$X_k$$ : time 도메인 신호에서 $$k/N$$ 에 해당하는 주파수 성분

<br>

(Discrete) Fourier Transform 공식을 복소 지수 함수를 사용하여 다시 나타내기

$$\begin{aligned} X_k &=\sum_{n=0}^{N-1}x_n \cdot e^{\frac{-i 2\pi k n}{N}} \\ &=\sum_{n=0}^{N-1}x_n \cdot [\text{cos}(\frac{2\pi}{N}kn) - i \cdot \text{sin}(\frac{2\pi}{N}kn)]\end{aligned}$$

<br>

# 3. DFT Matrix

- Discrete Fourier Transform (DFT)를 matrix 형태로 나타낸 것
- 일종의 선형 변환으로 볼 수 있음 ( $$\vec{X}=W \cdot \vec{x}$$ )
- notation
  - $$x$$ : time 도메인의 signal
  - $$W$$ : $$N \times N$$의 matrix
    - row index : k
    - column index : $$n$$
    - $$W_{kn}$$ : $$e^{\frac{-i 2\pi k n}{N}}$$ 
  - 여기서 $$W$$를 DFT matrix라고 한다 

<br>

<img src= "https://i.imgur.com/HBeaeDU.png" width="450" />.



$$W=\frac{1}{\sqrt{N}}\left[\begin{array}{cccccc}
1 & 1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^{2} & \omega^{3} & \cdots & \omega^{N-1} \\
1 & \omega^{2} & \omega^{4} & \omega^{6} & \cdots & \omega^{2(N-1)} \\
1 & \omega^{3} & \omega^{6} & \omega^{9} & \cdots & \omega^{3(N-1)} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} & \cdots & \omega^{(N-1)(N-1)}
\end{array}\right]$$

where $$\omega=\exp (-2 \pi i / N)$$

<br>

# 4. Fast Fourier Transform (FFT) with numpy

*FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently (numpy)*

- 직접 구현한 `discrete_fourier_transform`

```python
import matplotlib.pyplot as plt

X_time=np.sin(np.arange(256))

def discrete_fourier_transform(X):
    N = len(X)
    n = np.arange(0,N)
    k = n.reshape(N,1)
    W = np.exp(-2j * np.pi * k * n / N)
    return np.dot(W,X)    
```

- numpy에서 제공하는 `np.fft.fft`
  - `X_freq1.real` : 실수 부분
  - `X_freq1.imag` : 허수 부분

```python
X_freq1 = discrete_fourier_transform(X_time)
X_freq2 = np.fft.fft(X_time)

real_part=X_freq1.real
imag_part=X_freq1.imag
```

<br>

# 4. Reference

- https://ratsgo.github.io/speechbook/docs/fe/ft
- https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
- https://www.youtube.com/watch?v=60cgbKX0fmE

