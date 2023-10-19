---
title: Complex Numbers for Audio Signal Processing
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Complex Numbers for Audio Signal Processing

참고 : https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

<br>

# 1. Why use complex number in audio?

To express both (1) frequency & (2) phase!

- complex numbers : $$c = a+ib$$
  - where $$a,b \in \mathbb{R}$$.
  - $$a$$ : REAL part
  - $$ib$$ : IMAGINARY part

<br>

Plotting complex numbers in ...

(1) Cartesian coordinates

![figure2](/assets/img/audio/img25.png)

<br>

(2) Polar coordinates

![figure2](/assets/img/audio/img26.png)

- express using $$c$$ and $$\gamma$$

<br>

# 2. Polar Coordinates

<br>

$$\cos (\gamma)=\frac{a}{ \mid c \mid } $$.

$$\rightarrow$$ $$a= \mid c \mid  \cdot \cos (\gamma)$$

<br>

$$\sin (\gamma)=\frac{b}{ \mid c \mid }$$.

$$\rightarrow$$ $$b= \mid c \mid  \cdot \sin (\gamma)$$

<br>

 $$c = a+ib$$

$$\rightarrow$$ $$c= \mid c \mid  \cdot(\cos (\gamma)+i \sin (\gamma))$$.

<br>

$$\text{tan}(\gamma) = \frac{\sin (\gamma)}{\cos (\gamma)}=\frac{b}{a}$$.

$$\rightarrow$$ $$\gamma=\arctan \left(\frac{b}{a}\right)$$.

<br>

# 3. Eular Formula

## (1) Euler formula

$$e^{i \gamma}=\cos (\gamma)+i \sin (\gamma)$$.

<br>

## (2) Euler identity

$$e^{i\pi} +1=0$$.

$$\rightarrow$$ $$e^{i\pi}=-1$$.

$$\rightarrow$$ $$e^{i\pi}=-1$$.

<br>

## (3) Polar coordinates 2.0

$$c= \mid c \mid  \cdot(\cos (\gamma)+i \sin (\gamma))$$.

$$e^{i \gamma}=\cos (\gamma)+i \sin (\gamma)$$.

$$\rightarrow$$ $$c= \mid c \mid e^{i\gamma}$$.

- can express complex number $$c$$ with
  - (1) $$\mid c \mid$$ : magnitude
  - (2) $$\gamma$$ : direction

<br>

Interpretation

![figure2](/assets/img/audio/img27.png)

<br>

$$\rightarrow$$ why not use MAGNITUDE & PHASE as polar coordinates?

<br>

# 4. Fourier Transform using Complex Number

## (1) Magnitude & Phase

### Magnitude

![figure2](/assets/img/audio/img28.png)

![figure2](/assets/img/audio/img29.png)

<br>

### Phase

![figure2](/assets/img/audio/img30.png)

![figure2](/assets/img/audio/img31.png)

- meaning of $$-$$  : rotate "clock-wise"

<br>

## (2) Continuous audio signal

$$g(t) \quad g: \mathbb{R} \rightarrow \mathbb{R}$$.

 ![figure2](/assets/img/audio/img32.png)

<br>

## (3) COMPLEX Fourier Transform

$$\hat{g}(f)=c_f$$.

- $$\hat{g}: \mathbb{R} \rightarrow \mathbb{C}$$.

<br>

Mapping into a complex space!

 ![figure2](/assets/img/audio/img33.png)

<br>

### Mathematical Expression

$$\hat{g}(f)=\int g(t) \cdot e^{-i 2 \pi f t} d t$$.

<br>

 ![figure2](/assets/img/audio/img34.png)

<br>

## (4) Complex Fourier Transform coefficients

 ![figure2](/assets/img/audio/img35.png)

<br>

## (5) Magnitude & Phase

### $$c_f=\frac{d_f}{\sqrt{2}} \cdot e^{-i 2 \pi \varphi_f}$$.

<br>

### a) Magnitude

ABSOLUTE value of $$\hat{g}(f)$$

= REAL part of $$c_f$$

= $$d_f=\sqrt{2} \cdot \mid \hat{g}(f) \mid $$.

<br>

### b) Phase

IMAGINARY value of $$\hat{g}(f)$$

= $$\varphi_f=-\frac{\gamma_f}{2 \pi}$$.

<br>

# 5. Fourier & Inverse Fourier Transform

$$\begin{gathered}
\hat{g}(f)=\int g(t) \cdot e^{-i 2 \pi f t} d t \\
g(t)=\int c_f \cdot e^{i 2 \pi f t} d f
\end{gathered}$$.



