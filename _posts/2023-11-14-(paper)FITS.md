---
title: FITS; Modeling TS with 10k Parameters
categories: [TS]
tags: []
excerpt: ICLR 2024 (?)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# FITS: Modeling TS with 10k Parameters

<br>

# Contents

0. Abstract
0. Introduction
0. Related Work & Motivation
0. Method
0. Experiments

<br>

# Abstract

FITS

- lightweight model
- directly process raw time-domain (X)
- interpolation in complex **frequency** domain (O)

- use 10k parameters

<br>

# 1. Introduction

Frequency domain representation in TS: **compact & efficient**

Existing works: FEDformer, TimesNet

- Still... comprehensive utilization of frequency domain's compactness remains unexplored!

  ( i.e. employing **complex numbers** )

<br>

### FITS

- Reinterpret TS tasks (i.e. forecasting, reconstruction) as ***interpolation within frequency domain***
- Produce an extended TS segment by **interpolating the frequency representation of a provided segment**
  - ex) Forecasting: by extending the given look-back window with frequency interpolation
  - ex) Reconstruction: by interpolating the frequency representation of its downsampled counterpart

- Core of FITS = **complex-valued linear layer**
  - designed to learn "amplitude scling" & "phase shift"

- But still, fundamentally remains a time-domain model, by integrating rFFT
  - (1) transform input into frequency domain using rFFT
  - (2) mapped back to time domain
- Incorporates **low-pass filter**
  - ensures a **compact representation**
- Use only **10k params**

<br>

# 2. Related Work & Motivation

## (1) Frequency-aware TS Models

- FNet
- FEDFormer
- FiLM
- TimesNet

<br>

## (2) Divide & Conquer the Frequency Components

Treating the TS as **SIGNAL**

- Break down into **linear combination of sinusoidal components** ( w/o info loss )
  - each component = unique frequency & initial phase & amplitude

- Forecasting each frequency component: straightforward
  - only apply a phase bias to the sinusodial wave ( based on time shift )
  - then, linearly combine this shifted waves!

<br>

HOWEVER, forecasting each sinusoidal component in **TIME domain** can be cumbersom!

( $$\because$$ sinusoidal components are treated as a sequences of data points )

$$\rightarrow$$ Solution: perform it on **FREQUENCY domain**

<br>

# 3. Method

## (1) Preliminaries: FFT & Complex Frequency Domain

### a) FFT 

- Efficiently perform DFT on complex number sequences

- Transforms discrete-time signals from TIME $$\rightarrow$$ FREQUENCY

  ( $$N$$ real numbers $$\rightarrow$$ $$N/2+1$$ complex numbers )

<br>

### b) Complex Frequency Domain

Complex number

- captures both **amplitude** & **phase** of the component

- can be represented as a **complex exponential element** with a given amplitude & phase
- $$X(f)= \mid X(f) \mid  e^{j \theta(f)}$$.
  - $$X(f)$$ : complex number associated with the frequency component at frequency $$f$$
  - $$ \mid X(f) \mid $$ : amplitude
  - $$\theta(f)$$ : phase



### Complex plane

Complex exponential element can be visualized as ...

- a vector with a length equal to the amplitude and angle equal to the phase

- $$X(f)= \mid X(f) \mid (\cos \theta(f)+j \sin \theta(f))$$.

<br>

![figure2](/assets/img/ts/img541.png)

<br>

### Time Shift & Phase Shift

Time Shift = Phase Shift in FREQUENCY domain

- by multiplying a unit complex exponential element with the corresponding space ( in FREQ domain )

<br>

Shift signal $$x(t)$$ forawrd in TIME by $$\tau$$ = $$x(t-\tau)$$

$$\rightarrow$$ Fourier transform: $$X_\tau(f)=e^{-j 2 \pi f \tau} X(f)= \mid X(f) \mid  e^{j(\theta(f)-2 \pi f \tau)}=[\cos (-2 \pi f \tau)+j \sin (-2 \pi f \tau)] X(f)$$

- Amplitude : $$ \mid X(f) \mid $$
- Phase $$\theta_\tau(f)=\theta(f)-2 \pi f \tau$$
  - linear to the time shift.

<br>

## (2) FITS Pipeline

(Motivation) Longer TS = Higher frequency resolution

$$\rightarrow$$ Train FITS to extend TS segment by ***interpolating the frequency representation*** of input TS semgnet

![figure2](/assets/img/ts/img542.png)

<br>

LPF (Low-Pass Filter)

- To reduce the model size
- Eliminates HIGH-frequency components above certain cutoff

<br>

Forecasting

- generate the look-back window along with the horizon

  ( = combining backcast & forecast )

<br>

Reconstruction

- downsample the original TS based on specific downsampling rate
- then, perform frequency interpolation

<br>

## (3) Key Mechanism of FITS

### a) Complex Frequency Linear Interpolation

Interpolation rate: $$\eta$$

- ratio of the model's output length $$L_o$$ to its corresponding input length $$L_i$$. 

<br>

Frequency interpolation 

- operates on the **normalized** complex frequency representation ( = half the length of the original TS )

<br>

Interpolation rate can also be applied to the frequency domain

- $$\eta_{f r e q}=\frac{L_o / 2}{L_i / 2}=\frac{L_o}{L_i}=\eta$$.

<br>

With an arbitrary frequency $$f$$ ...

- Frequency band $$1 \sim f$$ in the original signal is linearly projected to the frequency band $$1 \sim \eta f$$ in the output signal. 
- Input length of our complex-valued linear layer = $$L$$ 
- Interpolated output length = $$\eta L$$. 

<br>

### b) Low Pass Filter (LPF)

- To compress the model's volume

- By discarding frequency components above a specified cutoff frequency (COF)

- Ensures that a significant portion of the original time series' meaningful content is preserved

  - High-frequency components filtered out by the LPF typically comprise noise,!

    ( = irrelevant for effective time series modeling )

![figure2](/assets/img/ts/img543.png)

<br>

Selecting COF? Nontrivial!

$$\rightarrow$$ propose method based on the **harmonic content of the dominant frequency**

<br>

Also adopt **channel independence**

<br>

# 4. Experiments

## (1) Forecasting as Frequency Interpolation

Input Length : $$L$$
Output Length : $$H$$
Combination of look-back window & forecasting horizon : $$L+H$$

<br>

Interpolation rate of the forecasting task:

- $$\eta_{\text {Fore }}=1+\frac{H}{L}$$.
