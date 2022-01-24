Change Point Detection in Time Series Data using Autoencoders with a Time-Invariant Representation

(2020, 4)

![figure2](/assets/img/ts/img164.png)



# 0. Abstract

CPD = locate **abrupt property changes**

This paper...

- uses **autoencoder (AE)** based methodology
  with **novel loss function**

- mitigate the issue of **false detection alarms**
  using a post-processing procedure

<br>

Allow user to indicate whether

- change points should be sought in **time domain**, **frequency domain**, or **both**

<br>

Detectable change points : abrupt changes in...

- slope / mean / variance / autocorrelation / frequency spectrum

<br>

# 1. Introduction

CPD's goal

- 1) goal in itself
- 2) pre-processing tool to segment a TS in homogeneous segments

<br>

CPD's categories

- 1) online CPD : 

  - real-time detection
  - dissimilarity = based on difference in distn of 2 intervals

- 2) retrospective (offline) CPD : 

  - robust detections
  - at the cost of needing more future data

  - *( this paper focuses on this method )*

<br>

Many CPD algorithms compare "past & future TS", 
by means of "dissimilarity" measure

<br>

Past Works ( algorithm : assumption(model) )

- ex 1) CUSUM : parametric probability distribution
- ex 2) GLR : autoregressive model
- ex 3) subspace method : state-space model
- ex 4) Bayesian online CPD

$\rightarrow$ performance depends on "how well actual data follows the assumed model"

- ex 5) KDE : parameter free
- ex 6) Related Density Ratio estimation : parameter free

<br>

***ex 7) autoencoder***

- pros

  - absence of distn assumptions
  - extract complex features in cost-efficient ways

- cons

  - no guarantee that distance between consecutive features reflect actual dissimilarity

  - correlated nature of TS samples is not properly used
  - absence of post-processing procedure preceding detection of peaks in the dissimilarity measure leads to high FP detection alarms

<br>

### TIRE (partially Time Invariate REpresentation)

- new AE-based CPD

- contribution

  - 1) **novel adaptation of AE** with a loss function that promotes **time-invariant features** ( + define new dissimilarity measure )

  - 2) focus on **non-iid data**, use discrete Fourier transform to obtain **temporally localized spectral information**,
    propose an approach that combines "time & frequency" domain info

<br>

# 2. Problem Formulation

$\mathbf{X}$ : time series of...

- \# of channel : $d$
  - $\mathbf{X}^{i}$ : $i$th channel
- length : $T$  ( where $0=T_{0} < T_1 < \cdots < T_p = T$ )
  - $T_1, T_2, \cdots$ : change points

<br>

$\left(\mathbf{X}\left[T_{k}+1\right], \ldots, \mathbf{X}\left[T_{k+1}\right]\right)$ : subsequence of time series

- realization of discrete time weak-sense stationary stochastic (WSS) process

<br>

Goal of CPD

- estimate change points, w.o prior knowledge on "number & location"

<br>

Dissimilarity Measure

- dissimilarity between (a) & (b)
  - (a) $(\mathbf{X}[t-N+1], \ldots, \mathbf{X}[t])$
  - (b) $(\mathbf{X}[t+1], \ldots, \mathbf{X}[t+N])$
- $N$ : user defined "window size"

<br>

Goal (1) :

- develop a **CPD-tailored feature embedding**
- and corresponding **dissimilarity measure $D_t$**
  - ( $D_t$ peaks, when the WSS restriction is violated )

Goal (2) :

- determine all **local maxima**
- label each local maximum, which exceed certain threshold $\tau$

<br>

# 3. AE based CPD

## (1) Preprocessing

**step 1) divide each channel into window size of $N$**

- $\mathbf{x}_{t}^{i}=\left[\mathbf{X}^{i}[t-N+1], \ldots, \mathbf{X}^{i}[t]\right]^{T} \in \mathbb{R}^{N}$.

<br>

**step 2) combine for every $t$ ( into single vector )**

- $\mathbf{y}_{t}=\left[\left(\mathbf{x}_{t}^{1}\right)^{T}, \ldots,\left(\mathbf{x}_{t}^{d}\right)^{T}\right]^{T} \in \mathbb{R}^{N d}$.

<br>

**step 3) Transformation**

- **(1) : DFT (discrete Fourier transform)**
  - DFT on each window
    - to obtain "temporally localized spectral information"

- **(2) : cropped to length $M$**

- (1) + (2) = $\mathcal{F}: \mathbb{R}^{N} \rightarrow \mathbb{R}^{M}$

- $\mathbf{z}_{t}=\left[\mathcal{F}\left(\mathbf{x}_{t}^{1}\right)^{T}, \ldots, \mathcal{F}\left(\mathbf{x}_{t}^{d}\right)^{T}\right]^{T} \in \mathbb{R}^{M d}$>

  ( = frequency-domain counterpart of $y_t$ )

<br>

## (2) Feature Encoding

[1] use AEs to extract features...

- from **time-domain(TD) windows **$\left\{\mathbf{y}_{t}\right\}_{t}$

- from **frequency-domain (FD) windows** $\left\{\mathbf{z}_{t}\right\}_{t}$.

<br>

[2] proposal of new loss function

- that promotes **time-invariance of the features** in consecutive windows

<br>

Notation

- ENC input : $\mathbf{y}_{t} \in \mathbb{R}^{N d}$
- ENC output : $\mathbf{h}_{t}=\sigma\left(\mathbf{W} \mathbf{y}_{t}+\mathbf{b}\right)$
- DEC output : $\tilde{\mathbf{y}}_{t}=\sigma^{\prime}\left(\mathbf{W}^{\prime} \mathbf{h}_{t}+\mathbf{b}^{\prime}\right)$
  - choose $\sigma=\sigma^{\prime}$ to be the hyperbolic tangent function
- minimize $\mid \mid \mathbf{y}_{t} - \tilde{\mathbf{y}}_{t} \mid \mid$

<br>

BUT, $\mathbf{h}_t$ will also contain information "NOT relevant to CPD"

- ex) phase shift, noise ...

$\rightarrow$ solve by introducing....

- 1) ***time invariant*** features
  - invariant over time within a WSS segment
  - ex) mean, amplitude, frequency
- 2) ***instantaneous*** features
  - all other info

<br>

