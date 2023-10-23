---
title: Mel-Frequency Cepstral Coefficients (MFCCs)
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Mel-Frequency Cepstral Coefficients (MFCCs)

참고 : https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

<br>

# 1. Introduction

Mel-Frequency **Cepstral** Coefficients 

- Cepstral: ***Cepstrum*** $$\leftrightarrow$$ ***Spectrum***

![figure2](/assets/img/audio/img58.png)

<br>

How to compute ***Cepstrum***?

### $$C(x(t))=F^{-1}[\log (F[x(t)])]$$.

<br>

![figure2](/assets/img/audio/img59.png)

![figure2](/assets/img/audio/img60.png)

<br>

# 2. Vocal tract

Vocal Tract acts as a **filter** of a speech

- vocal tract (성도) : 소리가 나가는 길

![figure2](/assets/img/audio/img61.png)

<br>

# 3. Decomposing Speech

![figure2](/assets/img/audio/img62.png)

$$\rightarrow$$ Peaks of **spectral envelope**, or **formants**, carry the identity of sound!

<br>

We can see "speech" as a "convolution of (1) with (2)"

- (1) vocal tract frequency response
- (2) glottal pulse

<br>

$$X(t)=E(t) \cdot H(t)$$.

$$\log (X(t))=\log (E(t) \cdot H(t))$$.

$$\log (X(t))=\log (E(t))+\log (H(t))$$.

![figure2](/assets/img/audio/img63.png)

<br>

![figure2](/assets/img/audio/img64.png)

<br>

# 4. Liftering

Removing the high quefruency values! ( or the **"glottal pulse"** )

![figure2](/assets/img/audio/img65.png)

<br>

# 5. Calculating MFCCs

Waveform $$\rightarrow$$ DFT $$\rightarrow$$ Log-amplitude Spectrum $$\rightarrow$$ ***Mel-scaling*** $$\rightarrow$$ ***Discrete cosine transform***

But why use **discrete cosine transform**?

( = similar to inverse transform )

- simplfied version of FT
- get real-valued coefficient
- decorrelate energy in different mel bands
- reduce \# of dim to represent spectrum

<br>

### How many coefficients to use?

- First 12~13 coefficients ( low frequencies )

  - 1st : Most information
    - corresponds to "formants", "spectral envelope"
  - Last : Least information

- Use $$\Delta$$ and $$\Delta \Delta$$ MFCCs

  $$\rightarrow$$ about 39 coffeicients per frame

<br>

### Visualization

![figure2](/assets/img/audio/img66.png)

<br>
