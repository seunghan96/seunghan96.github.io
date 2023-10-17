---
title: Audo data for DL (python)
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Audo data for DL (python)

참고 :

- https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

- https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/11-%20Preprocessing%20audio%20data%20for%20deep%20learning/code/audio_prep.py

<br>

# 1. Import Packages & Files

Packages

```python
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
FIG_SIZE = (15,10)
```

<br>

Files

```python
file = "blues.00000.wav"
signal, sample_rate = librosa.load(file, sr=22050)
```

<br>

# 1. WaveForm ( raw TS )

```python
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
```

![figure2](/assets/img/audio/img12.png)

<br>

# 2. Spectrum ( via FFT )

```python
fft = np.fft.fft(signal)
# (1) spectrum [Y axis]
spectrum = np.abs(fft)

# (2) frequency variable [X axis]
f = np.linspace(0, sample_rate, len(spectrum))

# need only first HALF
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]
```

```python
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
```

![figure2](/assets/img/audio/img13.png)

![figure2](/assets/img/audio/img14.png)

<br>

# 3. Spectogram ( via STFT )

( add TIME information )

```python
n_fft = 2048 # number of samples in window
hop_length = 512 # hop size of window ( = stride )

# STFT hop length duration
hop_length_duration = float(hop_length)/sample_rate

# STFT window duration
n_fft_duration = float(n_fft)/sample_rate

stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
spectrogram = np.abs(stft)
```

```python
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")
```

![figure2](/assets/img/audio/img15.png)

<br>

to log scale ( = dB )

```python
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
```

![figure2](/assets/img/audio/img16.png)

<br>

# 4. MFCCs

use 13 coefficients

```python
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
```

```python
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

plt.show()
```

![figure2](/assets/img/audio/img17.png)
