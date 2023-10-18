---
title: Melspectogram with Python
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Melspectogram with Python

참고 : https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

<br>

# 1. Import Packges & Datasets

```python
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

scale_file = "audio/scale.wav"
scale, sr = librosa.load(scale_file)
```

<br>

# 2. Mel filter banks

```python
num_bands = 10
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=num_bands)
print(filter_banks.shape)
```

```
(10,1025)
```

- 10 : number of bands

- 1025 = (2048/2+1)

  - 2048 : size of each frame 

    (  = 2048 samples in a single frame )


<br>

### Visualize Mel filter banks

```python
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()
```

![figure2](/assets/img/audio/img56.png)

<br>

# 3. Melspectogram

Apply **Mel filter banks ($M$)** to **Spectogram ($Y$)**

```python
mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, 
                                                 n_fft=2048, 
                                                 hop_length=512, 
                                                 n_mels=10)
print(mel_spectrogram.shape)
```

```
(10,342)
```

- 10 : number of bands
- 342 : number of frames

<br>

Convert it into **log scale**

```python
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
```

<br>

### Visualization

```python
plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
```

![figure2](/assets/img/audio/img57.png)
