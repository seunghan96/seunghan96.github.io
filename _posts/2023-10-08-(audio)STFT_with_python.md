---
title: STFT with Python
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# STFT with Python

참고 : https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

<br>

# 1. Import Packages

```python
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
```

<br>

# 2. Import Dataset

```python
scale_file = "audio/scale.wav"
debussy_file = "audio/debussy.wav"
redhot_file = "audio/redhot.wav"
duke_file = "audio/duke.wav"
```

<br>

listen to music!

```python
ipd.Audio(redhot_file)
```

<br>

```python
scale, sr = librosa.load(scale_file)
debussy, _ = librosa.load(debussy_file)
redhot, _ = librosa.load(redhot_file)
duke, _ = librosa.load(duke_file)
```

<br>

# 3. Extract STFT

- Frame size : size of the window
- Hop size : size of the window stride

```python
FRAME_SIZE = 2048
HOP_SIZE = 512
```

<br>

- `librosa.stft`

```python
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale.shape)
print(type(S_scale[0][0]))
```

```
(1025, 342)
numpy.complex64
```

- 1025 : number of frequency bins
  - 1025 = (2048/2 + 1)
- 342 : number of frames

<br>

# 4. Calculate Spectogram

Scale : $$\mid S \mid^2$$

```python
Y_scale = np.abs(S_scale) ** 2
print(Y_scale.shape)
print(type(Y_scale[0][0]))
```

```
(1025, 342)
numpy.float32
```

- taking magnitude => getting real number!

<br>

# 5. Visualization

```python
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
```

<br.

## (1) Raw-scale

```python
plot_spectrogram(Y_scale, sr, HOP_SIZE)
```

![figure2](/assets/img/audio/img48.png)

<br>

## (2) Log-amplitude

```python
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE)
```

![figure2](/assets/img/audio/img49.png)

<br>

## (3) Log-frequency

```python
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
```

![figure2](/assets/img/audio/img50.png)
