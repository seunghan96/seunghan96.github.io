---
title: MFCCs with Python
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# MFCCs with Python

참고 : https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf

<br>

# 1. Import Packages & Datasets

```python
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_file = "audio/debussy.wav"
signal, sr = librosa.load(audio_file)
```

<br>

# 2. Extract MFCCs

```python
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
print(mfccs.shape)
```

```
(13, 1292)
```

- 13 coefficients
- 1292 frames

<br>

# 3. Visualization

```python
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
```

![figure2](/assets/img/audio/img67.png)

<br>

# 4. $$\Delta$$ and $$\Delta \Delta$$ MFCCs

```python
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)
```

<br>

First derivative

```python
plt.figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
```

![figure2](/assets/img/audio/img68.png)

<br>

Second derivative

```python
plt.figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
```

![figure2](/assets/img/audio/img69.png)

<br>

# 5. Get MFCC features

Concatenate "original" & "first" derivative & "second" derivative

```python
mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
print(mfccs_features.shape)
```

```
(39,1292)
```

