---
title: \[coursera\] Sequences, Time Series and Prediction
categories: [TS]
tags: [TS]
excerpt: Week 3, RNN for Time Series
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Sequences, Time Series and Prediction

( 참고 : coursera의 Sequences, Time Series and Prediction 강의 )

<br>

# [ Week 3 ] RNN for Time Series

1. Import Packages & Dataset
2. Modeling (RNN)
3. Modeling (LSTM)

<br>

## 1. Import Packages & Dataset

### (1) Packages

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
```

<br>

### (2) Datasets

Hyperparameters

```python
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
```

<br>

TS data with trend+seasonality+noise

```python
time = np.arange(4 * 365 + 1, dtype="float32")
series = baseline + trend(time, slope) + seasonality(time, period=365, 
                                                     amplitude=amplitude)
series += noise(time, noise_level, seed=42)
```

<br>
Train & Validation Split

```python
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

<br>

Data Loader

```python
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

train_set = windowed_dataset(x_train, window_size, batch_size=128, shuffle_buffer=shuffle_buffer_size)
```

<br>

## 2. Modeling (RNN)

- `tf.keras.layers.Lambda` 를 사용하여 차원 조정해줌
- `tf.keras.layers.SimpleRNN` x 2
  - 1번째 layer : return_sequences=True
  - 2번째 layer : return_sequences=False
- output scale이 -100~100 사이즈음이므로, 100을 곱해줌

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```

<br>

`tf.keras.callbacks.LearningRateScheduler`

- Learning Rate 스케줄러

  ( 에폭이 지날수록 줄어들게끔! )

```python
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
```

<br>

사용하는 Loss Function : **Huber** loss function

```python
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```

<br>

## 3. Modeling (LSTM)

`tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))`

- bidirectional LSTM 사용하기
- hidden unit의 개수  = 32

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```

```python
model.compile(loss="mse", 
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5,momentum=0.9),
              metrics=["mae"])
history = model.fit(dataset,epochs=500,verbose=0)
```

