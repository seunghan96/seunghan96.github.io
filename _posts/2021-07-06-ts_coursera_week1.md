---
title: \[coursera\] Week 1, Sequence and Predictions 
categories: [TS0]
tags: [TS]
excerpt: Sequences, Time Series and Prediction
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Sequences, Time Series and Prediction

( 참고 : coursera의 Sequences, Time Series and Prediction 강의 )

<br>

# [ Week 1 ] Sequence and Predictions 

1. Import Packages
2. Plotting Function, `plot_series`
3. TS with trend
4. TS with seasonality
5. TS with trend + seasonality
6. TS with trend+seasonality+noise
7. Preparing Forecast
8. Forecast

<br>

## 1. Import Packages

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
```

```
2.6.0
```

<br>

## 2. Plotting Function, `plot_series`

```python
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
```

<br>

## 3. TS with trend

### (1) `trend`

```python
def trend(time, slope=0):
    return slope * time
```

<br>

### (2) make synthetic dataset

```python
time = np.arange(4 * 365 + 1)
series = trend(time, 0.1)
print(time)
print(series)
```

```
[   0    1    2 ... 1458 1459 1460]
[0.000e+00 1.000e-01 2.000e-01 ... 1.458e+02 1.459e+02 1.460e+02]
```

<br>

### (3) plotting

```python
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

![figure2](/assets/img/ts/img94.png)

<br>

## 4. TS with seasonality

## (1) `seasonal_pattern`

( 임의의 seasonal pattern을 만들어내는 함수 )

```python
def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi), # if TRUE
                    1 / np.exp(3 * season_time))     # if FALSE
```

<br>

```python
example= (time % 365) / 365
plt.plot(example)
```

![figure2](/assets/img/ts/img95.png)

<br>

```python
plt.plot(seasonal_pattern(example))
```

![figure2](/assets/img/ts/img96.png)

<br>

### (2) `seasonality`

```python
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    season_pattern = seasonal_pattern(season_time)
    return amplitude * season_pattern
```

<br>

진폭 ( scale )을 40배로!

365일마다 반복되는 seasonality

```python
amplitude = 40
period=365
series = seasonality(time, period=period, amplitude=amplitude)
```

<br>

시각화

```python
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

![figure2](/assets/img/ts/img97.png)

<br>

## 5. TS with trend + seasonality

```python
slope = 0.05
baseline=10

series = baseline + trend(time, slope) + seasonality(time, period=period, amplitude=amplitude)
```

```python
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

![figure2](/assets/img/ts/img98.png)

<br>

## 6. TS with trend+seasonality+noise

White Noise를 생성하는 함수

```python
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
```

<br>

Noise 수준 : $$5 \times N(0,1)$$

```python
noise_level = 5
noise = white_noise(time, noise_level, seed=42)
```

```python
plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()
```

![figure2](/assets/img/ts/img99.png)

<br>

위에서 생성한 time series에 noise를 더한 뒤 시각화

```python
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()
```

![figure2](/assets/img/ts/img100.png)

<br>

## 7. Preparing Forecast

### (1) make synthetic dataset

hyperparameters

```
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
```

<br>

trend + seasonality + noise

```
time = np.arange(4 * 365 + 1, dtype="float32")

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

series += noise(time, noise_level, seed=42)
```

<br>

### (2) Train & Validation Split

- ~1000개 : train
- 1001개~ : validation

```python
split_time = 1000

time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```

<br>

Univariate Time Series

```python
print(time_train.shape)
print(x_train.shape) # Univariate
print(time_valid.shape)
print(x_valid.shape) # Univariate
```

```
(1000,)
(1000,)
(461,)
(461,)
```

<br>

```python
plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()
```

![figure2](/assets/img/ts/img101.png)

<br>

## 8. Forecast

### (1) Naive Forecast

이전 시점의 값을 다음 시점의 예측값으로 사용

```python
naive_forecast = series[split_time - 1:-1]
```

<br>

Validation 데이터의 예측 결과

- 전체 (time 1000~1461)
- 확대 (time 1000~1150)

```python
# 전체
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

# 확대
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)
```

![figure2](/assets/img/ts/img102.png)

![figure2](/assets/img/ts/img103.png)

<br>

예측 성능 (MSE & MAE)

```python
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
```

```
61.827534
5.937908
```

<br>

### (2) Moving Average (MA)

window size를 지정해줘야

- "window size=1의 MA" = "naive forecast"

```python
def moving_average_forecast(series, window_size):
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)
```

<br>

length 확인하기

- `series` : 전체 데이터셋...train&valid ( = 0~1461 )
- `moving_average_forecast(series, 30)` : 예측 결과...train&valid ( = 0~(1461-1430) )
- `moving_avg` : 예측 결과...valid ( = 1000 ~ 1461 )

```python
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

print(len(series)) 
print(len(moving_average_forecast(series, 30)))
print(moving_avg)
```

```
1461
1431
461
```

<br>

예측 성능 (MSE & MAE)

```python
print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
```

```
106.674576
7.142419
```

<br>

### (3) 차분 후 MA

1년(=365일)전 값을 빼줌

- ex) 2021년 7월 29 값 - 2020년 7월 29일 값

```python
lag=365
diff_series = (series[lag:] - series[:-lag])
diff_time = time[lag:]
```

<br>

1461일-365일 = 1096일

```python
len(diff_series),len(diff_time)
```

```
(1096, 1096)
```

<br>

```python
plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()
```

![figure2](/assets/img/ts/img104.png)

<br>

```python
window_size=50
lag=365
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - lag - window_size:]
```

```python
plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - lag:])
plot_series(time_valid, diff_moving_avg)
plt.show()
```

![figure2](/assets/img/ts/img105.png)

<br>

365일 전 값들을 다시 더해줘야!

```python
diff_moving_avg_plus_past = series[split_time - lag:-lag] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()
```

<br>

예측 성능 (MSE & MAE)

```python
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
```

```
52.973663
5.839311
```

<br>

### (4) 차분 후 MA + smoothing

```python
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - lag - window_size:]
```

<br>

```python
# BEFORE (smoothing X)
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

# AFTER (smoothing O)
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - (lag+5):-(lag-5)], 10) + diff_moving_avg
```

<br>

```python
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()
```

![figure2](/assets/img/ts/img106.png)

<br>

예측 성능 (MSE & MAE)

```python
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
```

```
33.452263
4.569442
```

