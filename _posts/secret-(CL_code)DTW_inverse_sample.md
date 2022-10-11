---
title: (code) DTW inverse sample
categories: [CL, TS]
tags: []
excerpt:
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# DTW inverse sample

# 1. Settings

```python
N = 150 # number of TS
B = 64 # batch size
T = 222 # total length of TS
t = 10 # length of subseries
```

<br>

# 2. Datasets

```python
ts_full = np.random.randn(B,T)
print(ts_full.shape)
```

```
(64, 222)
```

<br>

```python
batch_data_idx = np.random.choice(N,B,replace=False)
```

```python
batch_data_idx
```

```
array([ 82, 105, 119,  43,  19,  34, 123,  49, 113, 149, 137,  72, 146,
       142,  42, 127,  84,  46, 124,  20,  68,  77,   7,  31, 100,   2,
       118,  10,  35,  30,  50,  41, 108,  54,  90,  32, 135, 114, 107,
       109,  96,  88,  95,  78, 115, 141,  53, 145,  40,  71,  86,   5,
        99, 117, 121,  65,  66,  79,  18,  33,  87, 132,  92,  25])
```

<br>

# 3.  DTW

```python
DTW_matrix = np.random.uniform(0,1,(N,N))
np.fill_diagonal(DTW_matrix, 0)
DTW_matrix_batch = DTW_matrix[batch_data_idx][:,batch_data_idx]
```

```python
print(DTW_matrix.shape)
print(DTW_matrix_batch.shape)
```

```
(150, 150)
(64, 64)
```

<br>

# 4. Negative Sampling, based on DTW

```python
def get_neg_indices(DTW_matrix_batch, K = 5):
    DTW_matrix_batch_norm = (DTW_matrix_batch/DTW_matrix_batch.sum(axis=0)).T
    c = DTW_matrix_batch_norm.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    if K ==1:
        indices = (u < c).argmax(axis=1)
        indices = indices.reshape(1,-1)
    else:
        indices = np.argpartition((u < c), -K)[-K:]
    
    return indices
```

<br>

```python
neg_indices = get_neg_indices(DTW_matrix_batch, K=1)
print(neg_indices.shape)
```

```
(1, 64)
```

<br>

`K` : number of negative samples

```python
neg_indices = get_neg_indices(DTW_matrix_batch, K=4)
print(neg_indices.shape)
```

```
(4, 64)
```

<br>

# 5. Sample Data

- `ts_part_ANCHOR`
- `ts_part_POS` 
- `ts_part_NEG` 

```python
start_times = np.random.choice(T-t+1, 3, replace=False)
start_anchor = start_times[0]
start_pos = start_times[1]
start_neg = start_times[2]

end_anchor = start_anchor + t 
end_pos = start_pos + t
end_neg = start_neg + t

ts_part_ANCHOR = ts_full[..., start_anchor:end_anchor] # (B, t)
ts_part_POS = ts_full[..., start_pos:end_pos]
ts_part_NEG1 = ts_full[neg_indices][..., start_neg:end_neg]
ts_part_NEG2 = ts_full[neg_indices.T][..., start_neg:end_neg]

print(ts_full.shape)
print(ts_part_ANCHOR.shape)
print(ts_part_POS.shape)
print(ts_part_NEG1.shape)
print(ts_part_NEG2.shape)
```

```
(64, 222)
(64, 10)
(64, 10)
(4, 64, 10)
(64, 4, 10)
```

