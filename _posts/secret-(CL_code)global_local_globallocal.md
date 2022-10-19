---
title: (code) Global, Local, Global & Local
categories: [CL, TS]
tags: []
excerpt:
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Global, Local, Global & Local

# Settings

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<br>

```python
B = 64
z_global_dim = 128
z_local_dim = 32

DTW_matrix = torch.randn((B,B))
```

<br>

# 1. Global : between TS

```python
Z = torch.randn((B,z_global_dim))
Z_norm = F.normalize(Z, p=2)
DTW_pred = (Z_norm @ Z_norm.T)
```

<br>

```python
mse_loss = nn.MSELoss()
mse_loss(DTW_pred, DTW_matrix)
```

```
tensor(1.0226)
```

<br>

# 2. Local : within TS

```python
K = 4

ts_part_ANCHOR = torch.randn(((64, z_local_dim)))
ts_part_POS = torch.randn((64, z_local_dim))
ts_part_NEG = torch.randn((K, 64, z_local_dim))
```

```python
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
```

```python
triplet_loss = 0
for k in range(K):
    triplet_loss += triplet_loss_fn(ts_part_ANCHOR, 
                                    ts_part_POS, 
                                    ts_part_NEG[k])
```

<br>

# 3. Global & Local interaction

```python
ts_pos_ANCHOR_pos = torch.cat([ts_part_ANCHOR,ts_part_POS], dim=1)

linear_model = nn.Linear(2*z_local_dim, z_global_dim)
aggregated = linear_model(ts_pos_ANCHOR_pos)
aggregated_norm = F.normalize(aggregated, p=2)
```

<br>

```python
pred = (aggregated_norm@Z_norm.T)
pred = F.softmax(pred,dim=1)

bce_loss_fn = nn.BCELoss()
bce_loss = bce_loss_fn(pred, torch.eye(B))
```

