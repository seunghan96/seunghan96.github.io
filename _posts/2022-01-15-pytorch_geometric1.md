---
title: Pytorch Geometric 1
categories: [GNN, DLF]
tags: [GNN]
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# PyTorch Geometric 1 - Intro

( 참고 : https://baeseongsu.github.io/posts/pytorch-geometric-introduction/ )

<br>

# 1. 그래프 데이터 다루기

## (1) torch_geometric.data.Data

$$G = (V,E)$$.

하나의 그래프는 `torch_geometric.data.Data` 클래스로 표현됨

<br>

Notation

- $$N_v$$ : number of nodes
- $$d_v$$ : number of node features
- $$N_e$$ : number of edges
- $$d_e$$ : number of edge features

<br> 

## (2) attribute

해당 클래스의 속성에는…

- `data.x` : node feature
  - shape : $$(N_v, d_v)$$
- `data.edge_index` : connectivity info
  - shape : $$(2, N_e)$$
- `data.edge_attr` : edge attribute
  - shape : $$(N_e, d_e)$$
- `data.y` : label (target)
  - shape : 
    - node label : $$(N_v, *)$$
    - graph label : $$(1,*)$$
- `data.pos` : node position
  - shape : $$(N_v, \text{num dimension})$$

<br>

### Example

```python
import torch
from torch_geometric.data import Data

# number of edges = 4
# number of nodes = 3
# node feature dimension : 1

#------------------------------------------------------#
# edge_index : (2,4)
edge_index = [[0, 1, 1, 2],
             [1, 0, 2, 1]]
edge_index = torch.tensor(edge_index, dtype=torch.long) # 정수 형태

# x (node feature) : (3,1)
x = [[-1],[0],[1]]
x = torch.tensor(x, dtype = torch.float)
#------------------------------------------------------#

data = Data(x = x, 
            edge_index = edge_index)
```

<br>

## (3) Method

- `data.keys` : name of feature
- `data.num_nodes` : $$N_v$$
- `data.num_edges` : $$N_e$$
- `data.contains_isolated_nodes()` : wheter there is isolated nodes
- `data.contains_self_loops()` : wheter there is self-loop 
- `data.is_directed()` : wheter the graph is directed graph

<br>

# 2. 벤치마크 데이터셋 :  [torch_geometric.datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

대표적인 데이터셋 : ENZYMES, Cora

<br>

## (1) ENZYMES 데이터셋

특징

- (1) 600개의 그래프
- (2) 6개의 노드 class 종류
- (3) 3차원의 node feature

<br>

1. **Import Dataset**

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
```

<br>

2. **Check attributes**

```python
# (1) Graph의 개수
len(dataset)
>>> 600

# (2) Node의 클래스 개수
dataset.num_classes
>>> 6

# (3) Node의 feature 차원
dataset.num_node_features
>>> 3
```

<br>

3. **Check specific graph**

```python
# (1) 1번째 그래프 가져오기
## edge 개수 : 168개
## node 개수 : 37개
## node 특징 차원 : 3개
## label : "graph" 단위
data = dataset[0]
>>> Data(edge_index=[2, 168], x=[37, 3], y=[1])

# (2) directed graph 여부 확인
data.is_undirected()
>>> True

# (3) 1~540번 그래프 가져오기
train_dataset = dataset[:540]
>>> ENZYMES(540)

# (4) 541~600번 그래프 가져오기
test_dataset = dataset[540:]
>>> ENZYMES(60)

# (5) 그래프 shuffle
dataset = dataset.shuffle()
>>> ENZYMES(600)
```

<br>

## (2) Cora 데이터셋

특징

- (1) 1개의 그래프
- (2) 7개의 노드 class 종류
- (3) 1433차원의 node feature

<br>

상세 특징

- 2708개의 “scientific publications”로 구성
- edge : **“인용 여부” ( = Citation Network )**
- node : **논문**
- node feature : 논문에서 자주 등장하는 1433개의 단어의 등장 여부 ( 1/ 0 )

<br>

1. Import Dataset

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

<br>

2. Check attributes

```python
# (1) Graph의 개수
len(dataset)
>>> 1

# (2) Node의 클래스 개수
dataset.num_classes
>>> 7

# (3) Node의 feature 차원
dataset.num_node_features
>>> 1433
```

<br>

3. Check specific graph

```python
# (1) 1번째 그래프 가져오기
data = dataset[0]
>>> Data(edge_index=[2, 10556], test_mask=[2708],
         train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

# (2) directed graph 여부 확인
data.is_undirected()
>>> True

# (3) 학습(train)을 위한 마스크(1/0)
data.train_mask.sum().item()
>>> 140

# (4) 검증(validation)을 위한 마스크(1/0)
data.val_mask.sum().item()
>>> 500

# (5) 테스트(test)을 위한 마스크(1/0)
data.test_mask.sum().item()
>>> 1000
```

<br>

# 3. 미니 배치

## (1) 미니 배치

**Sparse block diagonal adjacency matrices**를 통해 mini-batch 구성

<br>

ex) 1개의 batch

$$\mathbf{A}=\left[\begin{array}{ccc}
\mathbf{A}_{1} & & \\
& \ddots & \\
& & \mathbf{A}_{n}
\end{array}\right], \quad \mathbf{X}=\left[\begin{array}{c}
\mathbf{X}_{1} \\
\vdots \\
\mathbf{X}_{n}
\end{array}\right], \quad \mathbf{Y}=\left[\begin{array}{c}
\mathbf{Y}_{1} \\
\vdots \\
\mathbf{Y}_{n}
\end{array}\right]$$.

- 하나의 배치 : $$\left(A_{i}, X_{i}, Y_{i}, i \in\{1, \ldots, n\}\right)$$ 
- ***주의*** : 만약 batch size가 64라면,
  - node가 64개 (X)
  - **graph자체가 64개 (O)**


<br>

## (2) Data loader

1. Import dataset & data loader

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, 
                    batch_size=32, 
                    shuffle=True)
```

<br>

2. **Check data loader**

```python
for batch in loader:
    batch
    # batch size = 32 ( 32개의 그래프 )
    # node 개수 = 1082
    # edge 개수 = 4066
    # node feature = 21 ( 모든 그래프 동일 )
    # label : graph 단위
    >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    >>> 32
```

<br>

# 4. 데이터 변환

- `torchvision` : `torchvision.transforms`

  = `torch_geometric` : `torch_geometric.transforms`

- `torch_geometric.transforms.Compose`를 통해, pipeline 식으로 변환 가능

<br>

## (1) Example : `ShapeNet` 데이터셋

1. **Import Packages & Dataset**

```python
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/tmp/ShapeNet', 
                   categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6),
                   transform=T.RandomTranslate(0.01))
```

데이터를 애초에 불러올 때 부터, **변환**을 해서 불러올 수 있다.

- `pre_transform` : 
  - `pre_transform = T.KNNGraph(k=6)` : KNN 통해, 그래프 생성 ( 6개 )
- `transform` : 그래프 형성 후에 transform
  - `transform = T.RandomTranslate(0.01)` : node의 위치를 살짝 이동 ( perturbation )

<br>

2. **Check speific graph**

```python
dataset[0]
>>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
```

- 1번째 그래프 확인
  - edge 개수 : 15,108
  - node 개수 : 2,518
  - node feature 개수 : 3

<br>

# 5. 그래프로 학습하기 (GNN)

Task : **Graph Node CLASSIFICATION**

- (1) 특정 논문 내에 등장하는 단어 **(node feature)**
- (2) 논문들 사이의 인용 관계 **(adjacency matrix)**

를 통해, 어떠한 논문인지 맞추기

<br>

1. Import dataset

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

<br>

2. Build GNN

```python
import torch
import torch.nn.functional as F

# Graph Convolution Network
from torch_geometric.nn import GCNConv
```

<br>

모델링

- 1번째 layer : GCN layer
  - shape : $$(d_v, \text{middle dim})$$
- 2번째 layer : GCN layer
  - shape : $$(\text{middle dim}, \text{class 개수})$$ 

```python
class GNN(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        # [x] Node Feature (N_v, d_v)
        # [edge_index] Adjacency Info (2, N_e)
        x, edge_index = data.x, data.edge_index
				
        # (GCN-relu-dropout)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # (GCN-softmax)
        x = self.conv2(x, edge_index)
        out = F.log_softmax(x, dim=1)
        return out

```

<br>

3. **Train Model**

```python
# 1) check device (cpu/gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) send MODEL & DATA to device
model = GNN().to(device)
data = dataset[0].to(device)

# 3) build Optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01, 
                             weight_decay=5e-4)
```

<br>

Binary classification task!

- loss function : **“negative log likelihood”**

  ( `F.nll_loss(y_pred, y_true)` )

```python
# 4) Train model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(data)
    loss = F.nll_loss(y_pred[data.train_mask],
                      data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

<br>

4. **Evaluation**

```python
model.eval()

_, y_pred_class = model(data).max(dim=1) # (1) max value & (2) argmax
correct = float (y_pred_class[data.test_mask].eq(data.y[data.test_mask]).sum().item())

acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
>>> Accuracy: 0.8150
```

