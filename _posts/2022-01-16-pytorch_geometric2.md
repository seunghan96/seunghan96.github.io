---
title: Pytorch Geometric 2
categories: [GNN, DLF]
tags: [GNN]
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# PyTorch Geometric 2 - Message Passing

# 1. Graph Data in PyG

컴퓨터가 **edge**를 저장하는 방식 : **“indicies” ( shape = 2 x edge 개수 )**

따라서, 컴퓨터가 그래프 $$G=(V,E)$$를 표현하기 위해,

- **(1) FEATURE Matrix**
- **(2) INDICIES Matrix**

로 나타냄

<br>

Pytorch Geometric이 그래프를 정의하는 방식 : $$G = (X,(I,E))$$

- $$X$$ : Node feature ( $$\mathrm{X} \in \mathbb{R}^{ \mid V \mid  \times F}$$ ) 
- $$I$$ : Edge indices ( $$\mathrm{I} \in\{1, \ldots, N\}^{2 \times \mid E \mid }$$ )
- $$E$$ : Edge features ( $$\mathrm{E} \in \mathbb{R}^{ \mid E \mid  \times D}$$ )

<br>

# 2. Message Passing scheme

Target Node를 임베딩하는 방식 

$$\mathbf{x}_{i}^{(k)}=\gamma^{(k)}\left(x_{i}^{(k-1)}, \square_{j \in N(i)} \phi^{(k)}\left(\mathbf{x}_{i}^{(k-1)}, \mathbf{x}_{j}^{(k-1)}, \mathbf{e}_{j, i}\right)\right)$$.

- $$x$$ : Node embedding
  - $$k$$ : layer index
- $$e$$ : Edge feature
- Functions
  - (1) $$\phi$$  : **Message function**
  - (2) $$\square$$ : **Aggregation function**
    - ex) uniform, GAT, …
  - (3) $$\gamma$$ : **Update function**

<br>

# 3. Message Passing class

Class : `torch_geometric.nn.MessagePassing`

핵심

- (1) forward
- (2) message
- (3) propagate

<br>

Custom Message Passing 구성하기!

- `MessagePassing`을 상속받아, 새로운 class로 구성
- **새롭게 구현(override)할 메소드**
  - (1) `forward`
  - (2) `message`
- (3) `aggr` : 선택 가능한 aggregation 방법
  - Ex) add (default), mean, max 

<br>

`propagate`

- `MessagePassing` class가 가지고 있는 메소드 중 하나로,

- 아래의 인자들을 받는다.

  - args1 : edge index ( `edge_index` )
  - args2 : node feature (`x` )
  - args3 : edge feature ( `e` )

  ( `forawrd`도 위와 동일한 인자를 받는다 )

<br>

Example

```python
class MyOwnConv(MessagePassing):
    def __init__(self):
        super(MyOwnConv, self).__init__(aggr='add')
        #super(MyOwnConv, self).__init__(aggr='mean')
        #super(MyOwnConv, self).__init__(aggr='max')
        
    def forward(self, x, edge_index, e):
        return self.propagate(edge_index, x=x, e=e) 
    
    def message(self, x_j, x_i, e): 
      	# x_j,x_i : j,i번째 node의 feature들
        # e : edge feature
        return x_j * e 
```

<br>

# 4. Message Passing class ( details )

## (1) add & flow

`torch_geometric.nn.MessagePassing(aggr="add", flow="source_to_target")`

- `aggr` : 생성된 메세지들을 **aggregate하는 방식**

  - ex) add, mean, max

- `flow` : 메세지의 흐르는 **방향**

  - ex) source_to_target, target_to_source
    - **source_to_target** : 주변으로부터 전달 “받음”
    - **target_to_source** : 주변으로 전달 “함”

- `node_dim` : node dimension

  - 여기서 차원은, “특징의 개수”를 의미하는 차원이 아니라,

    **메세지를 전달할 “축”**을 의미함

  - (default) 0

<br>

## (2) propagate

위에서도 간단히 언급을 했지만, 보다 자세히 설명하자면…

`propagate(edge_index, size=None, **kwargs)`

<br>

**“forward 함수”**를,

- **”propagate 함수”**를 사용하여 구성하고,
  - propagate 함수는 내부적으로 **“message / aggregate 함수”**와 **“update 함수”**를 사용한다

<br>

( *source code* )

P = M+A+U

- P : Propagate

- M : message
- A : aggregate
- U : Update

```python
def propagate():
  # (방법 1) MP & AG 동시
  if ..........:
    out = self.message_and_aggregate(xxx)
  
  # (방법 2) MP & AG 따로
  else:
    out = self.message(xxx)
    out = self.aggregate(out,xxx)
    
 final_output = self.update(out,xxxx)
```

<br>

## (3) message

```python
def message(self, x_j : torch.Tensor) -> torch.Tensor:
  # write your code here~
  return x_j
```

- propagate 함수 실행 시, message 함수가 호출 됨
- **Node feature(들)**을 인자로 받는다

<br>

## (4) update

```python
def update(self, inputs : torch.Tensor) :
  # write your code here~
  return inputs
```

- node embedding을 update함
- **out = message(x),**
  **out = aggregate(out),** 
  를 거쳐서 나온 **“aggregated message”를 input으로 받는다**

<br>

# 5. ex) GCN

**GCN의 Message Passing :**

$$\mathbf{x}_{i}^{(k)}=\sum_{j \in \mathcal{N}(i) \cup i} \frac{1}{\sqrt{\operatorname{deg}(i)} \cdot \sqrt{\operatorname{deg}(j)}} \cdot\left(\boldsymbol{\Theta} \cdot \mathbf{x}_{j}^{(k-1)}\right)$$.

<br>

## 진행 순서

***( 1~3 : message 생성 )***

***( 4 : message 취합+전파 )***

1. Self-loop 추가
   - neighbor 뿐만 아니라, **target node 스스로의 feature도 인풋으로** 받기 때문에
2. Weight matrix 곱하기
3. Normalization coefficient 계산
4. Propagate message
   1. [M] message 함수 ( + normalize 하기 )
   2. [A] aggregation 함수
   3. [U] update 함수

<br>

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        # ( in_channels, out_channels )
        self.lin = nn.Linear(in_Channels, out_channels) 
        
    def forward(self, x, edge_index):
        # x : ( N_v, in_channels )
        # edge_index : ( 2, N_e )
        #===================================================#
        # [Step 1] Adjacency matrix에 self loop 추가
        # edge_index : ( 2, N_e + N_v )
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        #===================================================#
        # [Step 2] node feature에 weight matrix 곱하기
        # x : ( N_v, in_channels )
        x = self.lin(X)
        # x : ( N_v, out_channels )
        
        #===================================================#
        # [Step 3] Compute normalization coef
        row, col = edge_index  
        # row : edge의 왼쪽 ( shape : (N_e + N_v) )
        # col : edge의 오른쪽 ( shape : (N_e + N_v) )
        #----------------------------------------------------#
        deg = degree(row, x.size(0), dtype=x.dtype) 
        # deg : 모든 node들의 degree ( shape : (N_v) )
        deg_inv_sqrt = deg.pow(-0.5)
        # deg : ( shape : (N_v) )
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # norm : ( shape : (N_e + N_v) )
        
        #===================================================#
        # [Step 4] Propagate message
        ## 4-1) message 함수 ( + normalize 하기 )
        ## 4-2) aggregation 함수
        ## 4-3) update 함수
        return self.propagate(edge_index, 
                              size = (x.size(0), x.size(0)),
                              x = x,
                              norm = norm)
    
    def message(self, x_j, norm):
      	## 4-1) message 함수 ( + normalize 하기 )
        # norm : (N_e + N_v)
        # norm.view(-1,1) : (N_e + N_v, 1)
        # x_j : (N_e + N_v, out_channels)
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
      	## 4-3) update 함수
        # aggr_out : (N_v, out_channels)
        return aggr_out 

```

<br>

```python
conv = GCNConv(in_channels=16, out_channels=32)
x = conv(x=x, edge_index=edge_index)
```

