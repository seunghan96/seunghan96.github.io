---
title: Restricted Boltzmann Machine 코드 (pytorch)
categories: [DL,ML,STAT]
tags: [RBM, pytorch]
excerpt: Restricted Boltzmann Machine 2, Movie Recommendation
---

# Restricted Boltzmann Machine (RBM) 

https://github.com/GabrielBianconi/pytorch-rbm 를 참고하여, **pytorch**로 RBM을 구현해해보았다.

( RBM 이론 포스트에서는 언급하지 않은 momentum, weight decay, L2-regularization이 추가되어 있다. )

구현한 모델을 사용하여, user가 남긴 각각의 movie에 대한 rating을 바탕으로 Movie Recommendation을 해줄 

것이다.



## 1. Import Data,Libraries 


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```


```python
train = pd.read_csv('C:\\Users\\samsung\\Downloads\\u1.base.csv',sep='\t',header=None)
test = pd.read_csv('C:\\Users\\samsung\\Downloads\\u1.test.csv',sep='\t',header=None)
train.columns = ['user','movie','rating','time']
test.columns = ['user','movie','rating','time']
```



## 2. Data Preprocessing


```python
def convert_table(data):
    table = pd.pivot_table(data, values='rating',index=['user'],columns=['movie'])
    table[table<3]=0
    table[table>=3]=1
    table.fillna(0,inplace=True)
    return table
```


```python
train_t = convert_table(train)
test_t = convert_table(test)
```


```python
train_t.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movie</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1650 columns</p>
</div>


```python
Train = np.array(train_t,dtype='int')
Test = np.array(test_t,dtype='int')
```


```python
Train.shape, Test.shape
```


    ((943, 1650), (459, 1410))



## 3. Restricted Boltzmann Machine


```python
import torch
from torch.utils.data import TensorDataset, DataLoader
```


```python
train_tensor = torch.Tensor(Train) 
train_tensor2 = TensorDataset(train_tensor) 
train_data = DataLoader(train_tensor,batch_size=32)
```


```python
class RBM():
    def __init__(self,num_v,num_h,k,lr,mom_coef,w_decay,cuda=False):
        self.num_v = num_v
        self.num_h = num_h
        self.k  = k
        self.lr = lr
        self.mom_coef = mom_coef
        self.w_decay = w_decay
        self.cuda = cuda
        
        # weight(w) & bias(b)
        self.w = torch.randn(num_v,num_h)*0.1
        self.b_v = torch.ones(num_v) *0.5
        self.b_h = torch.zeros(num_h)
        
        # momentum(mom)
        self.w_mom = torch.zeros(num_v,num_h)        
        self.b_v_mom = torch.zeros(num_v)
        self.b_h_mom = torch.zeros(num_h)
        
        if self.cuda:
            self.w = self.w.cuda()
            self.b_v = self.b_v.cuda()
            self.b_h = self.b_h.cuda()

            self.w_mom = self.w_mom.cuda()
            self.b_v_mom = self.b_v_mom.cuda()
            self.b_h_mom = self.b_h_mom.cuda()
    
    def sig(self,x):
        return 1 / (1+torch.exp(-x))
    
    def rand_prob(self,num):
        rand_prob = torch.rand(num)
        if self.cuda :
            rand_prob = rand_prob.cuda()
        return rand_prob
    
    def sample_h(self,prob_v):
        h_act = torch.matmul(prob_v, self.w) + self.b_h
        h_prob = self.sig(h_act)
        return h_prob
    
    def sample_v(self,prob_h):
        v_act = torch.matmul(prob_h, self.w.t()) + self.b_v
        v_prob = self.sig(v_act)
        return v_prob
    
    def CD_k(self,x):
        pos_h_prob = self.sample_h(x)
        pos_h_act = (pos_h_prob > self.rand_prob(self.num_h)).float()
        pos = torch.matmul(x.t(), pos_h_act)
        
        h_act = pos_h_act
        for _ in range(self.k):
            v_prob = self.sample_v(h_act)
            h_prob = self.sample_h(v_prob)
            h_act = (h_prob >= self.rand_prob(self.num_h)).float()
        
        neg_v_prob = v_prob
        neg_h_prob = h_prob
        neg = torch.matmul(neg_v_prob.t(), neg_h_prob)
        
        self.w_mom *= self.mom_coef
        self.w_mom += (pos-neg)
        
        self.b_v_mom *= self.mom_coef
        self.b_v_mom += torch.sum(x-neg_v_prob,dim=0)
        self.b_h_mom *= self.mom_coef
        self.b_h_mom += torch.sum(pos_h_prob - neg_h_prob,dim=0)
        
        batch_size = x.size(0)
        self.w += self.w_mom * self.lr  / batch_size
        self.b_v += self.b_v_mom * self.lr / batch_size
        self.b_h += self.b_h_mom * self.lr / batch_size
        self.w -= self.w * self.w_decay
        
        error = torch.sum( (x-neg_v_prob)**2 )
        return error
        
```


```python
num_v = len(train_tensor[0])
num_h = 200
batch_size=64
k = 5
lr = 0.001
mom_coef = 0.9
w_decay = 0.001
epochs = 30
cuda = False
```


```python
rbm = RBM(num_v,num_h,k,lr,mom_coef,w_decay,cuda)
```


```python
for epoch in range(1,epochs+1):
    epoch_error = 0
    for batch in train_data:        
        batch = batch.view(len(batch),num_v)
        if cuda:
            batch = batch.cuda()

        batch_error = rbm.CD_k(batch)
        epoch_error += batch_error
    if epoch%5==0:
        print('Error (epoch=%d): %.4f' % (epoch, epoch_error))
```


```python

```
