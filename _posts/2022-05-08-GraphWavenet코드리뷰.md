---
title: GraphWavenet 코드 리뷰
categories: [GNN, TS]
tags: []
excerpt: pytorch, pytorch geometric
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# GraphWavenet 코드 리뷰

( 논문 리뷰 :  https://seunghan96.github.io/ts/gnn/ts34/  )

<br>

# 1. model.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
```

<br>

1. `nconv` : matrix multiplication ( of $$x$$ & $$A$$ )

```python
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

```

<br>

2. `linear` : 1x1 convolution
   - shape 맞춰주기 위해

```python
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out,
                                   kernel_size=(1, 1), 
                                   padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
```

<br>

3. `gcn` : graph convolution network

   - 사전 graph strutcture가
     - (있을 경우) $$\mathbf{Z}=\sum_{k=0}^{K} \mathbf{P}_{f}^{k} \mathbf{X} \mathbf{W}_{k 1}+\mathbf{P}_{b}^{k} \mathbf{X} \mathbf{W}_{k 2}+\tilde{\mathbf{A}}_{a p t}^{k} \mathbf{X} \mathbf{W}_{k 3}$$
     - (없을 경우) $$\mathbf{Z}+\tilde{\mathbf{A}}_{a p t}^{k} \mathbf{X} \mathbf{W}_{k 3}$$

   - 여기서 $$\mathbf{P}_{f}^{k}$$ 관련된 정보는 support에 담겨 있을 것 ( 없으면 empty list )

   - 위의 $$\mathbf{Z}$$ 자체를 여러개 쌓을 수 있음.

     여러개 쌓고, 1x1 convolution으로 크기 다시 맞춰주기!

```python
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x] 
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
```

<br>

`gwnet` : Graph WaveNet

![image-20220327140852813](/Users/LSH/Library/Application Support/typora-user-images/image-20220327140852813.png)

```python
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, 
                 supports=None, gcn_bool=True, addaptadj=True, aptinit=None, 
                 in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        #--------------------------------------------------------------------------------------#
        # [1. 기본 정보]
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool # GCN 여부
        self.addaptadj = addaptadj # Adaptive Adj Matrix 여부
        self.supports = supports # 사전 graph structure 정보
        
				#--------------------------------------------------------------------------------------#
        # [2. TCN]
        self.filter_convs = nn.ModuleList() # dilated convolution 필터 - (a) value 용
        self.gate_convs = nn.ModuleList() # dilated convolution 필터 - (b) gate 용
        
        #--------------------------------------------------------------------------------------#
        # [3. GCN]
        self.gconv = nn.ModuleList() # Adaptive Adjacency Matrix 계산 & GCN
        
        #--------------------------------------------------------------------------------------#
        # [4. etc]
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1)) # (1x1. 차원맞추기용)
        self.residual_convs = nn.ModuleList() # (1x1. 차원맞추기용)
        self.skip_convs = nn.ModuleList() # (1x1. 차원맞추기용)
        self.bn = nn.ModuleList() # Batch Normalization
        
        #--------------------------------------------------------------------------------------#
        # [5. layer 쌓기]
        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
				
        ## (5-1) Node별 embedding vector 생성
        if gcn_bool and addaptadj:
            # (initial 값 지정 X .... random)
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
                
						# (initial 값 지정 O .... SVD)
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # [TCN] dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # [GCN]
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                    
                # [1x1] residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # [1x1] skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                
                # [Batch Norm]
                self.bn.append(nn.BatchNorm2d(residual_channels))
  
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        #-----------------------------------------------------------------------#
        # (1) zero-padding
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
            
				#-----------------------------------------------------------------------#
        # (2) 1x1 conv로 차원 맞춰주고 시작
        x = self.start_conv(x)
        skip = 0

        #-----------------------------------------------------------------------#
        # (3) 매 iteration마다, 현재 step에 맞는 "Adaptive Adjacency matrix" 계산
        ## (3-1) self.support : 사전에 가지고 있는 graph structure ( 매 step 동일 )
        ## (3-1) adp : 매 step 마다 계산되는 Adaptive Adjacency matrix
        ## (3-1) & (3-2)가 합쳐져서 new_support를 생성 -> input으로 들
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        #-----------------------------------------------------------------------#
        # (4) WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, dilation, init_dilation, i)
            
            #---------------------------------------------------------#
            # (4-1) TCN ( dilated convolution )
            residual = x
            ## [ TCN-a ]
            filter = self.filter_convs[i](residual) 
            filter = torch.tanh(filter)
            ## [ TCN-b ]
            gate = self.gate_convs[i](residual) 
            gate = torch.sigmoid(gate)
            ## [ TCN-a & TCN-b ]
            x = filter * gate
            
            #---------------------------------------------------------#
            # (4-2) (Parameterized) Skip connection
            s = x
            s = self.skip_convs[i](s) # 차원 맞춰주기용
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

						#---------------------------------------------------------#
            # (4-3) GCN ( adpative adjacency matrix 계산 )
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj: # case 1) adaptive (O)
                    x = self.gconv[i](x, new_supports)
                else: # case 2) adaptive (X)
                    x = self.gconv[i](x,self.supports)
            else: # case 3) GCN 자체를 X
                x = self.residual_convs[i](x)
						
            #---------------------------------------------------------#
            # (4-4) Residual Connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

```

<br>

# 2. `engine.py`

![image-20220327142821564](/Users/LSH/Library/Application Support/typora-user-images/image-20220327142821564.png)

```python
import torch.optim as optim
from model import *
import util
```



```python
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, 
                 nhid , dropout, lrate, wdecay, device,
                 supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                           gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
```

