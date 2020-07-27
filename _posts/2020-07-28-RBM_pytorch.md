---
title: Restricted Boltzmann Machine 코드 (pytorch)
categories: [DL,ML,STAT]
tags: [RBM, pytorch]
excerpt: Restricted Boltzmann Machine 2
---

# Restricted Boltzmann Machine (RBM) 

https://github.com/GabrielBianconi/pytorch-rbm 를 참고하여, **pytorch**로 RBM을 구현해해보았다.

( RBM 이론 포스트에서는 언급하지 않은 momentum, weight decay, L2-regularization이 추가되어 있다. )

```python
import torch
```


```python
class RBM(): 
    def __init__(num_v,num_h,k,lr,mom_coef,w_decay,cuda=False):
        self.num_v = num_v # number of nodes in VISIBLE layer
        self.num_h = num_h # number of nodes in HIDDEN layer
        self.k  = k # k step in CD-k algorithm
        self.lr = lr # learning rate
        self.mom_coef = mom_coef # momentum coefficient
        self.w_decay = w_decay # weight decay
        self.cuda = cuda
        
        # weight(w) & bias(b)
        self.w = torch.randn(num_v,num_h) # weight between visibile & hidden layer
        self.b_v = torch.ones(num_v) / 2  # bias of visible layer
        self.b_h = torch.zeors(num_h)     # bias of hidden layer
        
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
    
    # sigmoid function
    def sig(self,x): 
        return 1 / (1+torch,exp(-x))
    
    # random probability
    def rand_prob(self,num):
        rand_prob = torch.rand(num)
        if self.cuda :
            rand_prob = rand_prob.cuda()
        return rand_prob
    
    # given Visible , sample Hidden
    def sample_h(self,prob_v):
        h_act = torch.matmul(prob_v, self.w) + self.b_h
        h_prob = self.sig(h_act)
        return h_prob
    
    # given Hidden , sample Visible
    def sample_v(self,prob_h):
        v_act = torch.matmul(prob_h, self.w.t()) + self.b_v
        v_prob = self.sig(v_act)
        return v_prob
    
    # Contrastive Divergence-k algorithm
    def CD_k(self,x):
        ## step 1) given visible node, sample the first hidden node
        pos_h_prob = self.sample_h(x)
        pos_h_act = (pos_h_prob > self.random_prob(self.num_h)).float()
        pos = torch.matmul(x.t(), pos_h_act)
        
        h_act = pos_h_act
        for _ in range(self.k):
            v_prob = self.sample_v(h_act)
            h_prob = self.sample_h(v_prob)
            h_act = (h_prob > self.rand_prob(self.num_h)).float()
        
        neg_v_prob = v_prob
        neg_h_prob = h_prob
        neg = torch.matmul(neg_v_prob.t(), neg_h_prob)
        
        self.w_mom *= self.mom_coef
        self.w_mom += (pos-neg)
        
        self.b_v_mom * self.mom_coef
        self.b_v_mom += torch.sum(x-neg_v_prob,dim=0)
        self.b_h_mom * self.mom_coef
        self.b_h_mom += torch.sum(pos_h_prob - neg_h_prob,dim=0)
        
        batch_size = x.size(0)
        self.w += self.w_mom * self.lr  / batch_size
        self.b_v += self.b_v_mom * self.lr / batch_size
        self.b_h += self.b_h_mom * self.lr / batch_size
        self.w -= self.w * self.w_decay
        
        error = torch.sum( (x-neg_v_prob)**2 )
        return error
        
```
