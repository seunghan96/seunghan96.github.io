---
title: Implementing DCGAN with Pytorch
categories: [DL,GAN]
tags: [Deep Learning, DCGAN]
excerpt: DCGAN
---

# DCGAN Implementation with Pytorch

https://towardsdatascience.com/how-to-build-a-dcgan-with-pytorch-31bfbf2ad96a 를 참고하여 작성하였습니다. 

## 1. Import Packages


```python
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn
import numpy as np
```



## 2. Generator


```python
class G(nn.Module):
    def __init__(self,depth,dim,dropout,window,input_dim,output_depth):
        super(G,self).__init__()
        self.depth = depth
        self.dim = dim
        self.dropout = dropout
        self.window = window
        self.input_dim =input_dim
        self.ouptut_depth = output_depth
        self._init_modules()
    
    def __init__modules(self):
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.ln1 = nn.Linear(self.input_dim,self.depth*self.dim*self.dim,bias=True)
        self.bn1 = nn.BatchNorm1d(self.depth*self.dim*self.dim)
        self.drop1 = nn.Dropout(self.dropout)
        
        self.conv2 = nn.Conv2d(self.depth,self.depth//2,kernel_size=self.window,
                              stride=1,padding=2,bias=False)
        self.bn2 = nn.BatchNorm2d(self.depth//2)
        self.drop2 = nn.Dropout2d(self.dropout)
        
        self.conv3 = nn.ConvTranspose2d(self.depth//2,self.depth//4,kernel_size=self.window-1,
                                       stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.depth//4)
        self.drop3 = nn.Dropout2d(self.dropout)
        
        self.conv4 = nn.ConvTranspose2d(self.depth//4,1,kernel_size=self.window-1,
                                       stride=2,padding=1,bias=False)
        
    def forward(self,x,input_dim=256,dim=7):
        x = self.leaky_relu(self.bn1(self.ln1(x)))
        x = x.view((-1,input_dim,dim,dim))
        x = self.drop1(x)
        
        xx = self.relu(self.bn2(self.conv2(x)))                
        xx = self.drop2(xx)
        
        xxx = self.leaky_relu(self.bn3(self.conv2(xx)))        
        xxx = self.drop3(xxx)
                
        output = self.tanh(self.conv4(xxx))
        return output    
        
        
```



## 3. Discriminator


```python
class D(nn.Module):
    def __init__(self,depth,dim,dropout,window,input_dim,output_depth):
        super(D, self).__init__()
        self.depth = depth
        self.dim = dim
        self.dropout = dropout
        self.window = window
        self.input_dim =input_dim
        self.ouptut_depth = output_depth
        self._init_modules()
    
    def __init__modules(self):
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.conv1 = nn.Conv2d(1,self.depth//4,kernel_size=self.window,
                              stride=2,padding=2,bias=True)
        self.drop1 = nn.Dropout2d()
        
        self.conv2 = nn.Conv2d(self.depth//4,self.depth//2,kernel_size=self.window,
                              stride=2,padding=2,bias=True)
        self.drop2 = nn.Dropout2d()
        
        self.lin3 = nn.Linear(self.depth//2,1,bias=True)
        
    def forward(x,input_dim=256,dim=7):
        x = self.leaky_relu(self.conv1(x))
        x = self.drop1(x)
        
        xx = self.relu(self.conv2(x))
        xx = self.drop2(xx)
        xx = xx.view((-1,input_dim/2,dim,dim))
        
        output = self.sigmoid(self.lin3(xx))
        return output
```



## 4. Training DCGAN


```python
class DCGAN():    
    def __init__(self, depth_dim,dropout,window,input_dim,output_depth,
                 noise_fn,dataloader,device='cpu',batch_size=128,lr_D=1e-3, lr_G=2e-4):
        self.Gen = G(depth,dim,dropout,window,input_dim,output_depth).to(device)
        self.Dis = D(depth,dim,dropout,window,input_dim,output_depth).to(device)
        self.noise_fn = noise_fn
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.BCELoss() # Binary Cross Entropy
        self.optim_D = optim.Adam(self.Dis.parameters(),
                                  lr=lr_D, betas=(0.5, 0.999))
        self.optim_G = optim.Adam(self.Gen.parameters(),
                                  lr=lr_G, betas=(0.5, 0.999))
        self.target_ones = torch.ones((batch_size, 1), device=device)
        self.target_zeros = torch.zeros((batch_size, 1), device=device)

    def gen_samples(self, latent_vec=None):       
        # (number, channels, height, width)
        num = self.batch_size
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.Gen(latent_vec)
        samples = samples.cpu()  
        return samples

    def train_G(self):        
        self.Gen.zero_grad()
        
        latent_vec = self.noise_fn(self.batch_size)
        generated = self.Gen(latent_vec)
        classified = self.Dis(generated)
        
        loss = self.criterion(classified, self.target_ones)
        loss.backward()
        self.optim_G.step()
        return loss.item()

    def train_D(self, real_samples):        
        self.Dis.zero_grad()        
        
        # [Loss 1] Predict "real" as "REAL" ?
        pred_real = self.Dis(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)
        
        # [Loss 2] Predict "false" as "FALSE" ?
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.Gen(latent_vec)
        pred_fake = self.Dis(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # [Loss 1] + [Loss 2]
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_D.step()
        return loss_real.item(), loss_fake.item()

    def train(self, check=10, max_steps=0):
        loss_G, loss_D_real, loss_D_fake = 0, 0, 0
        
        for batch_idx, (real_samples, label) in enumerate(self.dataloader):
            real_samples = real_samples.to(self.device)
            loss_d_real, loss_f_real = self.train_D(real_samples)
            
            loss_D_real += loss_d_real
            loss_D_fake += loss_f_real
            loss_G += self.train_G()
            
            if check and (batch_idx+1) % check == 0:
                print(f"{batch_idx+1}/{len(self.dataloader)}:"
                      f" G={loss_G / (batch_idx+1):.3f},"
                      f" D_real={loss_D_real / (batch_idx+1):.3f},"
                      f" D_fake={loss_D_fake / (batch_idx+1):.3f}",
                      end='\r',
                      flush=True)
            if max_steps and batch == max_steps:
                break
                
        if check:
            print("----------------------------------------")
        loss_G /= batch_idx
        loss_D_real /= batch_idx
        loss_D_fake /= batch_idx
        return (loss_G, (loss_D_real, loss_D_fake))
```
