# SOM-VAE

## EXPERIMENTS 

We performed experiments on MNIST handwritten digits (LeCun et al., 1998), Fashion-MNIST images of clothing (Xiao et al., 2017), synthetic time series of linear interpolations of those images, time series from a chaotic dynamical system and real world medical data from the eICU Collaborative Research Database (Goldberger et al., 2000). If not otherwise noted, we use the same architecture for all experiments, sometimes including the latent probabilistic model (SOM-VAE_prob) and sometimes excluding it (SOM-VAE). For model implementation details, we refer to the appendix (Sec. B)1 . We found that our method achieves a superior clustering performance compared to other methods. We also show that we can learn a temporal probabilistic model concurrently with the clustering, which is on par with the maximum likelihood solution, while improving the clustering performance. Moreover, we can learn interpretable state representations of a chaotic dynamical system and discover patterns in real medical data.



## CLUSTERING ON MNIST AND FASHION-MNIST 

In order to test the clustering component of the SOM-VAE, we performed experiments on MNIST and FashionMNIST. We compare our model (including different adjustments to the loss function) against k-means (Lloyd, 1982) (sklearn-package (Pedregosa et al., 2011)), the VQ-VAE (van den Oord et al., 2017), a standard implementation of a SOM (minisom-package (Vettigli, 2017)) and our version of a GB-SOM (gradient-based SOM), which is a SOM-VAE where the encoder and decoder are set to be identity functions. The k-means algorithm was initialized using k-means++ (Arthur and Vassilvitskii, 2007). To ensure comparability of the performance measures, we used the same number of clusters (i.e. the same k) for all the methods. The results of the experiment in terms of purity and normalized mutual information (NMI) are shown in Table 1. The SOM-VAE outperforms the other methods w.r.t. the clustering performance measures. It should be noted here that while k-means is a strong baseline, it is not density matching, i.e. the density of cluster centers is not proportional to the density of data points. Hence, the representation of data in a space induced by the k-means clusters can be misleading. As argued in the appendix (Sec. C), NMI is a more balanced measure for clustering performance than purity. If one uses 512 embeddings in the SOM, one gets a lower NMI due to the penalty term for the number of clusters, but it yields an interpretable two-dimensional representation of the manifolds of MNIST (Fig. 2, Supp. Fig. S4) and Fashion-MNIST (Supp. Fig. S5). The experiment shows that the SOM in our architecture improves the clustering (SOM-VAE vs. VQ-VAE) and that the VAE does so as well (SOM-VAE vs. GB-SOM). Both parts of the model therefore seem to be beneficial for our task. It also becomes apparent that our reconstruction loss term on ze works better in practice than the gradient copying trick from the VQ-VAE (SOM-VAE vs. gradcopy), due to the reasons described in Section 2.2. If one removes the ze reconstruction loss and does not copy the gradients, the encoder network does not receive any gradient information any more and the learning fails completely (no_grads). Another interesting observation is that stochastically optimizing our SOM loss using Adam (Kingma and Ba, 2014) seems to discover a more performant solution than the classical SOM algorithm (GB-SOM vs. minisom). This could be due to the dependency of the step size on the distance between embeddings and encodings, as described in Section 2.1. Since k-means seems to be the strongest competitor, we are including it as a reference baseline in the following experiments as well.



![image-20210929154545210](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20210929154545210.png).

## 1. Import Packages

```python
import pandas as pd
import numpy as np
import matplotlib as plt
%pylab inline

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
```



## 2. Model

### 2-1. Encoder

```python
class ConvEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim=64, kernel_size=[4, 4], strides=[1, 1], n_channels=[32, 128]):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, n_channels[0], kernel_size[0], stride=strides[0])
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_channels[0], n_channels[1], kernel_size[1], stride=strides[1])
        self.maxpool2 = nn.MaxPool2d(2, 2)
        flat_size = 4*4*n_channels[-1]
        self.latent_transform = nn.Linear(flat_size, latent_dim)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        out = self.latent_transform(x)
        return out
```

<br>

## 2-2. Decoder

```python
class ConvDecoder(nn.Module):
    def __init__(self, output_channels, output_size=(28, 28), latent_dim=64, flatten_shape=[4, 4, 128],
                 kernel_size=[4, 3], strides=[1, 1], n_channels=[128, 128]):
        super().__init__()
        self.latent_transform = nn.Linear(latent_dim,
                                      flatten_shape[0]*flatten_shape[1]*flatten_shape[2])
        self.deconv1 = nn.Conv2d(flatten_shape[-1], n_channels[0], 
                                 kernel_size[0], stride=strides[0])
        self.deconv2 = nn.Conv2d(n_channels[0], n_channels[-1], 
                                 kernel_size[-1], stride=strides[-1], padding=1)
        self.out_transform = nn.Conv2d(n_channels[-1], output_channels, 1, 1)
        self.flatten_shape = flatten_shape
        self.output_size = output_size
    def forward(self, x):
        x = self.latent_transform(x)
        x = x.view(x.shape[0], self.flatten_shape[2], 
                   self.flatten_shape[1], self.flatten_shape[0])
        x = F.interpolate(x, scale_factor=2)
        x = torch.relu(self.deconv1(x))
        x = F.interpolate(x, size=self.output_size)
        x = torch.relu(self.deconv2(x))
        out = torch.sigmoid(self.out_transform(x))
        
        return out
```

<br>

## 2-3. SOM-VAE

```python
class SOMVAE(nn.Module):
    def __init__(self, encoder, decoder_q, decoder_e, latent_dim=64, som_dim=[8,8],
                 input_length=28, input_channels=28, beta=1, gamma=1, alpha=1, tau=1):
        super().__init__()
        self.encoder = encoder
        self.decoder_e = decoder_e
        self.decoder_q = decoder_q
        self.input_length = input_length
        self.input_channels = input_channels
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.som_dim = som_dim
        self.latent_dim = latent_dim
        self.embeddings = nn.Parameter(torch.randn(som_dim[0],som_dim[1],
                                                   latent_dim)*0.05)
        self.mse_loss = nn.MSELoss()
        
        probs_raw = torch.zeros(*(som_dim + som_dim))
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
    def forward(self, x):
        z_e = self.encoder(x)
    
        z_dist = (z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0))**2
        z_dist_sum = torch.sum(z_dist, dim=-1)
        z_dist_flat = z_dist_sum.view(x.shape[0], -1)
        k = torch.argmin(z_dist_flat, dim=-1)
        
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        
        k_stacked = torch.stack([k_1, k_2], dim=1)
        z_q = self._gather_nd(self.embeddings, k_stacked)
        
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_stacked = torch.stack([k_1, k_2], dim=1)
        
        k1_not_top = k_1 < self.som_dim[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.som_dim[1] - 1
        k2_not_left = k_2 > 0
        
        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)
        
        z_q_up = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]
        
        z_q_down = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]
        
        z_q_right = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_right_ =  self._gather_nd(self.embeddings, 
                                      torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] == z_q_right_[k2_not_right == 1]
        
        z_q_left = torch.zeros(x.shape[0], self.latent_dim).cuda()
        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]
        
        z_q_neighbors = torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)
        
        x_q = self.decoder_q(z_q)
        x_e = self.decoder_e(z_e)
        
        return x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat
        
        
    def _loss_reconstruct(self, x, x_e, x_q):
        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q
        return mse_l
    
    def _loss_commit(self, z_e, z_q):
        commit_l = self.mse_loss(z_e, z_q)
        return commit_l
    
    def _loss_som(self, z_e, z_q_neighbors):
        z_e = z_e.detach()
        som_l = torch.mean((z_e.unsqueeze(1) - z_q_neighbors)**2)
        return som_l
    
    def loss_prob(self, k):
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old, k_1, k_2], dim=1)
        
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
        transitions_all = self._gather_nd(self.probs, k_stacked)
        prob_l = -self.gamma*torch.mean(torch.log(transitions_all))
        return prob_l
    
    def _loss_z_prob(self, k, z_dist_flat):
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old], dim=1)
        
        probs_raw = self.probs
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        self.probs = nn.Parameter(probs_pos/probs_sum)
        
        out_probabilities_old = self._gather_nd(self.probs, k_stacked)
        out_probabilities_flat = out_probabilities_old.view(k.shape[0], -1)
        weighted_z_dist_prob = z_dist_flat*out_probabilities_flat
        prob_z_l = torch.mean(weighted_z_dist_prob)
        return prob_z_l
    
    def loss(self, x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat):
        mse_l = self._loss_reconstruct(x, x_e, x_q)
        commit_l = self._loss_commit(z_e, z_q)
        som_l = self._loss_som(z_e, z_q_neighbors)
        prob_l = self.loss_prob(k)
        prob_z_l = self._loss_z_prob(k, z_dist_flat)
        l = mse_l + self.alpha*commit_l + self.beta*som_l + self.gamma*prob_l + 
        	self.tau*prob_z_l
        return l
        
    def _gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs
```



```python
batch_size = 32
latent_dim = 64
som_dim = [8,8]
learning_rate = 0.0005
alpha = 1.0
beta = 0.9
gamma = 1.8
tau = 1.4
decay_factor = 0.9
interactive = True
data_set = "MNIST_data"
save_model = False
time_series = True
mnist = True
```



```python
train = torchvision.datasets.MNIST('./', download=True)
test = torchvision.datasets.MNIST('./', train=False, download=True)
x_train = train.train_data.float()
y_train = train.train_labels 
x_test = test.data.float() 
y_test = test.targets

x_train /= 255
x_test /= 255
```



```python
def generate_batch(data):
    start_img = np.random.randint(0, len(data))
    end_img = np.random.randint(0, len(data))
    interpolation = interpolate_images(data[start_img], data[end_img])
    return interpolation + torch.randn(interpolation.shape)*0.01
```



```python
model = SOMVAE(ConvEncoder(1), ConvDecoder(1), ConvDecoder(1), alpha=alpha, 
               beta=beta, gamma=gamma, tau=tau,som_dim=[8, 8]).cuda()
```



```python
model_param_list = nn.ParameterList()
for p in model.named_parameters():
    if p[0] != 'probs':
        model_param_list.append(p[1])
        
probs_param_list = nn.ParameterList()
for p in model.named_parameters():
    if p[0] == 'probs':
        probs_param_list.append(p[1])
```



```python
opt_model = torch.optim.Adam(model_param_list, lr=learning_rate)
opt_probs = torch.optim.Adam(probs_param_list, lr=learning_rate*100)
```

```python
sc_opt_model = torch.optim.lr_scheduler.StepLR(opt_model, 1000, decay_factor)
sc_opt_probs = torch.optim.lr_scheduler.StepLR(opt_probs, 1000, decay_factor)
```



```python
for e in range(10000):
    opt_model.zero_grad()
    opt_probs.zero_grad()
    batch_x = generate_batch(x_train)
    batch_x = batch_x.cuda()
    x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(batch_x)
    l = model.loss(batch_x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
    l_prob = model.loss_prob(k)
    l.backward()
    opt_model.step()
    l_prob.backward()
    opt_probs.step()
    sc_opt_model.step()
    sc_opt_probs.step()
    if e%10 == 0:
        with torch.no_grad():
            batch_x = generate_batch(x_test)
            batch_x = batch_x.cuda()
            x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(batch_x)
            l = model.loss(batch_x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
            print("Loss: ", l.item())
```



```python
with torch.no_grad():
    batch_x = generate_batch(x_test)
    batch_x = batch_x.cuda()
    x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(batch_x)
```

