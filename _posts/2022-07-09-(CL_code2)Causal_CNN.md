---
title: (code) Causal CNN
categories: [CL, TS]
tags: []
excerpt:
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal CNN

```
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Advances in neural information processing systems 32 (2019).
```

<br>

references :

- https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
- https://arxiv.org/pdf/1901.10738.pdf

<br>

```python
import torch
```

<br>

# 1. Chomp1D

```python
class Chomp1d(torch.nn.Module):
    """
    Removes the 's' last elements of a time series
    -- Input : (B,C,L)
    -- Output : (B,C,L-s)
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]
```

<br>

```python
chomp = Chomp1d(chomp_size = 30)

B = 64
C = 32
L = 100
long_ts = torch.rand((B,C,L))
short_ts = chomp(long_ts)
print(long_ts.shape)
print(short_ts.shape)
```

```
torch.Size([64, 32, 100])
torch.Size([64, 32, 70])
```

<br>

# 2. Squeeze 

```python
class SqueezeChannels(torch.nn.Module):
    def __init__(self, squeeze_dim = 2):
        super(SqueezeChannels, self).__init__()
        self.squeeze_dim = squeeze_dim

    def forward(self, x):
        return x.squeeze(self.squeeze_dim)
```

<br>

```python
squeeze = SqueezeChannels(squeeze_dim = 2)

before_squeeze = torch.rand(10,20,1,2)
after_squeeze = squeeze(before_squeeze)
print(before_squeeze.shape)
print(after_squeeze.shape)
```

```
torch.Size([10, 20, 1, 2])
torch.Size([10, 20, 2])
```

<br>

```python
squeeze = SqueezeChannels(squeeze_dim = 3)
before_squeeze = torch.rand(10,20,2,1)
after_squeeze = squeeze(before_squeeze)
print(before_squeeze.shape)
print(after_squeeze.shape)
```

```
torch.Size([10, 20, 2, 1])
torch.Size([10, 20, 2])
```

<br>

# 3. CauCNN block

```python
class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block
    = 2 causal convolutions (with leaky ReLU) + parallel residual connection.
    -- Input : (B, C_in, L)
    -- Output : (B, C_out, L)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Left Padding ( = for "causality" )
        padding = (kernel_size - 1) * dilation

        #================================================================#
        # [ 1st causal convolution ]
        conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding, dilation = dilation)
        conv1 = torch.nn.utils.weight_norm(conv1)
        chomp1 = Chomp1d(padding) # Chomp ( no cheating )
        relu1 = torch.nn.LeakyReLU()
        #================================================================#

        #================================================================#
        # [ 2nd causal convolution ]
        conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=padding, dilation = dilation)
        conv2 = torch.nn.utils.weight_norm(conv2)
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
        #================================================================#

        #================================================================#
        # [ Causal network ]
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, 
                                          conv2, chomp2, relu2)

        # --- Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # --- Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)
```

<br>

```python
cau_cnn_block = CausalConvolutionBlock(in_channels=8, out_channels=32, 
                                 kernel_size=3, dilation=2, final=False)

B = 64
C_in = 8
L = 100

input = torch.randn((B, C_in, L))
output = cau_cnn_block(input)
print(input.shape)
print(output.shape)
```

```
torch.Size([64, 8, 100])
torch.Size([64, 32, 100])
```

<br>

# 4. CauCNN

```python
class CausalCNN(torch.nn.Module):
    """
    Causal CNN
    = sequence of causal convolution blocks.
    -- Input : (B, C_in, L)
    -- Output : (B, C_out, L)
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

<br>

```python
cau_cnn = CausalCNN(in_channels=8, mid_channels=32, depth=4,
                    out_channels=128, kernel_size=3)

B = 64
C_in = 8
L = 100

input = torch.randn((B, C_in, L))
output = cau_cnn(input)
print(input.shape)
print(output.shape)
```

```
torch.Size([64, 8, 100])
torch.Size([64, 128, 100])
```

<br>

# 5. CauCNN Encoder

```python
class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a TS using a causal CNN
    - (1) causal_cnn 
    ----- (B, C_in, L) -> (B,C_out,L)
    - (2) adaptive max pooling ( makes TS to fixed size )
    ----- (B, C_out, L) -> (B,C_out, 1)
    - (3) squeeze
    ----- (B,C_out, 1) -> (B,C_out)
    """
    def __init__(self, in_channels, mid_channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        
        causal_cnn = CausalCNN(in_channels, mid_channels, depth, reduced_size, kernel_size)
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels(squeeze_dim = 2) # Time dimension  
        linear = torch.nn.Linear(reduced_size, out_channels)
        
        self.network = torch.nn.Sequential(causal_cnn, reduce_size, squeeze, linear)

    def forward(self, x):
        return self.network(x)
```

<br>

```python
cau_cnn_encoder = CausalCNNEncoder(in_channels=8, mid_channels=32, depth=4,
                                   reduced_size = 128, out_channels=20, kernel_size=3)

B = 64
C_in = 8
L = 100

input = torch.randn((B, C_in, L))
output = cau_cnn(input)
print(input.shape)
print(output.shape)
```

```
torch.Size([64, 8, 100])
torch.Size([64, 20])
```

<br>

# + $$\alpha$$ ) LSTM

```python
class LSTMEncoder(torch.nn.Module):
    """
    Encoder of a TS using a LSTM ( 1D TS : C_in = 1 )
    - Input : (B, C_in, L)
    ----------(B, hidden_size, L) 
    - Output : (B, )
    """
    def __init__(self,input_dim=1, hidden_dim=256, output_dim=160 ):
        super(LSTMEncoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=2)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        print(x.shape)
        print(x.permute(2, 0, 1).shape)
        print(self.lstm(x.permute(2, 0, 1))[0].shape)
        print(self.lstm(x.permute(2, 0, 1))[0][-1].shape)
        print(self.linear(self.lstm(x.permute(2, 0, 1))[0][-1]).shape)
        
        return self.linear(self.lstm(x.permute(2, 0, 1))[0][-1])	
```

<br>

```python
lstm_enc = LSTMEncoder(input_dim=1, hidden_dim=256, output_dim=160)

B = 64
C_in = 1
L = 100

input = torch.randn((B, C_in, L))
output = lstm_enc(input)
```

```
torch.Size([64, 1, 100])
torch.Size([100, 64, 1])
torch.Size([100, 64, 256])
torch.Size([64, 256])
torch.Size([64, 160])
```



