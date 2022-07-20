# ResNet & DenseNet & SENet



# [1] ResNet

## 1. ResNet 3줄 요약

- 핵심 : **깊게 쌓자! Go Deeper!**

- 문제 : Gradient Vanishing

  $$\rightarrow$$ skip-connection으로 해결해주자!

- Deeper의 의미

  - (1) larger receptive field
  - (2) more non-linearities

  $$\rightarrow$$ 성능 향상!

<br>

## 2. Residual Blocks

- with Bottlenecks ( 1 x 1 )

![figure2](/assets/img/cv/cv165.png)

<br>

Build up to 152 layers !

<br>

## 3. Different view of Skip Connection

![figure2](/assets/img/cv/cv166.png)

<br>

## 4. 코드 실습

### (1) Import Packages & Dataset & Model

### a) Packages

```python
import torch, torchvision
import torchvision.models as models
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from PIL import Image
```

<br>

### b) Dataset ( CIFAR 10 )

```python
bs = 64

transformation = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
                                 
cifar10 = torchvision.datasets.CIFAR10(root='./', download=True, 
                                       transform = transformation)

dataloader = torch.utils.data.DataLoader(cifar10, batch_size=bs, 
                                         shuffle=True, num_workers=2)
```

<br>

### c) Model

```python
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet152 = models.resnet152(pretrained=True)
```

<br>

# [2] DenseNet

- continuously concatenate previous channel

- pros
  - parameter efficiency
  - computational efficiency
  - Keep low-level features

![figure2](/assets/img/cv/cv167.png)

<br>

```python
densenet121 = models.densenet121(pretrained=True)
densenet161 = models.densenet161(pretrained=True)
densenet169 = models.densenet169(pretrained=True)
densenet201 = models.densenet201(pretrained=True)
```

<br>

# [3] SENet

SE

- [S] Squeeze : capture distributions of **channel-wise response** by GAP
- [E] Excitation : gating channels by **channel-wise attentionw weights** 

![figure2](/assets/img/cv/cv168.png)

<br>

$$\rightarrow$$ can be applied to various DNN architectures

ex) SE-Inception module & SE-ResNet module

![figure2](/assets/img/cv/cv169.png)

