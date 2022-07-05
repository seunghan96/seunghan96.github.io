# ResNet

# 1. ResNet 3줄 요약

- 핵심 : **깊게 쌓자! Go Deeper!**

- 문제 : Gradient Vanishing

  $\rightarrow$ skip-connection으로 해결해주자!

- Deeper의 의미

  - (1) larger receptive field
  - (2) more non-linearities

  $\rightarrow$ 성능 향상!

<br>

# 2. 코드 실습

## (1) Import Packages & Dataset & Model

### a) Packages

```
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

```
resnet18 = models.resnet18(pretrained=True)
```

<br>

## (2) d



