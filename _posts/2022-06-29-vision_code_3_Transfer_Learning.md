---
title: (CV-project) 03.Transfer Learning
categories: [CV]
tags: []
excerpt: VGG16, U-Net, Covid chest-xray
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# 03.Transfer Learning

![figure2](/assets/img/cv/cv283.png)

Much less dataset!! how to solve?

$\rightarrow$ ***Transfer Learning!***

<br>

## 1. Backbone of U-Net Encoder

backbone : **VGG16**

How does the **architeture of backbone (VGG16)** look like?

```python
backbone = models.vgg16_bn(pretrained=False).features
backbone
```

```
Output exceeds the size limit. Open the full output data in a text editor
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (5): ReLU(inplace=True)
  (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (9): ReLU(inplace=True)
  (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

...
  (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (42): ReLU(inplace=True)
  (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

<br>

### (1) Encoder of U-Net

```python
class Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        backbone = models.vgg16_bn(pretrained=pretrained).features
        self.conv_block1 = nn.Sequential(*backbone[:6]) # use pre-trained weight
        self.conv_block2 = nn.Sequential(*backbone[6:13])
        self.conv_block3 = nn.Sequential(*backbone[13:20])
        self.conv_block4 = nn.Sequential(*backbone[20:27])
        self.conv_block5 = nn.Sequential(*backbone[27:34], 
                                         ConvLayer(512, 1024, kernel_size=1, padding=0))

    def forward(self, x):
        encode_features = [] # for connection to the DECODER part
        
        out = self.conv_block1(x)
        encode_features.append(out) # add
        
        out = self.conv_block2(out)
        encode_features.append(out) # add
        
        out = self.conv_block3(out)
        encode_features.append(out) # add
        
        out = self.conv_block4(out)
        encode_features.append(out) # add
        
        out = self.conv_block5(out)
        return out, encode_features
```

<br>

## 2. Weight Initialization ( vs Transfer Learning ) 

How to initialize all the parameters in model at once?

( with **He initialization** )

```python
def He_initialization(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight) 
    elif isinstance(module, torch.nn.BatchNorm2d):
        module.weight.data.fill_(1.0)
```

```python
model = UNet(num_classes=NUM_CLASSES, pretrained=False)
#--------------------------------------------------------#
model.apply(weight_He_initialization)
#--------------------------------------------------------#
```

<br>

## 3. Weight transfer with freezing encoder layer

```python
model = UNet(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)
#--------------------------------------------------------#
model.encoder.requires_grad_ = False
#--------------------------------------------------------#
```

<br>

## 4. Fine Tuning

![figure2](/assets/img/cv/cv286.png)