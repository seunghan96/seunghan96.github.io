---
title: (CV-project) 07.Few Shot Learning
categories: [CV]
tags: []
excerpt: ORL face dataset, Siamese Network, Contrastive Loss Function, Face Recognition
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# 07.Few Shot Learning

## 1. Import Packages  & Dataset

```python
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
```

<br>

ORL face dataset

```python
data_dir = "../DATASET/Face-Recognition/"
phase = "train"

person_items = []
for (root, dirs, files) in os.walk(os.path.join(data_dir, phase)):
    if len(files) > 0:
        for file_name in files:
            person_items.append(os.path.join(root, file_name))
            
person_items
```

```
['../DATASET/Face-Recognition/train\\s1\\1.png',
 '../DATASET/Face-Recognition/train\\s1\\10.png',
 '../DATASET/Face-Recognition/train\\s1\\2.png',
 '../DATASET/Face-Recognition/train\\s1\\3.png',
 '../DATASET/Face-Recognition/train\\s1\\4.png',
 '../DATASET/Face-Recognition/train\\s1\\5.png',
 '../DATASET/Face-Recognition/train\\s1\\6.png',
 '../DATASET/Face-Recognition/train\\s1\\7.png',
 '../DATASET/Face-Recognition/train\\s1\\8.png',
 '../DATASET/Face-Recognition/train\\s1\\9.png',
 '../DATASET/Face-Recognition/train\\s10\\1.png',
```

<br>

File Name

` '../DATASET/Face-Recognition/train\\s2\\8.png'`

- `s1` : person # 2
- `5.png` : 5th picture

```python
print(faceA_path)
print(Path(faceA_path).parent)
print(Path(faceA_path).parent.name)
```

```
'../DATASET/Face-Recognition/train/s2/8.png'
PosixPath('../DATASET/Face-Recognition/train/s2')
's2'
```

<br>

## 2. Dataset Class

```python
class Face_Dataset():
    def __init__(self, data_dir, phase, transformer=None):
        self.person_items = []
        for (root, dirs, files) in os.walk(os.path.join(data_dir, phase)):
            if len(files) > 0:
                for file_name in files:
                    self.person_items.append(os.path.join(root, file_name))

        self.transformer = transformer
        
    def __len__(self):
        return len(self.person_items)
    
    def __getitem__(self, index, ):
        faceA_path = self.person_items[index]
        person = Path(faceA_path).parent.name
        
        # 0 : different ( = negative )
        # 1 : same ( = positive )
        same_person = np.random.randint(2)
        
        # POSITIVE
        if same_person:
            same_person_dir = Path(faceA_path).parent
            same_person_fn = [fn for fn in os.listdir(same_person_dir) if fn.endswith("png")]
            faceB_path = os.path.join(same_person_dir, np.random.choice(same_person_fn))
        
        # NEGATIVE
        else:
            while True:
                faceB_path = np.random.choice(self.person_items)
                if person != Path(faceB_path).parent.name:
                    break
                    
        faceA_image = cv2.imread(faceA_path, 0)
        faceB_image = cv2.imread(faceB_path, 0)
        
        if self.transformer:
            faceA_image = self.transformer(faceA_image)
            faceB_image = self.transformer(faceB_image)
            
        return faceA_image, faceB_image, np.array([1-same_person])
```

<br>

## 3. Transformer for Data Augmentation

```python
def build_transformer(image_size=100):
    transformers = {}
    
    transformers["train"] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.9,1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    
    transformers["val"] = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    return transformers
```

<br>

## 4. Build Dataset & DataLoader

```python
def build_dataloader(data_dir, batch_size=64):
    dataloaders = {}
    
    transformers = build_transformer()
    tr_dataset = Face_Dataset(data_dir, phase="train", transformer=transformers["train"])
    dataloaders["train"] = DataLoader(tr_dataset, shuffle=True, batch_size=batch_size)
    
    val_dataset = Face_Dataset(data_dir, phase="val", transformer=transformers["val"])
    dataloaders["val"] = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    return dataloaders
```

<br>

```python
data_dir = "../DATASET/Face-Recognition/"
BATCH_SIZE = 64
dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
```

<br>

## 5. Siamese Network

![figure2](/assets/img/cv/cv291.png)

( https://towardsdatascience.com/what-are-siamese-neural-networks-in-deep-learning-bb092f749dcb )

<br>

building block ( Conv - ReLU - BN )

```python
def convBlock(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channel),
    )
```

<br>

**Siamese Network**

```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            convBlock(1,4),
            convBlock(4,8),
            convBlock(8,8),
            nn.Flatten(),
            nn.Linear(8*100*100, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x1, x2):
        out1 = self.features(x1)
        out2 = self.features(x2)
        return out1, out2
```

<br>

**example**

```python
IMAGE_SIZE = 100
rand_img1 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
rand_img2 = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)

model = SiameseNetwork()

z1, z2 = model(rand_img1, rand_img2)
print(z1.shape)
print(z2.shape)
```

```
torch.Size([1, 10])
torch.Size([1, 10])
```

<br>

## 6. Contrastive Loss

- https://seunghan96.github.io/cl/cv/ppt/SimCLR,MoCo/

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    
    def forward(self, z1, z2, label):
        dist = F.pairwise_distance(z1, z2, keepdim=True)
        loss = torch.mean((1-label) * torch.pow(dist, 2) + label * 
                          torch.pow(torch.clamp((self.margin - dist), min=0), 2))
        acc = ((dist > 0.6) == label).float().mean()
        return loss, acc
```

<br>

## 7. Face Recognition Task

```python
def train_one_epoch(dataloaders, model, criterion, optimizer, device):
    losses = {}
    accuracies = {}
    
    for phase in ["train", "val"]:
        running_loss = 0.0
        running_acc = 0
        
        if phase == "train":
            model.train()
        else:
            model.eval()
            
        for index, batch in enumerate(dataloaders[phase]):
            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
            label = batch[2].to(device)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                z1, z2 = model(x1, x2)
            loss, acc = criterion(z1, z2, label)
                
            if phase == "train":
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            running_acc += acc.item()
                    
        losses[phase] = running_loss/len(dataloaders[phase])
        accuracies[phase] = running_acc/len(dataloaders[phase])
    return losses, accuracies
```

<br>

```python
model = SiameseNetwork()
model = model.to(DEVICE)
criterion = ContrastiveLoss(margin=2.0)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
```

