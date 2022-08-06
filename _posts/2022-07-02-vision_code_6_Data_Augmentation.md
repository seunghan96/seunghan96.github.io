---
title: (CV-project) 06.Data Augmentation
categories: [CV]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# 06.Data Augmentation

```
A.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=(127,127,127), p=0.5)

A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE)

A.HorizontalFlip(p=0.5)

A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)

A.RandomBrightnessContrast(p=0.3)
```

<br>

## 1. Data Augmentation

```
import albumentations as A
from albumentations.pytorch import ToTensorV2
```

<br>

Put the stochastic augmentations in **transformer**

- only apply it to **Training dataset**

```python
IMAGE_SIZE = 448

transformer = A.Compose([
        A.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=(127,127,127), p=0.5),
        A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
)
```

<br>

Build **dataset & dataloader** with the transformer above

```python
data_dir = "../DATASET/Detection/"
BATCH_SIZE = 4

trainset = Detection_dataset(data_dir=data_dir, phase="train", transformer=transformer)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
```

<br>

Do it for both 

- (1) training data
- (2) validation data

$$\rightarrow$$ ***no augmentation in validationd dataset***

```python
def build_dataloader(data_dir, batch_size=4, image_size=448):
    # Augmentation (O)
    train_transformer = A.Compose([
        A.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=(127,127,127), p=0.5),
        A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    
    # Augmentation (X)
    val_transformer = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    
    dataloaders = {}
    train_dataset = Detection_dataset(data_dir=data_dir, phase="train", transformer=train_transformer)
    val_dataset = Detection_dataset(data_dir=data_dir, phase="val", transformer=val_transformer)
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloaders["val"] = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataloaders
```



![figure2](/assets/img/cv/cv252.png)

