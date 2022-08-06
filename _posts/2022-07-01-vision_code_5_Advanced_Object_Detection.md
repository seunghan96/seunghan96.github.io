---
title: (CV-project) 05.Advanced Object Detection
categories: [CV]
tags: []
excerpt: One-Stage Detection, YOLO v1, Real-Time Processing, NMS
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# 05.Advanced Object Detection

## 1. One Stage Detection

![figure2](/assets/img/cv/cv289.png)

- ex) YOLO versions

<br>

- needed for **real-time object detection**

![figure2](/assets/img/cv/cv288.png)

<br>

## 2. YOLO v1

![figure2](/assets/img/cv/cv290.png)

<br>

## 3. Albumentations

<br>

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
```

- why not **pytorch transform**??

$$\rightarrow$$ **albumentations** can also convert boxes!

<br>

```python
IMAGE_SIZE = 448

transformer = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
)
```

```python
data_dir = "../DATASET/Detection/"
BATCH_SIZE = 32

trainset = Detection_dataset(data_dir=data_dir, phase="train",
                             transformer=transformer)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=collate_fn)
```

<br>

## 4. Build DataLoader

```python
def build_dataloader(data_dir, batch_size=4):
    transformer = A.Compose([
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    
    dataloaders = {}
    train_dataset = Detection_dataset(data_dir=data_dir, phase="train", transformer=transformer)
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = Detection_dataset(data_dir=data_dir, phase="val", transformer=transformer)
    dataloaders["val"] = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataloaders
```

<br>

## 5. YOLO v1 code

<br>

```python
import torchvision
import torch.nn as nn
```

<br>

Build **custom head**

```python
def build_head(num_bboxes, num_classes, grid_size):
  head = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=(4+1)*num_bboxes+num_classes, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(output_size=(grid_size, grid_size))
        )
```





```python
class YOLOv1_RESNET(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_classes = num_classes
        num_bboxes = 2 # YoLo v1 = 2 boxes!
        grid_size = 7 # 7x7 grid size
        
        # backbone : ResNet18
        resnet18 = torchvision.models.resnet18(pretrained=True)
        layers = [m for m in resnet18.children()]
    
        self.backbone = nn.Sequential(*layers[:-2]) # except for head
        
        self.head = build_head(num_bboxes, num_classes, grid_size)
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out
```

<br>

Prediction result : ( 32 x 12 x 7 x 7 )

- 12 = (5+5)+2
  - 5 : first box
  - 5 : second box
  - 2 : class number

```python
data_dir = "../DATASET/Detection/"
BATCH_SIZE = 32

trainset = Detection_dataset(data_dir=data_dir, phase="train", 
                             transformer=transformer)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, 
                         collate_fn=collate_fn)

model = YOLOv1_RESNET(num_classes=NUM_CLASSES)
```

```python
for index, batch in enumerate(trainloader):
    images = batch[0]
    targets = batch[1]
    filenames = batch[2]
    predictions = model(images)
    break
    
print(predictions.shape)
```

```
torch.Size([32, 12, 7, 7])
```

<br>

## 6. Loss Function of YOLO v1

```python
print('batch size :',len(targets))
print('target shape :',targets[0].shape)
```

```
batch size : 2
target shape : (1, 5)
```

<br>

Shape of **"target"** & **prediction**

- target : [ n * [x_center, y_center, width, height, class_id] ] 
- prediction : (1+4)*num_anchors + num_classes
  - num_anchors = 2
  - num_classes =2

$$\rightarrow$$ need to convert **target** into the form of **prediction** ( = Build Target Grid )

<br>

Example :

- input : **target** 
- convert input into 7 x 7 x (1+4+2) matrix 

- ex)

  - truck : [1, xc, yc, w, h, 1, 0]
  - bus : [1, xc, yc, w, h, 0, 1]

  ( xc, yc, w, h = normalized value ( divided by image shape ) )

- output : **7 x 7 x (1+4+2) matrix**

```python
def build_target_grid(target, num_classes, grid_size):
    # (1+4)*2 + num_classes (X)
    # (1+4) + num_classes (O)
    ## "ground truth(target)" has only 1 box!
    ## ( predicted values have 2 boxes )
    target_grid = torch.zeros((1+4+num_classes, grid_size, grid_size), 
                              device=device)
    for gt in target:
        xc, yc, w, h, cls_id = gt
        x_norm = (xc % (1/grid_size))
        y_norm = (yc % (1/grid_size))
        cls_id = int(cls_id)
        i_grid = int(xc * grid_size)
        j_grid = int(yc * grid_size)
        #------------------------------------------------------------#
        target_grid[0, j_grid, i_grid] = 1
        target_grid[1:5, j_grid, i_grid] = torch.Tensor([x_norm,y_norm,w,h])
        target_grid[5+cls_id, j_grid, i_grid] = 1
        #------------------------------------------------------------#
    return target_grid


def build_batch_target_grid(targets, num_classes, grid_size):
    target_grid_batch = torch.stack([build_target_grid(target, num_classes, grid_size) for target in targets], dim=0)
    return target_grid_batch
```

<br>

```python
num_classes = 2
grid_size = 7

# before : 2 x (1,5)
# after : (2,7,7,7)
groundtruths = build_batch_target_grid(targets)
print(groundtruths.shape)
```

```
torch.Size([2, 7, 7, 7])
```

<br>

```python
groundtruths[0]
# 1st 7x7 : objectness
# 2nd 7x7 : x_center
# 3rd 7x7 : y_center
# 4th 7x7 : W
# 5th 7x7 : H
# 6th 7x7 : truck
# 7th 7x7 : bus
```

```
tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.1077, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.1027, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.5537, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.2500, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
```

<br>

### IoU (Intersection over Union)

to calculate IoU...

(step 1) need to convert the format of coordinate!

- (before) x_norm, y_norm, w, h
- (after) x_min, y_min, x_max, y_max ( unnormalized version )

(2) calculate **intersection & union**

<br>

```python
def generate_xy_normed_grid(grid_size):
    y_offset, x_offset = torch.meshgrid(torch.arange(grid_size), 
                                        torch.arange(grid_size))
    xy_grid = torch.stack([x_offset, y_offset], dim=0)
    xy_normed_grid = xy_grid / grid_size
    return xy_normed_grid
```

```python
# example
generate_xy_normed_grid(4)
```

```
tensor([[[0.0000, 0.2500, 0.5000, 0.7500],
         [0.0000, 0.2500, 0.5000, 0.7500],
         [0.0000, 0.2500, 0.5000, 0.7500],
         [0.0000, 0.2500, 0.5000, 0.7500]],

        [[0.0000, 0.0000, 0.0000, 0.0000],
         [0.2500, 0.2500, 0.2500, 0.2500],
         [0.5000, 0.5000, 0.5000, 0.5000],
         [0.7500, 0.7500, 0.7500, 0.7500]]])
```

<br>

**convert the format of coordinate!**

- (before) x_norm, y_norm, w, h
- (after) x_min, y_min, x_max, y_max ( unnormalized version )

```python
def xywh_to_xyxy(bboxes):
    #------------------------------------------------#
    # bboxes : (B, C, H, W) ... normalized
    num_batch, _, grid_size, grid_size = bboxes.shape
    #------------------------------------------------#
    # generate grid : (B,H,W)
    xy_normed_grid = generate_xy_normed_grid(grid_size=grid_size)
    #------------------------------------------------#    
    # first channel (0) = x_n
    # second channel (1) = y_n
    xcyc = bboxes[:,0:2,...] + xy_normed_grid.tile(num_batch, 1,1,1)
    wh = bboxes[:,2:4,...]
    #------------------------------------------------#
    x1y1 = xcyc - (wh/2)
    x2y2 = xcyc + (wh/2)
    return torch.cat([x1y1,x2y2], dim=1)
```

<br>

**get IoU with 2 boxes**

```python
def get_IoU(cbox1, cbox2):
    box1 = xywh_to_xyxy(cbox1)
    box2 = xywh_to_xyxy(cbox2)
    
    # be careful of max/min!
    x1 = torch.max(box1[:, 0, ...], box2[:, 0, ...])
    y1 = torch.max(box1[:, 1, ...], box2[:, 1, ...])
    x2 = torch.min(box1[:, 2, ...], box2[:, 2, ...])
    y2 = torch.min(box1[:, 3, ...], box2[:, 3, ...])

    intersection = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    union = abs(cbox1[:, 2, ...]*cbox1[:, 3, ...]) + \
            abs(cbox2[:, 2, ...]*cbox2[:, 3, ...]) - intersection

    # .gt : True if >0, False o.w
    intersection[intersection.gt(0)] = intersection[intersection.gt(0)] / union[intersection.gt(0)]
    return intersection
```

<br>

Example )

```python
# 2 pred box
pred_box1 = predictions[:, 1:5, ...]
pred_box2 = predictions[:, 6:10, ...]

# 1 ground truth
gt_box = groundtruths[:, 1:5, ...]

iou1 = get_IoU(pred_box1, gt_box)
iou2 = get_IoU(pred_box2, gt_box)
```

```python
iou1 # IoU of 1 prediction result
```

```
tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.3112, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],
       grad_fn=<IndexPutBackward0>)
```

<br>

**IoU of all the boxes ?**

```python
ious = torch.stack([iou1, iou2], dim=1)
print(ious.shape) # B x (2x7x7) ... 2 : number of pred boxes 
```

```python
torch.Size([2, 2, 7, 7])
```

<br>

among the two boxes of every data in batch...

$$\rightarrow$$ get the max value ( = filtering = choosing the box with highest IoU )

```
max_iou, best_box = ious.max(dim=1, keepdim=True) # (B x (1x7x7))
max_iou = torch.cat([max_iou, max_iou], dim=1) # (B x (2x7x7))
best_box = torch.cat([best_box.eq(0), best_box.eq(1)], dim=1) # (B x (1x7x7))
```

<br>

### Calculate Final Loss

**(1) Predicted Values**

```python
# before : (B x 10(=(4+1)*2) x 7 x 7)
# after : (B x 2 x 5 x 7 x 7)
predictions_ = predictions[:, :5*2, ...].reshape(BATCH_SIZE, 2, 5, grid_size, grid_size)

obj_pred = predictions_[:, :, 0, ...]
xy_pred = predictions_[:, :, 1:3, ...]
wh_pred = predictions_[:, :, 3:5, ...]
cls_pred = predictions[:, 5*2:, ...]
```

<br>

**(2) Ground Truth**

```python
# before : (B x 5(=4+1) x 7 x 7)
# after : (B x 1 x 5 x 7 x 7)
groundtruths_ = groundtruths[:, :5, ...].reshape(BATCH_SIZE, 1, 5, grid_size, grid_size)

obj_target = groundtruths_[:, :, 0, ...]
xy_target = groundtruths_[:, :, 1:3, ...]
wh_target= groundtruths_[:, :, 3:5, ...]
cls_target = groundtruths[:, 5:, ...]
```

<br>

**(3)** 3 kinds of losses

```python
positive = obj_target * best_box
```

```python
# (1) Objectiveness Loss
obj_loss = mse_loss(positive * obj_pred, positive * ious)
noobj_loss = mse_loss((1 - positive) * obj_pred, ious*0)

# (2) Box Regression Loss
xy_loss = mse_loss(positive.unsqueeze(dim=2) * xy_pred, 
                   positive.unsqueeze(dim=2) * xy_target)
wh_loss = mse_loss(positive.unsqueeze(dim=2) * (wh_pred.sign() * (wh_pred.abs() + 1e-8).sqrt()),
                   positive.unsqueeze(dim=2) * (wh_target + 1e-8).sqrt())

# (3) Classification Loss
cls_loss = mse_loss(obj_target * cls_pred, cls_target)
```

<br>

### Summary of YOLO v1 loss

```python
class YOLOv1_LOSS():
    def __init__(self, num_classes, device, lambda_coord=5., lambda_noobj=0.5):
        self.num_classes = num_classes
        self.device = device
        self.grid_size = 7
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(self, predictions, targets):
        self.batch_size, _, _, _ = predictions.shape
        groundtruths = self.build_batch_target_grid(targets)
        groundtruths = groundtruths.to(self.device)
        
        with torch.no_grad():
            iou1 = self.get_IoU(predictions[:, 1:5, ...], groundtruths[:, 1:5, ...])
            iou2 = self.get_IoU(predictions[:, 6:10, ...], groundtruths[:, 1:5, ...])

        ious = torch.stack([iou1, iou2], dim=1)
        max_iou, best_box = ious.max(dim=1, keepdim=True)
        max_iou = torch.cat([max_iou, max_iou], dim=1)
        best_box = torch.cat([best_box.eq(0), best_box.eq(1)], dim=1)

        predictions_ = predictions[:, :5*2, ...].reshape(self.batch_size, 2, 5, self.grid_size, self.grid_size)
        obj_pred = predictions_[:, :, 0, ...]
        xy_pred = predictions_[:, :, 1:3, ...]
        wh_pred = predictions_[:, :, 3:5, ...]
        cls_pred = predictions[:, 5*2:, ...]

        groundtruths_ = groundtruths[:, :5, ...].reshape(self.batch_size, 1, 5, self.grid_size, self.grid_size)
        obj_target = groundtruths_[:, :, 0, ...]
        xy_target = groundtruths_[:, :, 1:3, ...]
        wh_target= groundtruths_[:, :, 3:5, ...]
        cls_target = groundtruths[:, 5:, ...]
        
        positive = obj_target * best_box

        obj_loss = self.mse_loss(positive * obj_pred, positive * ious)
        noobj_loss = self.mse_loss((1 - positive) * obj_pred, ious*0)
        xy_loss = self.mse_loss(positive.unsqueeze(dim=2) * xy_pred, positive.unsqueeze(dim=2) * xy_target)
        wh_loss = self.mse_loss(positive.unsqueeze(dim=2) * (wh_pred.sign() * (wh_pred.abs() + 1e-8).sqrt()),
                           positive.unsqueeze(dim=2) * (wh_target + 1e-8).sqrt())
        cls_loss = self.mse_loss(obj_target * cls_pred, cls_target)
        
        obj_loss /= self.batch_size
        noobj_loss /= self.batch_size
        bbox_loss = (xy_loss+wh_loss) / self.batch_size
        cls_loss /= self.batch_size
        
        total_loss = obj_loss + self.lambda_noobj*noobj_loss + self.lambda_coord*bbox_loss + cls_loss
        return total_loss, (obj_loss.item(), noobj_loss.item(), bbox_loss.item(), cls_loss.item())
    
    def build_target_grid(self, target):
        target_grid = torch.zeros((1+4+self.num_classes, self.grid_size, self.grid_size), device=self.device)

        for gt in target:
            xc, yc, w, h, cls_id = gt
            xn = (xc % (1/self.grid_size))
            yn = (yc % (1/self.grid_size))
            cls_id = int(cls_id)

            i_grid = int(xc * self.grid_size)
            j_grid = int(yc * self.grid_size)
            target_grid[0, j_grid, i_grid] = 1
            target_grid[1:5, j_grid, i_grid] = torch.Tensor([xn,yn,w,h])
            target_grid[5+cls_id, j_grid, i_grid] = 1

        return target_grid
    
    def build_batch_target_grid(self, targets):
        target_grid_batch = torch.stack([self.build_target_grid(target) for target in targets], dim=0)
        return target_grid_batch
    
    def get_IoU(self, cbox1, cbox2):
        box1 = self.xywh_to_xyxy(cbox1)
        box2 = self.xywh_to_xyxy(cbox2)

        x1 = torch.max(box1[:, 0, ...], box2[:, 0, ...])
        y1 = torch.max(box1[:, 1, ...], box2[:, 1, ...])
        x2 = torch.min(box1[:, 2, ...], box2[:, 2, ...])
        y2 = torch.min(box1[:, 3, ...], box2[:, 3, ...])

        intersection = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
        union = abs(cbox1[:, 2, ...]*cbox1[:, 3, ...]) + \
                abs(cbox2[:, 2, ...]*cbox2[:, 3, ...]) - intersection

        intersection[intersection.gt(0)] = intersection[intersection.gt(0)] / union[intersection.gt(0)]
        return intersection
    
    def generate_xy_normed_grid(self):
        y_offset, x_offset = torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size))
        xy_grid = torch.stack([x_offset, y_offset], dim=0)
        xy_normed_grid = xy_grid / self.grid_size
        return xy_normed_grid.to(self.device)

    def xywh_to_xyxy(self, bboxes):
        xy_normed_grid = self.generate_xy_normed_grid()
        xcyc = bboxes[:,0:2,...] + xy_normed_grid.tile(self.batch_size, 1,1,1)
        wh = bboxes[:,2:4,...]
        x1y1 = xcyc - (wh/2)
        x2y2 = xcyc + (wh/2)
        return torch.cat([x1y1, x2y2], dim=1)
```

<br>

## 7. Train YOLO v1

```python
BATCH_SIZE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "../DATASET/Detection/"
train_dset = Detection_dataset(data_dir=data_dir, phase="train", 
                             transformer=transformer)
train_dloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, 
                         collate_fn=collate_fn)
```

<br>

```python
model = YOLOv1_RESNET(num_classes=NUM_CLASSES)
criterion = YOLOv1_LOSS(num_classes=NUM_CLASSES, device=DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=5e-4)
```

<br>

( Simple Training Code )

```python
for index, batch in enumerate(train_dloader):
    images = batch[0].to(DEVICE)
    targets = batch[1]
    filenames = batch[2]
    
    predictions = model(images)
    loss, (obj_loss, noobj_loss, bbox_loss, cls_loss) = criterion(predictions, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if index % 5 == 0:
        print(loss.item(), obj_loss, noobj_loss, bbox_loss, cls_loss)
```

<br>

Training Code Function

```python
def train_one_epoch(dataloaders, model, criterion, optimizer, device):
    train_loss = defaultdict(float)
    val_loss = defaultdict(float)
    
    for phase in ["train", "val"]:
      	# different in (1) Dropout, (2) BatchNorm
        if phase == "train":
            model.train() 
        else:
            model.eval()
        
        running_loss = defaultdict(float)
        for index, batch in enumerate(dataloaders[phase]):
            images = batch[0].to(device)
            targets = batch[1]
            filenames = batch[2]
            
            with torch.set_grad_enabled(phase == "train"):
                predictions = model(images)
            loss, (obj_loss, noobj_loss, bbox_loss, cls_loss) = criterion(predictions, targets)
  
            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss["total_loss"] += loss.item()
                running_loss["obj_loss"] += obj_loss
                running_loss["noobj_loss"] += noobj_loss
                running_loss["bbox_loss"] += bbox_loss
                running_loss["cls_loss"] += cls_loss
                
                train_loss["total_loss"] += loss.item()
                train_loss["obj_loss"] += obj_loss
                train_loss["noobj_loss"] += noobj_loss
                train_loss["bbox_loss"] += bbox_loss
                train_loss["cls_loss"] += cls_loss
                
                if (index > 0) and (index % VERBOSE_FREQ) == 0:
                    text = f"iteration:[{index}/{len(dataloaders[phase])}] - "
                    for k, v in running_loss.items():
                        text += f"{k}: {v/VERBOSE_FREQ:.4f}  "
                        running_loss[k] = 0.
                    print(text)
            else:
                val_loss["total_loss"] += loss.item()
                val_loss["obj_loss"] += obj_loss
                val_loss["noobj_loss"] += noobj_loss
                val_loss["bbox_loss"] += bbox_loss
                val_loss["cls_loss"] += cls_loss

    for k in train_loss.keys():
        train_loss[k] /= len(dataloaders["train"])
        val_loss[k] /= len(dataloaders["val"])
    return train_loss, val_loss
```

<br>

## 8. Final Code

(1) Hyperparameters

```python
NUM_CLASSES = 2
IMAGE_SIZE = 448
BATCH_SIZE = 12
VERBOSE_FREQ = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available and is_cuda else 'cpu')
```

<br>

(2) Dataset / Model / Optimizer / Loss function

```python
data_dir = "../DATASET/Detection/"
dataloaders = build_dataloader(data_dir=data_dir, batch_size=BATCH_SIZE)
model = YOLOv1_RESNET(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
criterion = YOLOv1_LOSS(num_classes=NUM_CLASSES, device=DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
```

<br>

(3) Train

```python
num_epochs = 100

best_epoch = 0
best_score = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss, val_loss = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"epoch:{epoch+1}/{num_epochs} - Train Loss: {train_loss['total_loss']:.4f}, Val Loss: {val_loss['total_loss']:.4f}")
    
    if (epoch+1) % 10 == 0:
        save_model(model.state_dict(), f'model_{epoch+1}.pth')
```

<br>

## 9. Confidence threshold & Non-maximum suppression (NMS)

not at training, only at **inference** stage!

- filter 1) confidence threshold
- filter 2) NMS (Non-maximum suppression)

```python
@torch.no_grad()
def model_predict(image, model, conf_thres=0.3, iou_threshold=0.1):
    predictions = model(image)
    prediction = predictions.detach().cpu().squeeze(dim=0)
    
    grid_size = prediction.shape[-1]
    y_grid, x_grid = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
    stride_size = IMAGE_SIZE/grid_size

    conf = prediction[[0,5], ...].reshape(1, -1)
    xc = (prediction[[1,6], ...] * IMAGE_SIZE + x_grid*stride_size).reshape(1,-1)
    yc = (prediction[[2,7], ...] * IMAGE_SIZE + y_grid*stride_size).reshape(1,-1)
    w = (prediction[[3,8], ...] * IMAGE_SIZE).reshape(1,-1)
    h = (prediction[[4,9], ...] * IMAGE_SIZE).reshape(1,-1)
    cls = torch.max(prediction[10:, ...].reshape(NUM_CLASSES, -1), dim=0).indices.tile(1,2)
    
    x_min = xc - w/2
    y_min = yc - h/2
    x_max = xc + w/2
    y_max = yc + h/2

    prediction_res = torch.cat([x_min, y_min, x_max, y_max, conf, cls], dim=0)
    prediction_res = prediction_res.transpose(0,1)

    prediction_res[:, 2].clip(min=0, max=image.shape[1])
    prediction_res[:, 3].clip(min=0, max=image.shape[0])
    
		# 1~4th : box coordinates
    # 5th : confidence score
    ### (filter 1) score > confidence_threshold
    pred_res = prediction_res[prediction_res[:, 4] > conf_thres]
		### (filter 2) iou > iou_threshold
    nms_index = torchvision.ops.nms(boxes=pred_res[:, 0:4], scores=pred_res[:, 4], iou_threshold=iou_threshold)
    pred_res_ = pred_res[nms_index].numpy()
    
    n_obj = pred_res_.shape[0]
    bboxes = np.zeros(shape=(n_obj, 4), dtype=np.float32)
    bboxes[:, 0:2] = (pred_res_[:, 0:2] + pred_res_[:, 2:4]) / 2
    bboxes[:, 2:4] = pred_res_[:, 2:4] - pred_res_[:, 0:2]
    scores = pred_res_[:, 4]
    class_ids = pred_res_[:, 5]
    
    return bboxes, scores, class_ids
```

