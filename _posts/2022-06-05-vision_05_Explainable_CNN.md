# Explainable CNN



# 1.소개

Explainable CNN을 위해, layer를 통과해서 나온 **feature map들을 살펴보자** !!

( **layer-wise network visualization** )

참고 자료 :

- https://github.com/ashutosh1919/explainable-cnn
- https://github.com/ashutosh1919/explainable-cnn/blob/main/examples/explainable_cnn_usage.ipynb

<br>

# 2. Import Packages & Datasets

우선, 패키지를 설치한다.

```bash
! pip install explainable-cnn
```

<br>

관련 패키지들을 모두 불러온다

```python
from explainable_cnn import CNNExplainer
import pickle
import torch
from torchvision import models
import matplotlib.pyplot as plt

from PIL import Image
```

<br>

ImageNet데이터셋은, 1000개의 클래스로 이루어진 이미지 데이터셋이다.

각 label에 해당하는 class가 무엇인지를 나타내는 파일을 불러오자.

```python
CLASS_LABEL_DIR = "./data/imagenet_class_labels.pkl"
with open(CLASS_LABEL_DIR, "rb") as label_file:
    imagenet_class_labels = pickle.load(label_file)
```

<br>

Example ) 5번 레이블에 해당하는 클래스는?

```python
imagenet_class_labels[5]
```

```
'electric ray, crampfish, numbfish, torpedo'
```

<br>

# 3. Model & CNN Explainer 불러오기

- **Model** : VGG16

- **CNN Explainer** : layer 이후의 feature map을 시각화해주는 instance

```python
model = models.vgg16(pretrained=True)
cnn_explainer = CNNExplainer(model, imagenet_class_labels)
```

<br>

# 4. Visualization

Sample 이미지 불러오기

- ex) Tiger Shark 이미지 
  ( 해당 클래스의 이미지 label = 3 )

```python
IMG_PATH = "./data/tiger_shark.jpeg"
image = Image.open(IMG_PATH).convert('RGB')
```

![figure2](/assets/img/cv/cv163.png)

<br>

## (1) Salieny Map

어느 부분이 이미지 내에서 **핵심**인지를 나타내는 부분

( Saliency Map은 **layer-wise하게 시각화하지 않고, model 단위로 시각화** 한다. )

- 메소드 : `.get_saliency_map()` 

```python
saliency_map = cnn_explainer.get_saliency_map(
    IMG_PATH, 3, (224, 224) ) # 3의 의미 : class의 레이블
```

<br>

## (2) Guided Backpropagation

( Saliency Map과 마찬가지로 **layer-wise하게 시각화하지 않고, model 단위로 시각화** 한다. )

- 메소드 : `.get_guided_back_propagation()` 

```python
guided_backprop = cnn_explainer.get_guided_back_propagation(
    IMG_PATH, 3, (224, 224) 
    )
```

<br>

## (3) GradCAM

- 메소드 : `.get_grad_cam()` 

```python
grad_cam = cnn_explainer.get_grad_cam(
    IMG_PATH, 3, (224, 224), ["features"] 
    )
```

앞선 saliency map & guided backpropagation 과의 차이점

-  **layer-wise하게 시각화 가능!!**

- 따라서, 위의 `[“features”]`와 같이 명시적으로 layer명을 지정해줘야 한다

  ( + 해당 layer내에서 몇 번째 인지 … `grad_cam[0]` )

<br>

## (4) Guided GradCAM

- 메소드 : `.get_guided_grad_cam()` 

**Gradient-weighted** class activated activation

( Guided Backprop + Grad CAM )

```python
guided_grad_cam = cnn_explainer.get_guided_grad_cam(
    IMG_PATH, 3, (224, 224), ["features"] 
    )
```

<br>

## (5) 전부 시각화

```python
plt.imshow(saliency_map, cmap="hot")
plt.imshow(guided_backprop.astype('uint8'))
plt.imshow(grad_cam[0].astype('uint8'))
plt.imshow(guided_grad_cam[0].astype('uint8'))
```

![figure2](/assets/img/cv/cv164.png)
