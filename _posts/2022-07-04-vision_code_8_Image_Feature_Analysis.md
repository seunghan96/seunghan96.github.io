---
title: (CV-project) 08.Image Feature Analysis
categories: [CV]
tags: []
excerpt: ORL face dataset, Feature Visualization, Tensorboard
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# 08.Image Feature Analysis

## 1. Load Model

```python
transformer = build_transformer()
dataset = Face_Dataset(data_dir, transformer=transformer)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
```

```python
ckpt_path = "./trained_model/model_76.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(ckpt_path, DEVICE)
```

<br>

## 2. Visualizing Latent Vector with Tensorboard

```python
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/face_features')
```

<br>

```python
print(os.listdir())
print(os.listdir('runs'))
print(os.listdir('runs/face_features'))
```

```
['trained_model', 'SOLUTION.ipynb', 'runs']
['face_features']
['events.out.tfevents.1659588627.iseunghan-ui-MacBookPro.local']
```

- all the results/objects will be saved here!

<br>

How to **write data/results to TensorBoard**?

```python
all_images = []
all_labels = []
all_embeds = []

for idx, sample in enumerate(dataloader):
    image = sample[0]
    label = sample[1]

    with torch.no_grad():
        embed = model(image.to(DEVICE))
    embed = embed.detach().cpu().numpy()
    
    image = make_grid(image, normalize=True).permute(1,2,0)
    image = cv2.resize(np.array(image), dsize=(80, 80), interpolation=cv2.INTER_NEAREST)
    
    all_images.append(image)
    all_labels.append(label)
    all_embeds.append(embed)

all_images = torch.Tensor(np.moveaxis(np.stack(all_images, axis=0), 3, 1))
all_embeds = torch.Tensor(np.stack(all_embeds, axis=0).squeeze(1))
all_labels = np.concatenate(all_labels).tolist()
```

```python
writer.add_embedding(all_embeds, 
                     label_img=all_images, 
                     metadata=all_labels)
writer.close()
```

<br>

open terminal...

```bash
$ tensorboard.exe --logdir runs/
```

- connect to localhost

![figure2](/assets/img/cv/cv292.png)

<br>

