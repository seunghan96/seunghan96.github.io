---
title: (CV summary) 19. Instance Segmentation - Mask RCNN
categories: [CV]
tags: []
excerpt: Mask RCNN
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# Instance Segmentation - Mask RCNN

<br>

## 1. Mask RCNN

(Mask R-CNN, He et al., ICCV 2017)

<br>

RCNN family

- RCNN, Fast RCNN, Faster RCNN : **Object Detection**
- Mask RCNN : **Instance Segmentation**

![figure2](/assets/img/cv/cv301.png)

<br>

Summary

- Extend Faster RCNN to **pixel-level instage segmentation**
- **RoI Align**

![figure2](/assets/img/cv/cv302.png)

<br>

$$\mathcal{L}=\mathcal{L}_{\text {cls }}+\mathcal{L}_{\text {box }}+\mathcal{L}_{\text {mask }}$$.

- $$\mathcal{L}_{\text {cls }}$$ & $$\mathcal{L}_{\text {box }}$$ : same as Faster RCNN
- $$\mathcal{L}_{\text {mask }}=-\frac{1}{m^{2}} \sum_{1 \leq i, j \leq m}\left[y_{i j} \log \hat{y}_{i j}^{k}+\left(1-y_{i j}\right) \log \left(1-\hat{y}_{i j}^{k}\right)\right]$$.

<br>

## 2. Summary of RCNN family

( https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/ )

![figure2](/assets/img/cv/cv303.png)