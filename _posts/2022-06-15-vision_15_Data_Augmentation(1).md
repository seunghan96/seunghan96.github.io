---
title: (CV summary) 15. Data Augmentation (1) - Rule based
categories: [CV]
tags: []
excerpt: Rule based Data Augmentation

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# Data Augmentation (1)

Categories of Data Augmentation

- (1) **Rule-based**
- (2) **GAN-based**
- (3) **AutoML-based**

<br>

## (1) Rule-based

(1-1) Color Transformation

- Gaussian Blur, Motion Blur
- Brightness Jitter, Contrast Jitter, Saturation Jitter
- ISO Noise
- JPEG compression

<br>

(1-2) Spatial Transformation

- Flip, Rotation, Crop, Affine …

<br>

![figure2](/assets/img/cv/cv231.png)

<br>

(1-3) ETC

- **a) Data Mixing / Erasing**
- **b) PatchSuffle Regulaization**
- **c) Sample Pairing**
- **d) Mixup**
- **e) Mosaic Augmentation** ( Cropping and Patching )
- **f) Multiple Way of Mixing**
- **g) Manifold Mixup**
- **h) Random Erasing / Cutout**
- **i) Hide-And-Seek for Weakly-Supervised Localization**
- **j) CutMix**
- **k) AugMix**
- **l) SmoothMix**

<br>

**a) Data Mixing / Erasing**

- train by **N-label prediction** task

<br>

**b) PatchSuffle Regulaization**

- shuffle pixels inside window
- cons) need hyperparameter $$N$$ ( = window size )

![figure2](/assets/img/cv/cv232.png)

<br>

**c) Sample Pairing**

- mix 2 images pixel-wise
- training : by $$N$$-class multi-label prediction

![figure2](/assets/img/cv/cv233.png)

<br>

**d) Mixup**

- mix 2 images, with **linear interpolation**

![figure2](/assets/img/cv/cv234.png)

![figure2](/assets/img/cv/cv235.png)

<br>

**e) Mosaic Augmentation** ( Cropping and Patching )

- gather multiple patches from multiple images

- solve **N-label classification** task

![figure2](/assets/img/cv/cv236.png)

<br>

**f) Multiple Way of Mixing**

- mix 2 images pixel-wise
- propose **8-ways of mixing**

![figure2](/assets/img/cv/cv237.png)

<br>

**g) Manifold Mixup**

- Mix up in the **hidden representation** space!
- more natural/smooth boundary

![figure2](/assets/img/cv/cv238.png)

<br>

**h) Random Erasing / Cutout**

- erase randomly ( black / white / random … )

- not only “random erasing + random cropping”,

  but also “image/object-aware erasing”

- used in **object detection** & **person re-identification**

![figure2](/assets/img/cv/cv239.png)

<br>

**i) Hide-And-Seek for Weakly-Supervised Localization**

- gather multiple patches from multiple images

- task : **object localization**

  ( = finding bounding box )

![figure2](/assets/img/cv/cv240.png)

<br>

**j) CutMix**

- cropping + patching
- task : classification, weakly-supervised object localization

![figure2](/assets/img/cv/cv241.png)

<br>

**k) AugMix**

- mix augmentations!
- with different weight per augmentation methods!

![figure2](/assets/img/cv/cv242.png)

<br>

**l) SmoothMix**

- similar to **CutMix**, but when patching, **smooth the boundary!**

![figure2](/assets/img/cv/cv243.png)