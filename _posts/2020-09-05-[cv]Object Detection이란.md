---
title: Object Detection이란
categories: [DL,CV]
tags: [Deep Learning, CV]
excerpt: Object Detection,1-stage Detector,2-stage Detector
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Object Detection이란?  ]

# 1. Object Detection

- **"여러"** 물체의 **"(1) class"** & 물체의 **"(2) 위치"**를 둘 다 파악하기!

![figure2](/assets/img/cv/cv51.png)

<br>

즉, Object Detection은 결국 (1) + (2) 이다

- (1) Multi-label **CLASSIFICATION**
- (2) Bounding Box Regression ( = **LOCALIZATION** )

![figure2](/assets/img/cv/cv52.png)

<br>

# 2. 두 종류의 Object Detection

## a) 1-stage Detector

- Regional Proposal & Classification이 **"한번에"** 이루어짐

  ( end-to-end처럼, **bounding box regression & classification을 한번에** )

- ex) **YOLO, SSD**

![figure2](/assets/img/cv/cv53.png)

<br>

### b) 2-stage Detector

- Regional Proposal & Classification이 **"순차적으로"** 이루어짐
  - **step 1)** proposal region이라고 하는 **bounding box**를 만들어낸 다음
  - **step 2)** **classifcation**을 수행함
- ex) **R-CNN 계열**

![figure2](/assets/img/cv/cv54.png)

<br>

# 3. History

![figure2](/assets/img/cv/cv55.png)

<br>

# 4. Object Detection의 사용 분야

- 자율주행 자동차
- OCR
- Aerial image 분석
- CCTV 감시
- 스포츠 경기 분석
- 무인 점포
- 불량 제품 검출

<br>

