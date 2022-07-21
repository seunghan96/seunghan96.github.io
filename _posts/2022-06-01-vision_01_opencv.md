---
title: (CV summary) 01. OpenCV
categories: [CV]
tags: []
excerpt: OpenCV in python
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# OpenCV

```python
import cv2 as cv
import matplotlib.pyplot as plt 
```

<br>

# 1. 기본 문법

## `cv.imread`

```python
img_path = 'image.jpeg'
image = cv.imread(img_path)

print(image.shape)
print(image.max())
print(image.min())
```

```
(1440, 1440, 3)
255
0
```

<br>

```python
plt.imshow(img)
```

<br>

## `cv.cvtColor`

- convert colors
  - `cv.COLOR_BGR2RGB` : Blue,Green,Red $$\rightarrow$$ Red,Green,Blue
  - `cv.COLOR_BGR2GRAY` : Blue,Green,Red $$\rightarrow$$ Gray

```python
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

plt.imshow(image_rgb)
plt.imshow(image_gray, cmap='gray')
```

<br>

# 2. Image Cropping

Cropping : 이미지의 일부를 자르는(Crop) 것

대상 : 2개의 이미지

## (1) Image의 H & W 가져오기

```python
H1, W1 = image1.shape[:2]
H2, W2 = image2.shape[:2]

size = 720
```

<br>

## (2) Image의 중앙을 자르기

```python
def crop_center(image, H, W, size):
  x1 = int(H/2 - size/2)
  x2 = int(H/2 + size/2)
  y1 = int(W/2 - size/2)
  y2 = int(W/2 + size/2)
  return image[x1:x2, y1:y2]
```

<br>

```python
image1_center = crop_center(image1, H1, W1, size)
image2_center = crop_center(image2, H2, W2, size)

plt.imshow(image1_center)
plt.imshow(image2_center)
```

<br>

## (3) Image Masking

### a) Square mask

```python
image1_mask = np.zeros_like(image1)
image2_mask = np.zeros_like(image2)

x1 = int(size/2)
x2 = size

image1_mask[x1:x2,] = image1[x1:x2,] # 위의 절반을 0으로 마스킹
image2_mask[0:x1,] = image2[0:x1,] # 밑의 절반을 0으로 마스킹

plt.imshow(image1_mask)
plt.imshow(image2_mask)
```

<br>

### b) `cv.bitwise_or`

OR 연산 적용

```python
# 위 절반은 이미지2
# 아래 절반은 이미지1
img1_img2_merged = cv.bitwise_or(image1_mask, image2_mask)

plt.imshow(img1_img2_merged)
```

<br>

### c) `cv.bitwise_and` & `cv.circle`

AND 연산 적용 ( 둘 다 값이 0이아니어야만 값이 나옴 )

- 이번엔, **원형 마스크** 적용

```
circle_mask = np.zeros_like(image2)
cv.circle(circle_mask, (size//2,size//2), size//3, (255,255,255), -1 )
plt.imshow(circle_mask)
```

- 가운데에 **흰색 원**
- 테두리는 **검은 배경**

<br>

이것을 Image2에 적용

```python
circle_mask = np.zeros_like(image2)
masked_image2 = cv.bitwise_and(image2, circle_mask)

cv.circle(circle_mask, (size//2,size//2), size//3, (255,255,255), -1 )

#------------------------------------------------------------------------#
plt.imshow(circle_mask)
plt.imshow(masked_image2)
```

<br>

# 3. ORB detector

Oriented FAST and rotated BRIEF

- `kp` : key point
- `des` : descriptor
  - descriptor : 500개의 32차원 벡터

<br>

## (1) Keypoint & Descriptor 찾기

```python
orb = cv.ORB_create()


kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
```

<br>

```python
print(len(kp1),len(des1))
print(len(kp2),len(des2))

print(des1.shape) 
print(des2.shape)
```

```
500 500
500 500
(500, 32)
(500, 32)
```

<br>

## (2) Brute Force Matcher (`cv.BFMatcher`) 로 매칭

```python
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x : x.distance) # sorting
```

<br>

## (3) Visualize Matches

```python
top_K_matches = 10
matched_image = cv.drawMatches(image1, kp1, image2, kp2, matches[:top_K_matches], None, 
                      flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(matched_image)
```

<br>

![figure2](/assets/img/cv/cv133.png)