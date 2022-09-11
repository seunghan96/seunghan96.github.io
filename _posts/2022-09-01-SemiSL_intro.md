# Semi-supervised Learning

# 1. Limitation of Supervised Learning

labeled data를 확보하기 어려울 수

labeled data가 적은 경우, 진짜 데이터의 분포 전체를 커버하지 못할 수도 있음

<br>

# 2. What is Semi-supervised Learning?

적은 labeled data +  많은 unlabeled data

label을 맞추는 모델에서 벗어나 데이터 자체의 본질적인 특성이 모델링 된다면 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올릴 수 있다는 것

Semi-supervised learning의 목적함수는 supervised loss Ls 와 unsupervised loss Lu 의 합을 최소화하는 것으로 표현

supervised + unsupervised task가 1 stage로 이루어짐

$L = L_s + L_u$

- $L_s$ : (continuous value) regression loss / (discrete value) classificaiton loss

<br>

# 3. Assumption of Semi-supervised Learning

## (1) Smoothness Assumption

**"만약 데이터 포인트 x1 과 x2가 고밀도 지역에서 가까이 위치하다면, 해당하는 출력 y1 과 y2 도 가까워야한다."**

- 이는 같은 class에 속하고 같은 cluster인 두 입력이 입력공간 상에서 고밀도 지역에 위치하고 있다면, 해당하는 출력도 가까워야한다는 것을 의미한다. 

- 반대도 역시 성립하는데, 만약 두 데이터포인트가 저밀도지역에서 멀리 위치한다면, 해당하는 출력도 역시 멀어야 한다. 

- 이러한 가정은 classification엔 도움이 되는 가정이지만 regression에선 별로 도움이 안된다.



## (2) Cluster assumption

**"만약에 데이터 포인트들이 같은 cluster에 있다면, 그들은 같은 class일 것이다."**

하나의 cluster는 하나의 class를 나타낼 것

decision boundary는 저밀도지역을 통과해야만 한다



## (3) Manifold Assumption

**"고차원의 데이터를 저차원 manifold로 보낼 수 있다."**

만약 data를 더 낮은 차원으로 보낼 수 있다면 우리는 unlabeled data를 사용해서 저차원 표현을 얻을 수 있고 labeled data를 사용해 더 간단한 task를 풀 수 있다.

<br>

# 4. Algorithms

 대부분 이미지 데이터(CIFA-100, ImageNet 등)의 classification task

