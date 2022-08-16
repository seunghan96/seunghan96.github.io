# DA for MTS forecasting

## 1. Paper List

1. DATSING : Data Augmented Time Series Forecasting with Adversarial Domain Adaptation
   - https://arxiv.org/pdf/2102.06828.pdf

2. Domain Adaptation for TSF via Attention Sharing
   - https://arxiv.org/abs/2102.06828
3. Domain-Adversarial Training of Neural Networks
   - https://arxiv.org/abs/1505.07818
4. A DIRT-T Approach to Unsupervised Domain Adaptation
   - https://arxiv.org/abs/1802.08735
5. Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
   - https://arxiv.org/abs/1712.02560





## 2. Key Points

### (1) DATSING : Data Augmented Time Series Forecasting with Adversarial Domain Adaptation

- data augmentation
  - 아이디어 1) ***국내의 (일부) 데이터를 친환경으로 취급 & 국내(일부)+친환경 으로 모델 학습***
- transfer “domain-INVARIANT” feature representation, from a “pre-trained stacked deep residual network” to “target domains”
  - 아이디어 2) ***(선) “국내”로 모델 학습 $\rightarrow$ (후) “친환경”으로 weight transfer***



### (2) Domain Adaptation for TSF via Attention Sharing

![figure2](https://seunghan96.github.io/assets/img/ts/img203.png).

- 아이디어 3) ***국내 & 친환경을 구분하지 못하게끔 유도하는 discriminator 장치 두기***

  - 해당 representation를 뽑아내는 shared layer는 어떻게 할지는 무궁무진

    ( 여기서는 shared “attention” … Q & K )



### (3) Domain-Adversarial Training of Neural Networks

![figure2](https://seunghan96.github.io/assets/img/da/img1.png).

- 비교적 간단한 구조

- 아이디어 4) ***2개의 loss 두기***

  - **loss 1 : forecasting loss**

  - **loss 2: (음의) domain 구분 loss**

    ( 사실상, (2) DAF와 그 아이디어는 유사하다 볼 수 있음 )



### (4) A DIRT-T Approach to Unsupervised Domain Adaptation

- 데이터들간에는 숨겨진 cluster가 있을 것!

- 그 외의 내용들은 너무 복잡…loss에 뭐 이런거저런거 많이 섞구…

  ![image-20220815174350688](/Users/LSH/Library/Application Support/typora-user-images/image-20220815174350688.png)



### (5) Domain Adaptation with Representation Learning and Nonlinear Relation for Time Series

![image-20220815183130634](/Users/LSH/Library/Application Support/typora-user-images/image-20220815183130634.png)

![image-20220815183147123](/Users/LSH/Library/Application Support/typora-user-images/image-20220815183147123.png)

![image-20220815183159128](/Users/LSH/Library/Application Support/typora-user-images/image-20220815183159128.png)

![image-20220815183206488](/Users/LSH/Library/Application Support/typora-user-images/image-20220815183206488.png)

MMD

- https://seunghan96.github.io/gan/(DGM)12.Difference_of_two_distn/
- MMD 처럼 latent space를 확률적으로 모델링하면 uncertainty 접근도 가능해서 보다 나을 듯?

