---
title: \[continual\] (paper 1) Progressive Neural Network
categories: [CONT,STUDY]
tags: [Continual Learning]
excerpt: Progressive Neural Network에 관한 소개글
---

# Progressive Neural Network

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Contents

1. Introduction
2. Transfer Learning
3. Catastrophic Forgetting
4. Progressive Neural Network (Prognets)

<br>

# 1. Introduction

새로운 것을 배우는 것은, **“기존에 알던 것을 기반”**으로 하는 경우가 많음

ex) 4발 자전거 배운 뒤, 3발 자전거

 <br>

**Transfer Learning** : 하나의 task의 지식을 다른 task를 풀기 위해 전이(Transfer)시키는 것

하지만, NN의 경우, 새로운 task를 배우는 과정에서 (새로운 task를 풀기 위해 parameter/weight를 update하는 과정에서), **기존의 task를 푸는 지식에 손상**이 갈 수 있다.

<br>

### Key words

transfer learning / catastrophic forgetting / progressive NN

<br>

# 2. Transfer Learning

**하나의 task의 지식을 다른 task를 풀기 위해 전이**

ex) NLP : pretrained embedding models ( ELMo, BERT, GPT3... )

  ( step 1 ) **model은 frozen**

  ( step 2 ) **fine-tuning**
  ( optimization은 embedder 윗 부분 header에서만 task-specific하게 수행 )

  여기에 들어간 가정 : ***서로 다른 텍스트라도, “언어 구조”라는 것 자체는 큰 차이가 없다***

ex) CV : ResNet-50, GoogLeNet, VGG-19

 <br>

위의 대표적인 두 경우 (NLP,CV)에서 알 수 있듯, fine-tuning은 effective! 하지만 단점은?

**Catastrophic Forgetting!**

<br>

# 3. Catastrophic Forgetting

**fine tuning 과정**에서 발생할 수 있는 문제점

Catastrophic Forgetting : 새로운 data를 fitting 시키는 과정에서, **기존의 중요한 feature가 변하게 되는 것!**

<br>

이를 극복하기 위해 제안된 여러 알고리즘 들 중 하나가 바로 **Progressive NN**

**Progressive NN**

- 1) Catastrophic Forgetting을 극복하면서도
- 2) 여전히 Effective



# 4. Progressive Neural Network (Prognets)

( DeepMind, 2016 )

- simple,powerful,creative

- 두 줄 요약 : 
  - 1) **“immune to forgetting” **
  - 2) **“can leverage prior knowledge via lateral connections to previously learned features”**



### Algorithm

**STEP 1) single column NN으로 시작**

- initial task 학습

- $$K$$개의 task, $$L$$개의 blocks, $$W$$ neurons per layer

- 처음이니까 lateral connection X (일반 NN와 동일)

- $$h_{i}^{k}=f\left(W_{i}^{k} h_{i-1}^{k}+\sum_{j<k}\left(U_{i}^{k: j} h_{i-1}^{j}\right)\right)$$.

<br>

[ FIGURE ] $$K=3$$, $$L=3$$

![figure2](/assets/img/CONT/img1.png)

<br>

example) Task4의 Block 3은,

- "Task 4의 Block 2"와
- "Task 1,2,3의 Block 2"  의 output으로 생성된 것!

<br>

다르게 이해하자면, **각 block을 일종의 Black Box로 생각**! 

***task를 풀기 위해 가장 중요한 information을 잘 뽑아내는 모델***

( 만약 위 그림에서 $$h_2^{(3)}$$이, $$h_1^{(1)}$$이 가장 중요하다 느낀다면 $$h_1^{(2)}$$, $$h_1^{(3)}$$을 모두 zero-out할 수 있음 )

$$\rightarrow$$  **knowledge transfer도 쉬울 뿐만 아니라, 불필요한 정보 또한 차단 OK **

<br>

### Reference

- Progressive Neural Networks ( AA Rusu et al., 2016 )

- https://towardsdatascience.com/progressive-neural-networks-explained-implemented-6f07366d714d

  






