---
title: Batch Learning vs Online Learning
categories: [ML]
tags: []
excerpt:
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Batch Learning & Online Learning ]

<br>

# 1. Batch Learning

- 한번에 모든 data를 학습

- Offline Learning이라고도함

  ( online으로 streaming data를 받는 상황 X )

- 모델 한번 학습 이후, 추가적인 데이터가 들어와도 업데이트 X
  - if 재학습 시키고자하면, 기존의 데이터를 포함하여 재학습해야!

<br>

한계점 : 데이터 양이 매우 많을 경우에...?

$$\rightarrow$$ 새로운 데이터가 들어오면 실시간으로/능동적으로 학습할 필요가 있음!

$$\rightarrow$$ 이래서 등장한 것이 **Online Learning**

<br>

# 2. Online Learning

- 이미 학습이 완료된 모델에 대해서, **Mini-batch** 단위의 새로운 데이터가 주입되어서 모델을 추가적으로 학습하는 방법!

- Mini-batch 

  - 크기가 작긱 때문에, 추가적인 학습비용이 적게 듬
  - 데이터가 도착하는대로, 즉시 추가적인 학습 가능!

- **Incremental Learning** (점진적인 학습) 이라고도 부름

- 학습이 끝난 데이터는 더 이상 보관 필요 X

  $$\rightarrow$$ 저장공간 절약 가능!

<br>

언제 유용?

- 모델 학습을 위한 자원이 한정된 환경
- 새로운 데이터가 지속적으로 들어오는 환경

<br>

Online Learning에서 중요한 **Learning Rate (lr)**

$$\rightarrow$$ lr이 클 경우, 새로운 데이터에 빠르게 적응 / but 과거 데이터 정보 잘 잊음

<br>

# Reference

https://gooopy.tistory.com/123