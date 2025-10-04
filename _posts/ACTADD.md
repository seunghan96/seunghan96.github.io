# ACTADD (Activation Addition) (ICLR 2025 submission)

📄 [논문: STEERING LANGUAGE MODELS WITH ACTIVATION ENGINEERING](https://arxiv.org/pdf/2308.10248)

<br>

## 1. 아이디어

- **Contrast prompt**를 활용해서 **shift vector**를 만드는 방법.
- 예:
  - 긍정적인 프롬프트(“The answer is helpful, polite, …”)
  - 부정적인 프롬프트(“The answer is toxic, rude, …”)
- 두 프롬프트를 모델에 입력했을 때의 **activation 차이**를 계산 

→ 이것이 곧 “원하는 방향 - 회피할 방향”을 나타내는 벡터.

<br>

## 2. **과정**

1. **Contrast pair** 준비: $$(prompt^+, prompt^-)$$
2. 해당 프롬프트를 LM에 넣고 **특정 레이어에서의 hidden activation** 추출
3. ***차이(Δactivation) = shift vector***로 정의
4. Inference 시, 모델의 activation에 이 shift vector를 더해주면, **모델 출력이 원하는 방향으로 이동**

<br>

## 3. **특징**

- **학습 불필요**: optimization 없이 activation level에서 직접 제어
- **단순하지만 강력**: 특정 속성(예: toxic→non-toxic) 제어 가능

<br>

## 4. **한계**

- **단일 contrast prompt pair**로만 작동 → 데이터 다양성이 부족
- 따라서 **일반화 성능 제한** (프롬프트 상황이 바뀌면 효과 떨어짐)

<br>

![figure2](/assets/img/llm/img820.png)

![figure2](/assets/img/llm/img821.png)

![figure2](/assets/img/llm/img822.png)