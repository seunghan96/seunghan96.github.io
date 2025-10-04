# REPE (Representation Engineering) (arXiv 2023)

📄 [논문: REPRESENTATION ENGINEERING: A TOP-DOWN APPROACH TO AI TRANSPARENCY](https://arxiv.org/pdf/2310.01405)

<br>

## (1) **아이디어**

- 기존 ACTADD, CAA는 **사전에 정의된 contrast prompts**로부터 shift vector를 학습해 사용했습니다.
- **REPE**는 inference 시점에서 **prompt pairs**를 직접 생성해 shift vector를 만들고, 이를 이용해 steering하는 방법론.
- 즉, “on-the-fly”로 프롬프트 쌍을 비교해 activation 차이를 뽑고 → steering vector를 얻어 출력 방향을 제어합니다.

한 줄 요약: **REPE**는 “prompt pairs 기반, inference-time에서 동적으로 shift vector 생성”이라는 점에서 **고정된 벡터**를 쓰는 ACTADD/CAA보다 훨씬 **유연**합니다.

<br>

## (2) **과정**

1. **Prompt pair 생성**
- Positive prompt: “The assistant responds in a polite and respectful manner.”
   - Negative prompt: “The assistant responds in a rude and offensive manner.”

2. **Activation 추출 & 차이 계산**
   - 두 prompt를 모델에 통과시켜 특정 레이어 activation을 얻고, 차이를 계산 → shift vector
   
3. **Inference 시 적용**
   - 사용자 입력 프롬프트가 들어오면, shift vector를 activation에 더해 모델 출력 방향을 제어
   

<br>

## (3) 장점

- **유연성**: 사전에 정해둔 single vector 대신, 상황마다 prompt pair를 조합해 steering 가능
- **범용성**: toxic 억제, helpfulness 강화, 스타일 제어 등 다양한 시나리오에 적용 가능

<br>

## (4) 단점



<br>

## (5) Details

- inference-time에 prompt pairs를 매번 생성해야 → **추론 비용 증가**
- shift vector 품질이 prompt pair 설계 품질에 크게 좌우됨





------



✅ 요약



- 





------



👉 “다음”이라고 해주시면, 이어서 **Wu et al. (2024)** 과 **Geiger et al. (2024)**의 activation steering 방법을 설명드리겠습니다.