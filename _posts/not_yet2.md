좋습니다 👍 이어서 **Wu et al. (2024)** 과 **Geiger et al. (2024)** 연구를 설명드릴게요.



------





## **🔹 Wu et al. (2024)**





📄 [논문: Counterfactual Steering of Language Models via Low-Rank Projections](https://arxiv.org/pdf/2404.03592)





### **아이디어**





- 모델 내부 activation을 직접 다루되, **low-rank projection**을 통해 효율적으로 제어.
- 특정 속성(예: toxic vs non-toxic, helpful vs harmful)에 대한 activation 차이를 counterfactual 방식으로 학습.
- **Counterfactual 표현**을 도입해, “만약 이 입력이 다른 속성을 가졌다면 activation이 어떻게 바뀌었을까?”를 추정하고 이를 통해 추가적인 정보 교환 및 제어 가능.







### **특징**





- **저차원(low-rank) 공간**에 projection → 계산 효율적
- 다양한 속성을 동시에 제어 가능
- counterfactual 설정으로 더 정밀한 control 가능





------





## **🔹 Geiger et al. (2024)**





📄 [논문: Rotating Features to Align Representations in Language Models](https://arxiv.org/pdf/2303.02536)





### **아이디어**





- 특정 속성을 나타내는 **feature subspace**를 찾아내고, 이 subspace에서 **rotation**을 적용해 activation을 alignment.
- 즉, feature 공간을 회전시켜서 모델이 “원하는 방향”으로 더 잘 정렬되도록 만드는 방식.







### **특징**





- 기존 shift vector 방식은 단순 평행이동(translation)만 제공 → 표현력 한계.
- Rotation은 subspace 내부 구조를 유지하면서 방향만 바꾸므로, **보다 세밀한 activation alignment**가 가능.
- 속성 간 간섭(interference)을 줄이고, multi-attribute control에 유리.





------





## **✅ 두 연구의 차별점**





- **Wu et al.**: low-rank projection + counterfactual 표현 → 효율적이고 정밀한 제어
- **Geiger et al.**: feature subspace rotation → 표현 간 alignment 개선, 속성 간 충돌 최소화





------



👉 “다음”이라고 해주시면, 이어서 **Language Model Arithmetic (Dekoninck et al., 2023)**를 설명드리겠습니다.