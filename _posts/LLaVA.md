# LLaVA: Visual Instruction Tuning

## (1) Dataset

- Input: Prompt + (Caption of image, bounding box of image)
- Model: GPT4
- Output: Visual Instruction Tuning datasets (Three types)



## (2) Architecture

- [Freeze] **Vision Encoder**: CLIP ViT-L/14 
- [Train] **Projection Layer (W)**: Trainable linear projection
  - CLIP feature → LLM embedding space 변환
- **Language Model**: Vicuna (7B/13B)
  - Vicuna = LLaMA 기반 instruction-following LLM 

- (참고) 더 복잡한 방법 (Flamingo의 cross-attention, BLIP-2의 Q-former)은 future work로 남겨둠 .

  

## (3) Training

- System message + (Human instruction + Assistant answer) 시퀀스로 구성
-  토큰을 사용해 각 턴 종료 표시
- 모델은 **assistant 답변만 예측**하도록 학습 (auto-regressive) 

<br>

#### **Stage 1: Pre-training for Feature Alignment**

**목적: Alignment (Vision feature ↔ LLM)**

- Train & Freeze
  - Train: Projection layer
  - Freeze: Vision encoder + LLM
- 목적: Alignment (Vision feature ↔ LLM)

- **데이터**: CC3M → noun-phrase 기반 filtering으로 **595K image-text pairs** 선정
  - “이미지를 짧게 설명해라” 같은 **단순** instruction만 사용.
  - 이때 중요한 건 **concept coverage** (다양한 객체/상황을 포괄)

- **방식**: naive expansion → Q: “이미지를 간단히 설명해라” / A: caption

<br>

#### **Stage 2: Fine-tuning End-to-End**

**목적: 실제 instruction-following 능력 학습**

- Train & Freeze
  - Train: Projection layer + LLM
  - Freeze: Vision encoder
- **데이터**: LLaVA-Instruct-158K 
  - 앞서 GPT-4로 생성한 Conversation / Detailed / Reasoning 데이터
- **두 가지 시나리오**:
  1. **Multimodal Chatbot**: conversation + detailed + reasoning 데이터 혼합 학습
  2. **ScienceQA**: multimodal QA 데이터셋 (문제 + context(텍스트/이미지) → reasoning + 답) 단일 턴 형식으로 학습 

<br>

📌 요약

- **Stage 1**: CLIP feature ↔ LLM embedding alignment (Projection Layer만 학습)
- **Stage 2**: Instruction-following fine-tuning (Projection + LLM 학습, Vision Encoder는 frozen)
- 최종적으로 **이미지+텍스트 instruction → Assistant 답변**을 생성할 수 있는 end-to-end 모델 완성.





------



👉 여기까지 **Visual Instruction Tuning** 섹션 정리였습니다.

“다음”이라고 하시면, 이어서 **5. Experiments**(Multimodal Chatbot, LLaVA-Bench, ScienceQA 결과)로 넘어가겠습니다.



네네, 

- DT = 90점
- RF (DTxM개 parallel) = 95점 > 90점
- XGB (DTxM개 sequential) = 97점 > 90점

- RF (XGBxM개 parallel) = ?? 
- Stacking (RF+XGB+kNN+...)