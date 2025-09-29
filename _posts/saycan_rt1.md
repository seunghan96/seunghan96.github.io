좋습니다 👍

지금 말씀하신 **SayCan**과 **RT-1**은 **VLA(Vision-Language-Action)** 분야에서 굉장히 중요한 연구들이에요. 둘 다 구글 로보틱스 그룹에서 발표한 대표적인 work인데, 접근 방식이 다르면서도 서로 연속선상에 있습니다. 차근차근 설명드릴게요.



------





## **1. SayCan (Google, 2022)**







### **배경**





- LLM (특히 PaLM 같은 대형 언어 모델)을 로봇 제어에 연결하는 초기 시도.
- 사람은 로봇에게 **“Say”**(자연어 지시)를 하고, 로봇은 실제 환경에서 **“Can”**(가능한 행동)을 판단해야 함.
- 순수 LLM은 로봇의 물리적 제약을 모름 → “Say”와 “Can”을 결합하여 실행 가능성을 고려하는 구조 설계.







### **핵심 아이디어**





- **LLM이 high-level 계획 생성** → 예: “맥주를 가져다줘” → “1) 냉장고를 연다, 2) 맥주를 집는다, 3) 사용자에게 가져다준다”
- **로봇 affordance model이 실행 가능성 평가** → 각 step이 로봇 하드웨어와 환경에서 “Can” 가능한지 확률을 예측.
- 최종적으로 LLM이 제안한 계획 중 **실행 가능성이 높은 step**들을 선택.







### **구조**





1. **Language Model (Say)**: PaLM 같은 대규모 LLM이 자연어 명령을 받아 sequence of actions(고수준 plan)을 생성.
2. **Value Function (Can)**: 학습된 로봇 affordance model이 해당 행동을 실제로 실행할 수 있을지 평가.
3. **Say+Can 결합**: LLM이 제안한 후보 action에 대해 affordance score를 곱하여 최적 action을 선택.







### **의의**





- LLM의 지식(일상적 목표 해석) + 로봇의 물리적 가능성(실행 제약)을 합친 **VLA의 초창기 프레임워크**.
- 다양한 물체와 작업을 **few-shot prompt**만으로 수행 가능함을 보여줌.





------





## **2. RT-1 (Robotics Transformer, Google, 2022)**







### **배경**





- SayCan이 **“언어로 계획 → 로봇 행동 실행 가능성 평가”**였다면, RT-1은 로봇 제어 자체를 Transformer 모델로 통합.
- LLM과 Vision encoder를 활용해 **end-to-end policy learning**을 시도.







### **핵심 아이디어**





- **Vision + Language + Action을 토큰 시퀀스로 통합**해서 Transformer가 직접 policy를 학습.
- 입력: 카메라 이미지 + 자연어 지시어 (예: “컵을 테이블 위에 올려놔”).
- 출력: 로봇 action sequence (예: joint command, gripper action).







### **데이터**





- **130k+ 로봇 시연 데이터** (약 700가지 태스크, 13개월 동안 실제 로봇에서 수집).
- 다양한 지시어와 시각 입력을 통해 **generalization**을 학습.







### **모델 구조**





1. **Vision encoder**: 이미지 → patch embedding.
2. **Language encoder**: instruction을 토큰화.
3. **Transformer policy**: multimodal input을 받아 action 토큰을 autoregressive하게 출력.







### **특징**





- **end-to-end 학습**: SayCan처럼 LLM + affordance model을 따로 두는 게 아니라, Transformer 안에서 통합.
- **generalization**: 새로운 태스크에서도 학습된 representation을 활용해 zero-shot 성능 발휘.
- **scalability**: 데이터셋 크기를 늘릴수록 성능이 향상되는 scaling law 확인.







### **의의**





- RT-1은 사실상 **VLA의 기초 모델** 같은 역할.
- 후속으로 **RT-2 (2023)**가 나오면서 PaLM-E 같은 거대 VLA 모델로 진화.





------





## **🔑 비교 정리**



| **항목** | **SayCan (2022)**                       | **RT-1 (2022)**                      |
| -------- | --------------------------------------- | ------------------------------------ |
| 접근     | LLM + Affordance 결합                   | End-to-End Transformer               |
| 입력     | 자연어 지시                             | 이미지 + 언어                        |
| 출력     | High-level 계획 (후보 actions)          | Low-level actions (로봇 제어)        |
| 데이터   | LLM은 웹, Affordance는 시뮬/로봇 데이터 | 130k 로봇 시연 데이터                |
| 장점     | LLM 지식 활용 가능, 복잡한 계획 생성    | Generalization 강함, End-to-End 학습 |
| 한계     | 분리된 모듈 (LLM + affordance) 필요     | 대규모 데이터 없으면 학습 어려움     |



------



👉 요약하면:



- **SayCan**은 “LLM이 계획 짜고, 로봇이 가능한 행동만 실행” → **계획-실행 분리형**.
- **RT-1**은 “이미지+언어를 Transformer에 넣어 곧바로 행동 출력” → **End-to-End 실행형**.





------



혹시 원하시면 제가 이 두 논문을 **VLA의 역사적 맥락 속에서 (PaLM-E, RT-2까지)** 연결해서 정리해드릴까요?



아주 중요한 포인트를 짚으셨어요 👌

말씀하신 것처럼 **SayCan에서의 계획(Step 1~8)은 유일하지 않고, 여러 가지 경로가 있을 수 있습니다.** 예를 들어 Coke를 가져오는 방법만 해도:



- **방법 A**: 부엌 → 냉장고 → Coke 집기 → 사용자
- **방법 B**: 부엌 → 식탁 위 Coke 집기 (냉장고 안 들어갈 수도 있음)
- **방법 C**: Coke 대신 Pepsi 가져오기 (사용자가 “콜라”라고 했을 때 대체 가능)





------





## **📌 SayCan이 다양한 방법을 다루는 방식**







### **1. LLM이** 

### **여러 후보 계획**

### **을 생성**





- LLM은 prompt에 따라 **여러 개의 합리적인 계획**을 내놓습니다.

- 예:

  

  - 계획1: 냉장고에서 Coke → 가져다주기
  - 계획2: 식탁 위 Coke → 가져다주기

  

- 이때 각 계획은 여러 step으로 구성됩니다.





------





### **2. 각 step을** 

### **affordance model이 평가**





- 로봇은 환경을 관찰하면서, 후보 step이 지금 가능한지 점수를 매깁니다.

- 예:

  

  - “냉장고 열기” → 현재 로봇 위치에서는 손잡이가 안 보임 → 낮은 점수 (0.3)
  - “식탁에서 Coke 집기” → 카메라에 Coke가 인식됨 → 높은 점수 (0.8)

  





------





### **3.** 

### **Say × Can = 최종 점수화**





- Say (LLM의 plausibility: 이게 상식적으로 맞는 계획인가?)
- Can (로봇 affordance: 지금 실행 가능한가?)
- 두 값을 곱하거나 결합해서 **후보 계획별 점수**를 냅니다.





예시:

| **계획**                 | **Say 확률** | **Can 확률** | **최종 점수** |
| ------------------------ | ------------ | ------------ | ------------- |
| 냉장고에서 Coke 가져오기 | 0.9          | 0.3          | 0.27          |
| 식탁에서 Coke 가져오기   | 0.8          | 0.8          | 0.64          |

👉 이 경우, 로봇은 “식탁에서 Coke 집기” 경로를 선택합니다.



------





### **4. 실행 중** 

### **동적 재평가**





- SayCan은 한 번에 끝까지 확정하지 않고, **각 step 실행 시마다 다시 평가**합니다.

- 예:

  

  - Step 1: “부엌으로 이동” 실행 → 성공.
  - Step 2: “냉장고 열기” 시도 → 손잡이가 인식 안 됨 → 실패 → 다른 계획(식탁 Coke)으로 전환.

  





------





## **🔑 정리**





- **다양한 방법이 존재할 때** → LLM은 여러 계획을 생성.
- **로봇은 실행 가능성을 점수화하여 최적의 계획을 선택**.
- 실행 과정에서도 상황이 바뀌면 **재평가 & replanning**을 반복.





------



즉, SayCan은 **“LLM이 풍부한 지식을 활용해 다양한 후보를 제시 → 로봇이 현실에 맞게 필터링하고 조정”**하는 구조입니다.



------



👉 원하시면 제가 위 과정을 **실제 수식/의사코드(pseudocode)** 형태로 간단히 정리해드릴까요? (Say 확률 × Can 점수 계산까지 보여드리는 방식)



좋습니다 👍

SayCan의 핵심 메커니즘을 pseudocode로 정리해드릴게요.



------





## **📝 SayCan Pseudocode**



```
# Input: natural language command (instruction)
# Output: selected action to execute

def saycan(instruction, env_state):
    # 1. Say (LLM이 후보 계획 생성)
    candidate_plans = LLM.generate_plans(instruction)
    # 예: [["go_to_kitchen", "open_fridge", "grab_coke", "deliver_to_user"],
    #      ["go_to_kitchen", "grab_coke_on_table", "deliver_to_user"]]

    plan_scores = []

    # 2. 각 plan에 대해 step별 평가
    for plan in candidate_plans:
        step_scores = []
        for step in plan:
            # (a) LLM이 해당 step을 추천할 확률
            say_prob = LLM.probability(step, context=instruction)

            # (b) 로봇 affordance model이 현재 상태에서 step 실행 가능성 평가
            can_prob = affordance_model.predict(env_state, step)

            # (c) 결합 점수
            score = say_prob * can_prob
            step_scores.append(score)

        # 3. plan 전체 score = step 점수들의 평균 (또는 곱)
        plan_score = aggregate(step_scores)
        plan_scores.append((plan, plan_score))

    # 4. 가장 점수가 높은 계획 선택
    best_plan, _ = max(plan_scores, key=lambda x: x[1])

    # 5. 최종 실행할 action = best_plan의 첫 번째 step
    return best_plan[0]
```



------





## **📌 설명**





1. **LLM (Say)**

   

   - 자연어 지시어로부터 여러 후보 계획을 생성.
   - 각 step이 등장할 확률(say_prob)도 계산 가능.

   

2. **Affordance Model (Can)**

   

   - 현재 환경 상태(env_state)에서 로봇이 실제 실행할 수 있는지 확률(can_prob)을 예측.

   

3. **Say × Can 결합**

   

   - 각 step마다 score = say_prob * can_prob.
   - 전체 plan 점수는 step들의 평균이나 곱으로 집계(aggregate).

   

4. **최적 plan 선택 후 실행**

   

   - 가장 점수가 높은 plan을 고르고, 그 안에서 첫 step부터 실행.
   - 실행 후 환경 상태가 바뀌면 반복.

   





------



👉 이렇게 보면 SayCan은 **LLM의 “지식 기반 후보 생성”**과 **로봇의 “실행 가능성 예측”**을 확률적으로 곱해서 최적 plan을 찾는 구조임을 알 수 있습니다.



------





좋습니다 👍

이번에는 **RT-1 (Robotics Transformer-1, 2022, Google Research)** 의 작동 과정을 예시와 pseudocode 형태로 보여드릴게요.



------





## **🎯 RT-1 기본 아이디어**





- **입력**: (카메라 이미지, 자연어 명령)
- **출력**: 로봇의 low-level action (예: joint 값, gripper 열기/닫기)
- 구조: Vision + Language + Action → Transformer policy로 통합





즉, 사용자가 말하면 로봇이 바로 실행 가능한 제어 명령을 내보내는 **end-to-end policy** 입니다.



------





## **📌 RT-1 동작 예시**





사용자가 로봇에게 말합니다:

**“Put the apple into the bowl.”**



1. **Input Encoding**

   

   - 이미지: 카메라에서 장면 캡처 → CNN/ViT로 embedding
   - 언어: “Put the apple into the bowl” → 토큰화 후 embedding

   

2. **Transformer Policy**

   

   - Vision 토큰 + Language 토큰을 하나의 시퀀스로 결합
   - Transformer가 autoregressive하게 다음 Action Token을 예측

   

3. **Action Output (Discrete Tokens)**

   

   - 예: [MoveArm(x=0.3,y=0.1,z=0.2), CloseGripper, MoveArm(x=0.5,y=0.2,z=0.1), OpenGripper]
   - 즉, RT-1은 **Action을 토큰화(discretize)** 해서 Transformer 출력으로 다룸

   

4. **실행**

   

   - 출력된 action token들을 로봇 제어기로 보냄 → 물체 집기 + bowl에 넣기 수행

   





------





## **📝 RT-1 Pseudocode**



```
# Input: image (I), natural language instruction (T)
# Output: sequence of robot actions (A)

def RT1_policy(I, T):
    # 1. Encode vision
    vision_tokens = VisionEncoder(I)  # e.g., ViT patches

    # 2. Encode language
    text_tokens = TextTokenizer(T)    # e.g., BPE tokens
    text_embeddings = TextEncoder(text_tokens)

    # 3. Concatenate vision + language tokens
    input_sequence = concatenate(vision_tokens, text_embeddings)

    # 4. Transformer predicts action tokens
    action_tokens = []
    for step in range(MAX_ACTION_LENGTH):
        logits = Transformer(input_sequence)
        action_token = sample_from_logits(logits)
        action_tokens.append(action_token)

        # feed back for autoregressive decoding
        input_sequence = append(input_sequence, action_token)

    # 5. Decode action tokens into low-level robot commands
    actions = ActionDecoder(action_tokens)
    return actions
```



------





## **🔑 차이점 (SayCan vs RT-1)**





- **SayCan**

  

  - LLM이 “계획(Say)”을 생성 → 로봇 affordance가 “실행 가능성(Can)” 평가
  - 고수준 계획 중심, step별 실행 가능성 고려

  

- **RT-1**

  

  - Vision+Language+Action을 한 Transformer에 넣고 바로 “로봇 제어 action” 출력
  - 완전히 end-to-end, 후보 계획과 실행 가능성 분리 없음
  - 대규모 시연 데이터(130k+)로 학습 → 일반화 성능 확보

  



(see Table~\ref{} for further analysis)

------



✅ 정리하면:

RT-1은 **이미지+언어를 입력받아 바로 로봇 행동을 토큰 시퀀스로 출력하는 end-to-end Transformer**입니다.

SayCan이 **계획/실행 분리**였다면, RT-1은 **하나의 모델 안에서 통합**한 구조예요.



------



원하시면 제가 RT-1의 **실제 데이터 표현 방식 (image tokens, language tokens, action tokens 예시)**까지 표로 정리해드릴까요?