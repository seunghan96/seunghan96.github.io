좋습니다 👍

**RT-2(2023)**는 **웹 스케일 VLM**을 **로봇 시연 데이터**와 함께 **공학적으로 한 덩어리**로 학습해, 카메라 이미지와 언어 지시를 곧바로 **로봇 행동 토큰**으로 내보내는 **VLA(vision-language-action)** 모델입니다. 핵심은 **행동을 텍스트 토큰으로 표현**해서, 인터넷 VQA·캡션 같은 데이터와 로봇 궤적을 **같은 형식**으로 한 모델에서 **공훈련(co-fine-tuning)** 한다는 점이에요. 



------





# **RT-2 한눈에**





- **아이디어**: 웹에서 배운 **시각-언어 지식**을 로봇 제어로 **직접 이전**. 이를 위해 **행동을 언어 토큰**으로 다루어 VLM과 **동일 포맷**으로 학습. 
- **베이스**: **PaLI-X(5B/55B)**, **PaLM-E(12B)** 등 대형 VLM/멀티모달 LLM을 기반으로 한 **RT-2-PaLI-X**, **RT-2-PaLM-E** 두 계열. 
- **학습법**: 인터넷 규모의 VQA·캡셔닝과 **로봇 궤적**을 **공-파인튜닝**; 결과적으로 **정확한 제어**와 **웹 지식 기반 추론**이 동시에 가능. 
- **능력**: 6k회 이상의 실험으로 **신물체/신명령 일반화**, **상징·숫자 이해**, **크기/가까움 비교**, 심지어 **CoT 기반 다단계 추론**(예: 망치 대용으로 바위 선택, 피곤한 사람에겐 에너지 드링크) 확인. 





------





## **입력→출력 파이프라인**





1. **Vision 토큰**: 카메라 이미지 → ViT 등으로 패치 토큰화
2. **Language 토큰**: 지시문 → 토큰 임베딩
3. **Action 토큰**: 로봇 행동을 **이산 토큰 시퀀스**로 설계(엔드이펙터 포즈, 그리퍼 열고닫기 등)
4. **Transformer**가 멀티모달 시퀀스를 받아 **다음 액션 토큰**을 오토리그레시브로 생성 → **저수준 제어**로 디코딩해 실행. 





------





## **예시 시나리오**





지시: **“Put the apple into the bowl with the star icon.”**



- RT-2는 장면에서 **사과**와 **별 아이콘**이 붙은 그릇을 식별(웹 학습으로 **아이콘 의미**를 이해) → **집기→이동→놓기** 행동 토큰 시퀀스를 생성.
- 이런 **상징/아이콘 조건부 조작**은 로봇 데이터에 없었더라도 웹에서 학습한 **시각-언어 지식**으로 제로샷 수행이 가능함이 보고됨. 





------





## **Pseudocode (학습/추론)**







### **학습(co-fine-tuning)**



```
# D_web: (image, text) for VQA/captioning, etc.
# D_robot: (image, text_instruction, action_tokens) robot trajectories

for batch in loader(mix(D_web, D_robot, ratio=α)):
    I, T, maybe_A = batch

    vision_tokens = VisionEnc(I)
    text_tokens   = TextEnc(T)
    seq_in = concat(vision_tokens, text_tokens)

    if batch in D_web:
        # 언어 목표(정답 텍스트) 예측
        logits = Transformer(seq_in)
        loss = xent(logits, target_text_tokens)
    else:  # batch in D_robot
        # 행동 목표(액션 토큰) 예측
        logits = Transformer(seq_in)
        loss = xent(logits, action_tokens)

    loss.backward(); opt.step()
```

→ **행동을 텍스트 토큰**으로 두어 **웹-태스크와 동일한 방식**으로 학습. 





### **추론(실행)**



```
def rt2_infer(image, instruction, max_len=H):
    v = VisionEnc(image)
    t = TextEnc(instruction)
    seq = concat(v, t)
    actions = []
    for _ in range(max_len):
        logits = Transformer(seq)
        a_tok = sample(logits)     # next action token
        actions.append(a_tok)
        seq = append(seq, a_tok)   # autoregressive feed-back
        if a_tok == <END>:
            break
    return ActionDecoder(actions)  # low-level commands
```

→ 이미지+지시로부터 **액션 토큰**을 생성해 곧바로 제어. 



------





## **RT-1 / PaLM-E와 뭐가 다른가?**





- **RT-1(2022)**: 로봇 시연 중심 **end-to-end policy**.
- **PaLM-E(2023)**: 초대형 멀티모달 LLM로 **설명/답변/계획**까지 폭넓게.
- **RT-2(2023)**: **웹-스케일 VLM 지식**을 **행동 토큰**으로 **직접 이전** → **상징/숫자/범주 지식**을 활용한 **제로샷 조작**이 강함. 





------





## **장점과 한계**





- **장점**: 웹 지식 기반의 **강한 일반화**, **추론(Chain-of-Thought) + 제어**의 결합, 데이터 효율성 향상. 
- **한계**: 행동을 텍스트로 이산화할 때의 **해상도 손실**, 실로봇에서 **실시간성/안전성** 보장 문제는 별도 엔지니어링이 필요. (논문도 “여전히 해야 할 일 많다”는 톤) 





------



더 원하시면, **행동 토큰 설계(좌표/그리퍼/매크로액션 등) 분해 방식**이나, **아이콘·숫자 조건부 조작** 같은 제로샷 벤치마크를 예시 이미지와 함께 풀어드릴게요.