---
title: KV Cache, Prefill chunk
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# KV Cache, Prefill chunk

Speculative decoding과 비슷한 “가속을 위한 아이디어”라는 점!

<br>

## 1. KV Cache란?

Transformer/LLM에서 **과거 token들의 Key/Value를 저장해 두는 cache**

$$\rightarrow$$ 한 번 계산해 두면 다음 token 생성 때 **다시 계산할 필요 없음**

→ 이것 때문에 **LLM의 decoding 속도가 빨라짐**

<br>

## 2. **KV Cache Prefill**

**Prefill** = Prompt 전체에 대한 KV Cache를 한 번에 (or 큰 덩어리로) 계산해 저장해두는 단계

한 줄 요약: **LLM 실행의 초기 비용을 한 번에 치르는 단계**

<br>

Ex) Prompt가 4000 token이라면...

- Prefill = 4000 token **전부를 먼저 처리**해서 KV cache에 넣는 과정
- 이후 생성 단계는 **1token씩 처리하면서 cache만 lookup**하면 됨

<br>



## 3. Prefill chunk란?

필요성

- 대형 모델에서는 Prompt가 너무 긴 경우 **전체 prefill을 한 번에 처리하면 너무 느리거나 GPU 메모리가 부족**해짐.

해결책

- Prompt를 **여러 chunk(조각)로 나누어**
- 각 chunk마다 KV Cache를 부분적으로 채워 넣는 방식

예시

- Prompt = 10,000 token

- chunk size = 2,000 token

```
Chunk 1 (0~1999 token) → KV Cache 저장
Chunk 2 (2000~3999 token) → KV Cache 추가 저장
Chunk 3 (4000~5999 token)
Chunk 4 (6000~7999 token)
Chunk 5 (8000~9999 token)
```

<br>

이렇게 하면:

- GPU RAM 최적화
- Latency도 더 일정하게
- Streaming 입력 (실시간 입력)도 쉽게 처리됨

<br>

## 4. Prefill chunk의 중요성

1. **긴 문서 prompt가 많아짐**

   - 100k–1M token context 모델 시대

2. **전체 prefill을 한 번에 하면 메모리 터짐**

3. chunk 단위 prefill은

   - 메모리 안전

   - GPU 활용도 증가

   - 대기 시간 감소

   - 파이프라이닝이 가능

     이런 이점이 있음.

<br>

## 5. 실제 예시 (초간단 흐름)

```python
for chunk in split(prompt, chunk_size):
    hidden, kv = model.forward(chunk, kv_cache)
    kv_cache.append(kv)
```

이렇게 chunk 단위로 **prefill → KV Cache 누적** 과정을 수행

<br>

마지막 chunk까지 끝나면 LLM은 이제 full prompt에 대한 KV Cache를 보유!

그 다음부터는:

```
decode next token → KV 추가
decode next token → KV 추가
```

<br>

## 6. System Prompt & User Prompt

System Prompt와 Prefill Chunk의 실제 연관성?

- LLM 실행에서 **system prompt와 user prompt는 역할이 다르기 때문에**, KV Cache prefill chunk의 필요성도 두 가지 관점!

- (1) **System Prompt는 길이가 고정되어 있으므로, 미리 KV Cache를 만들어둘 수 있음**

  - System prompt는 **모델 서비스 배포 시 이미 확정된 텍스트**

  - 길이를 정확히 알고 있으므로,

    - 미리 **KV Cache를 prefill → 저장**해 놓을 수 있고
    - Serving 시에는 **즉시 decode 단계로 진입 가능**

  - 즉, system prompt는 **static prompt** → prefill 완전 가능

    → 이것만으로도 **초기 응답 속도(latency)가 크게 단축됨**

- (2) **User Prompt는 길이를 예측할 수 없으므로, 전체 prefill을 미리 할 수 없음**

  - User prompt의 길이는 요청마다 **달라짐**
    - 10 token일 수도
    - 10,000 token일 수도 

  - 그래서 “전체 user prompt에 대해 prefill을 한 번에 계산”하는 방식은

    - GPU 메모리 예측 불가능
    - 최악의 경우 OOM 발생!!

    → 이 문제를 해결하는 것이 **prefill chunk**

<br>

User prompt를 **chunk 단위**로 쪼개어

- GPU 메모리 사용을 일정하게 유지
- 긴 입력도 안전하게 처리
- 스트리밍 입력도 지원

<br>

## 6. Summary

- **Prefill** = KV Cache 미리 채워두기

- **Prefill chunk** = Chunk 단위로 계산해서 계산량 상한 걸어두기

- **System prompt**:

  고정 길이 → KV Cache를 사전에 prefill하여 저장 가능 → latency 감소

- **User prompt**:

  길이 예측 불가 → 전체 prefill 불가능 → **prefill chunk로 GPU memory를 안전하게 유지**

