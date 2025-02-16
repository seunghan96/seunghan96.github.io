---
title: DeepSpeed
categories: [DLF, LLM, Python, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# DeepSpeed

<br>

# 1. DeepSpeed란?

- [DeepSpeed](https://www.deepspeed.ai/)는 Microsoft에서 개발한 PyTorch 기반의 라이브러리로
- **대규모 딥러닝 모델의 학습**을 가속하고 메모리 효율을 높이는 데 사용

<br>

한 줄 요약: DeepSpeed는 ***대규모 모델 학습***을 위한 최적화된 라이브러리이며, 특히 ***ZeRO를 통해 GPU 메모리 부담을 획기적으로 줄일 수 있음***

<br>

# 2. DeepSpeed의 핵심 기능

## (1) ZeRO (Zero Redundancy Optimizer)

- 효과: **메모리 사용량 감소**

- How? 기존 DDP보다 더 효율적으로 **모델, 옵티마이저, 그래디언트**를 여러 GPU에 분산 저장

- 단계별 메모리 절약 (ZeRO-1, ZeRO-2, ZeRO-3)

  - ZeRO-1: **옵티마이저 상태**만 분산
  - ZeRO-2: **옵티마이저 상태** + **그래디언트** 분산
  - ZeRO-3: **모델 파라미터까지** 분산 (FSDP와 유사)

<br>

```python
from deepspeed import DeepSpeedConfig, init_distributed
import deepspeed

# ZeRO 설정
ds_config = {
    "train_micro_batch_size_per_gpu": 2,
    "zero_optimization": {
        "stage": 2  # ZeRO-2: 옵티마이저 상태와 그래디언트 분산
    }
}

# DeepSpeed 모델 초기화
model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
```

<br>

## (2) 3D 병렬화 (ZeRO + 모델 병렬화 + 데이터 병렬화)

- **DP (데이터 병렬화)**: 여러 GPU에서 동일한 모델을 학습 (DP, DDP)
- **MP (모델 병렬화)**: 모델을 여러 GPU에 나누어 저장 (Tensor Parallelism, Pipeline Parallelism)
- **ZeRO 메모리 최적화**까지 적용

$$\rightarrow$$ 초대형 모델 학습 가능!!

<br>

## (3) Offload 기술 (CPU/NVMe Offloading)

(**옵티마이저 상태** 등의) 일부 연산을 **"CPU 또는 NVMe로 이동"**하여 GPU 메모리 부담을 줄임

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: 파라미터까지 분산 저장
        "offload_optimizer": {
            "device": "cpu"  # 옵티마이저 상태를 CPU로 오프로딩
        }
    }
}

model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
```

<br>

## (4) 자동 혼합 정밀도 학습 (FP16, BF16)

- **FP32 → FP16 변환**: 연산 속도 증가 및 메모리 절약
- NVIDIA의 Tensor Core 최적화 활용 가능.

```python
ds_config = {
    "fp16": {
        "enabled": True  # FP16 연산 활성화
    }
}

model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
```

<br>

## (5) 비동기 I/O 및 체크포인트 최적화

- **체크포인트 저장/불러오기** 속도 향상
- 여러 GPU에서 동시에 체크포인트 로드 가능

<br>

### Gradient Checkpointing

순전파 시 모든 활성화 값(activation)을 저장하지 않고, 필요할 때 다시 계산해서 메모리 절약.

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 512)
        self.linear2 = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = checkpoint(self.linear1, x)  
        x = checkpoint(self.linear2, x)  
        return x

model = CheckpointedModel()
```

<br>

### Activation Partitioning

순전파 시 GPU에 저장되는 activation을 여러 GPU에 분산하여 메모리 절약.

```python
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": True,
        "partition_activations": True  # Activation을 여러 GPU에 분산 저장
    }
}

model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=ds_config)
```

<br>

# 3. Summary

### DeepSpeed를 사용해야 하는 경우?

- GPU 메모리가 부족한 경우 (ZeRO-3 + Offload 기능으로 해결)
- 초대형 모델 (GPT-3, LLaMA 등) 학습을 해야 할 때
- FSDP보다 더 세밀한 메모리 최적화가 필요할 때
- 데이터 병렬화 + 모델 병렬화 + ZeRO를 동시에 활용할 때

<br>

### Summary Table

| DeepSpeed 기능          | 역할                                              | 코드                            |
| ----------------------- | ------------------------------------------------- | ------------------------------- |
| ZeRO                    | 옵티마이저 상태, 그래디언트, 파라미터를 분산 저장 | `zero_optimization` 설정        |
| Gradient Checkpointing  | Activation을 다시 계산하여 메모리 절약            | `torch.utils.checkpoint` 사용   |
| Offload                 | 옵티마이저 상태를 CPU/NVMe로 이동                 | `"offload_optimizer"` 설정      |
| FP16/BF16               | 반정밀도 연산으로 메모리 절약 및 속도 향상        | `"fp16": {"enabled": True}`     |
| Activation Partitioning | Activation을 여러 GPU에 분산 저장                 | `"partition_activations": True` |
