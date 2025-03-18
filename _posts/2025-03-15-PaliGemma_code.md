---
title: PaliGemma 구현 Part 1
categories: [LLM, MULT, NLP]
tags: []
excerpt: modeling_siglip, processing_paligemma

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# PaliGemma 구현 Part 1

(Reference: https://www.youtube.com/watch?v=vAmKB7iPkWw&t=19426s)

- **[Vision] `modeling_siglip.py`**
- **[Vision2Text] `processing_paligemma.py`**
- [Total] `modeling_gemma.py`

<br>

# [Vision] `modeling_siglip.py`

`SiglipVisionModel` (configuration: `SiglipVisionConfig`)

- `SiglipVisionTransformer`
  - `SiglipVisionEmbeddings`: ViT의 (첫 번째) patch embedding layer
  - `SiglipEncoder`: ViT의 (main) Encoder
    - `SiglipEncoderLayer`: ViT Encoder의 layer
      - `SiglipAttention`: MHA
      - `SiglipMLP`: MLP

<br>

# [Vision2Text] `processing_paligemma.py`

`PaliGemmaProcessor`: Gemma의 입력을 위한 image & text 토큰 전처리

- `add_image_tokens_to_prompt`:  image token을 text token 앞에 이어서 붙인다.
- `process_images`: numpy로 된 image를 전처리하는 함수
  - `rescale`, `resize`, `normalize`

<br>

# [Total] `modeling_gemma.py`

`PaliGemmaForConditionalGeneration`  (configuration: `PaliGemmaConfig` - `GemmaConfig`)

- `SiglipVisionModel`: Vision encoder
- `PaliGemmaMultiModalProjector`: Vision encoder에서 나온 image token을 LLM space로 projection
- `GemmaForCausalLM`: Multimodal decoder + LM head
  - `GemmaModel`: Multimodal decoder
    - `GemmaDecoderLayer`: Multimodal decoder layer
      - `GemmaAttention`
        - `GemmaRotaryEmbedding`: RoPE
      - `GemmaMLP`: MLP layer (gating + gelu)
      - `GemmaRMSNorm`: RMS norm
    - `GemmaRMSNorm`: RMS norm

<br>

# 1. Vision (`modeling_siglip.py`)

```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
```

<br>

(1) `SiglipVisionConfig`: Vision Encoder의 configuration

```python
class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
```

<br>

(2) `SiglipVisionModel`: Vision Encoder

```python
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [B,C,H,W] -> [B,N,D], where N = number of patches
        return self.vision_model(pixel_values=pixel_values) 
```

<br>

(3) `SiglipVisionTransformer`: Vision Encoder 모델 = ViT

```python
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,N,D]
        hidden_states = self.embeddings(pixel_values)
        # SAME
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        # SAME
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
```

<br>

(4) `SiglipVisionEmbeddings`: ViT의 (첫 번째) patch embedding layer

```python
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size, # w/o overlap
            padding="valid", # padding (X)
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [B,C,H,W]
        _, _, height, width = pixel_values.shape # 
        # [B,C,H,W] -> [B,D,N**0.5,N**0.5]
        patch_embeds = self.patch_embedding(pixel_values)  
        # [B,D,N**0.5,N**0.5] -> [B,D,N]
        embeddings = patch_embeds.flatten(2)
        # [B,D,N] -> [B,N,D]
        embeddings = embeddings.transpose(1, 2)
        # SAME
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
```

<br>

(5) `SiglipEncoder`: ViT의 (main) Encoder

```python
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # [B,N,D]
        hidden_states = inputs_embeds
				# SAME
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
```

<br>

(6) `SiglipEncoderLayer`: ViT Encoder의 layer

```python
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # [B,N,D]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

<br>

(7) `SiglipAttention`: MHA와 동일

```python
class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # [B,N,D]
        batch_size, seq_len, _ = hidden_states.size()
        
        # SAME
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # [B,N,D] -> [B,H,N,d] ... D=Hxd
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [B,H,N,N]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # [B,N,D] -> [B,H,N,d]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [B,H,N,d] -> [B,N,H,d]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B,N,H,d] -> [B,N,D]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
```

<br>

(8) `SiglipMLP`: MLP

```python
class SiglipMLP(nn.Module):
	  def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B,N,D] -> [B,N,D2]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [B,N,D2] -> [B,N,D]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

<br>

# 2. Vision2Text (`processing_paligemma.py`)`

```python
from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
```

<br>

`PaliGemmaProcessor`: Gemma의 입력을 위한 image & text 토큰 전처리

- Tokenizer 정보: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md

```python
class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        # (Extra tokens) Object detection용
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  
        # (Extra tokens) Object segmentation용
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # BOS, EOS를 직접 더할 것이기 때문에 False로!
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # (1) Image 전처리
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # (2) Image stack & tensor화: [B,C,H,W]
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        # (3) Image token의 길이(개수)만큼 "<image>"를 앞에 붙이기 (prepend)
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # (4) (Image + BOS + Text + \n) Tokenize
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
```

<br>

`add_image_tokens_to_prompt`: image token을 text token 앞에 이어서 붙인다.

```python
def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
```

<br>

`process_images`: numpy로 된 image를 전처리하는 함수

- 세부 구성: `rescale`, `resize`, `normalize`

```python
def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    # (1) 사전에 정의된 크기로 resize하기
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
		# (2) Numpy로 변환
    images = [np.array(image) for image in images]
    # (3) Rescale (0~1 사이로)
    images = [rescale(image, scale=rescale_factor) for image in images]
    # (4) Normalization (정규화)
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # (5) Reshape: (C,H,W) 가 되도록
    images = [image.transpose(2, 0, 1) for image in images]
    return images
```

```python
# 사전에 정의된 크기로 resize하기
def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image
  
# Rescale (0~1 사이로)
def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

# Normalization (정규화)  
def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image
```
