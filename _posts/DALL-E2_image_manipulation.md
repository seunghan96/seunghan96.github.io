# Image Manipulation of DALL-E v2 (feat. CLIP latent space)

<br>

# 개요

**Image Manipulation = CLIP latent 공간을 만지작거리기**

1. **Variations** = 같은 Image의 여러 버전 만들기
2. **Interpolation**  = 두 Image/Style 섞기
3. **Text-guided Editing / Text-diff** = Text로 변화 방향 지정하기

<br>

# 1. Variations

## (1) 핵심 아이디어

원본 Image → CLIP image embedding → Decoder로 **"여러 번 sampling"**

- CLIP embedding은 **고정**
- Diffusion decoder는 **noise 초기값**에 따라 매번 다른 Image를 sampling

<br>

## (2) Procedrue

1. 원본 Image $$x$$를 CLIP image encoder에 넣음
   - $$z_i = \text{CLIP\_image\_encoder}(x)$$.
2. Decoder에 z_i를 conditioning으로 주고, noise seed만 바꿔가며 여러 번 sampling
   - $$\hat{x}^{(1)} = \text{Decoder}(z_i, \epsilon^{(1)})$$.
   - $$\hat{x}^{(2)} = \text{Decoder}(z_i, \epsilon^{(2)})$$.
   - …

<br>

## (3) Results

- 모든 샘플은 **동일한 CLIP latent($$z_i$$)** 를 공유 

  → 크게 보면 **같은 콘텐츠/구도/Style** 느낌을 유지

- Noise 때문에 **디테일, 배경, 조명, 질감** 등은 조금씩 달라짐

<br>

# 2. Interpolation

## (1) 핵심 아이디어

**두 Image의 CLIP embedding을 interpolation → 그 중간 latent를 Decoder에 넣기**

- 참고) CLIP latent 공간은 Image의 **semantic 방향**을 잘 표현
- 따라서, 단순 선형 보간이나 slerp만 해도 꽤 자연스러운 “중간 Style/콘텐츠”가 나옴!

<br>

## (2) 수식

두 Image $$x_a$$, $$x_b$$에 대해..

- $$z_a = \text{CLIP\_image\_encoder}(x_a)$$.
- $$z_b = \text{CLIP\_image\_encoder}(x_b)$$.

<br>

Interpolation: $$z_\alpha = (1-\alpha) z_a + \alpha z_b \quad (\alpha \in [0,1])$$.

Slerp: $$z_\alpha = \text{slerp}(z_a, z_b, \alpha)$$.

<br>

위 결과를 Decoder에 넣기: $$\hat{x}_\alpha = \text{Decoder}(z_\alpha)$$.

<br>

## (3) Results

- $$\alpha = 0$$: 거의 x_a와 비슷
- $$\alpha = 1$$: 거의 x_b와 비슷
- 중간값: 두 Image의 **콘텐츠/Style이 섞인 Image**

<br>

# 3. Text-guided Editing / Text-diff 

## (1) 핵심 아이디어

“Image latent”와 “Text latent”가 **같은 CLIP joint space**에 있기 때문에,

Text 쪽에서 정의된 “의미 변화 방향”을 Image에도 적용할 수 있다!

<br>

## (2) Example

$$t_1 = \text{CLIP\_text\_encoder}(\text{“a photo of a cat”})$$.

$$t_2 = \text{CLIP\_text\_encoder}(\text{“a photo of a cat wearing sunglasses”})$$.

<br>

따라서, $$t_2 - t_1$$ = **“고양이에게 선글라스를 추가하는 방향”**

을 나타내는 벡터!

<br>

## (3) Application

(a) 원본 Image $$x$$의 CLIP image embedding: $$z_i = \text{CLIP\_image\_encoder}(x)$$

(b) 변화 방향 $$d = t_2 - t_1$$

(a) & (b)  **변형된 Image latent**: $$z'_i = z_i + \lambda d$$.

$$\hat{x}' = \text{Decoder}(z'_i)$$

<br>

→ 원래 Image x와 비슷하지만, “고양이가 선글라스를 쓴 버전”에 가까운 결과가 나오도록 유도.

<br>

# Summary

1. **Variations**
   - $$z_i$$ 고정, decoder sampling만 여러 번
   - → 같은 의미, 다른 디테일
2. **Interpolation**
   - $$z_\alpha = (1-\alpha) z_a + \alpha z_b$$.
   - → 두 Image/Style을 섞은 결과
3. **Text-guided Editing (Text-diff)**
   - $$d = t_2 - t_1, z'_i = z_i + \lambda d$$.
   - → Text로 정의한 의미 변화 방향을 Image에 적용