## **1. 직관 (Intuition)**

- **문제**: Diffusion model은 기본적으로 조건부 생성(예: text-to-image)을 할 수 있음

  $$\rightarrow$$ 다만, **조건을 강하게 따르게 만들기가 쉽지 않음**!!

  - 너무 강하면 품질이 떨어지고
  - 너무 약하면 조건이 잘 반영되지 않음

- **해결**: Classifier-free guidance (CFG)는 

  - **conditional prediction**과 

  - **unconditional prediction**을 

    함께 사용해서, 조건을 더 “강조”할 수 있게 만든 기법!

- **핵심 아이디어**: 모델이 예측한 두 가지 noise 분포를 섞음

  - 조건 없는 (no condition) 예측
  - 조건 있는 (with condition) 예측

  $$\rightarrow$$ ***"둘의 차이"***를 이용해 조건을 강조하는 방향으로 샘플링을 유도.

<br>

## **2. 수식 (Formulation)**

## (1) Diffusion model

Diffusion model은 step $$t$$에서 noise를 예측:

- $$\epsilon_\theta(x_t, t, c)$$.

  - $$x_t$$: Noisy sample at step $$t$$

  - $$c$$: 조건 (예: text prompt), 없으면 $$\varnothing$$

<br>

## (2) CFG 공식

$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \left( \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing) \right)$$.

- $$\epsilon_\theta(x_t, t, \varnothing)$$: **"Unconditional"** prediction
- $$\epsilon_\theta(x_t, t, c)$$: **"Conditional"** prediction
- $$w$$: Guidance scale (보통 5~10 사용)

<br>

# 3. Example

조건 (prompt): *“a cat wearing sunglasses”*

- Unconditional 예측: 그냥 **“동물 모양”**에 가까운 fuzzy noise 제거
- Conditional 예측: **“고양이 + 선글라스”** 형태의 구체적인 noise 제거 방향

최종 guidance 예측: 

- **"Unconditional"** 결과에 **"conditional"** 방향을 $$w$$배 강화

<br>

$$w$$의 효과

- $$w=0$$: 조건 무시 (unconditional)
- $$w=1$$: 보통 conditional 결과
- $$w$$ 클수록: 조건을 더 강하게 반영하지만, 너무 크면 artifacts 발생