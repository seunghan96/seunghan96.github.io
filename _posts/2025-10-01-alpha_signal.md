---
title: Alpha Signal
categories: [ASSET]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Alpha Signal

**Quant 투자(quant trading)**나 **포트폴리오 관리**에서 매우 핵심적인 개념

<br>

## 1. Alpha의 기본 정의

**Alpha ($$\alpha$$)** = 자산의 **초과수익 (excess return)**

→ 즉, **시장 평균이나 벤치마크 대비 얼마나 더(혹은 덜) 벌었는가**

<br>

$$R_i = \alpha_i + \beta_i R_m + \varepsilon_i$$.

- $$R_i$$: 자산 (또는 포트폴리오)의 수익률
- $$R_m$$: 시장 (benchmark)의 수익률
- $$\beta_i$$: 시장 민감도 (systematic exposure)
- $$\alpha_i$$: **시장으로 설명되지 않는 초과수익 (=알파)**
- $$\varepsilon_i$$: 오차항

<br>

$$\alpha$$의 해석

- $$\alpha>0$$: outperform
- $$\alpha <0$$: underperform

<br>

## 2. Alpha Signal의 의미

***“앞으로 시장을 이길 확률이 높은 신호(feature)”***

| **신호 유형**               | **설명**                                                   | **성격**                                |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| **Momentum**                | 최근 수익률이 높은 종목이 앞으로도 오를 가능성             | 단기 $$\alpha$$ 신호                    |
| **Value**                   | P/E, P/B 낮은 저평가 종목이 향후 반등 가능성               | 중기 $$\alpha$$ 신호                    |
| **Sentiment**               | (뉴스나 보고서의) 긍정/부정 감정 점수                      | 비정형 $$\alpha$$ 신호                  |
| **Theme embedding (THEME)** | 특정 theme와 의미적으로 관련 있고, 단기 수익률이 좋은 종목 | **semantic + temporal $$\alpha$$ 신호** |

즉, $$\alpha$$ 신호는 **모델이 예측 가능한 초과수익의 근거가 되는 정보**!

<br>

## 3. THEME 논문에서의 Alpha Signal

두 부분으로 구성:

1. **Semantic $$\alpha$$**:
   - theme text로부터 추출된 **장기 구조적 연관성 신호**
   - 예: “AI chipmaker” theme에 속한 기업들의 fundamental alignment
2. **Temporal $$\alpha$$**:
   - 최근 **수익률(Return) 패턴**에서 얻는 단기 신호
   - 예: 최근 60일 수익률이 상승 모멘텀인 종목을 favor

<br>

THEME의 임베딩?

- $$\text{Stock Embedding} = f(\text{semantic features}, \text{temporal $$\alpha$$ signals})$$.

- 요약 “**thematically aligned stocks 중에서도 초과수익 가능성이 높은 종목**”을 embedding 공간 상에서 가깝게 매핑

<br>

## 4. 왜 중요하냐?

- Quant 관점에서 “$$\alpha$$ 신호”는 모델의 **예측 근거(정보력)**

- THEME의 차별성
- 기존 NLP 기반 모델: **‘theme 일관성’**만 고려
  - THEME: **‘theme + 수익률 예측력($$\alpha$$)’**을 동시에 반영


<br>

## 5. Summary

- **Alpha signal** = 시장 평균보다 높은 수익을 낼 **예측 가능 신호**
- THEME은 **semantic + temporal 두 종류의 $$\alpha$$**를 embedding에 통합해 “theme에 맞고 수익도 잘 나는 종목”을 찾는 모델

