# 1. VP-SDE (Variance Preserving SDE) 증명

**정의**: $dx_t=-\tfrac12\,\beta(t)\,x_t\,dt+\sqrt{\beta(t)}\,dW_t$.

- $W_t$는 표준 브라운 운동

<br>

**Brownian motion**

1. **정의**: $\{W_t\}_{t\ge0}$은 
   - $W_0=0$, 독립증분, 정규분포 증분을 가지는 연속 확률과정.
2. **분포**: $W_t - W_s \sim \mathcal{N}(0,\, t-s)$
   - $W_t \sim \mathcal{N}(0,\, t)$.
3. **성질**: $\mathbb{E}[W_t]=0,\;\text{Var}(W_t)=t,\;\text{Cov}(W_s,W_t)=\min(s,t)$.

<br>

### a) 적분인자

양변에 적분인자를 다음과 같이 설정하기!

$\rightarrow$ $I(t)=\exp\!\Big(\tfrac12\int_0^t\beta(u)\,du\Big)$

- 위와 같이 선정한 이유?
  - (1) I(t) = $\exp\!\Big(\int p(t)\,dt\Big)$.
  - (2) p(t) = $\frac{1}{2}\beta(t) $ 이므로

- $I(t) = \exp\!\Big(\int \tfrac{1}{2}\beta(t)\,dt\Big) = \exp\!\Big(\tfrac{1}{2}\int_0^t \beta(u)\,du\Big)$.

<br>

(편의 상, $I(t)$의 역수를 $\alpha(t)$로 정의함)

$\alpha(t)\;\coloneqq\;\exp\!\Big(-\tfrac12\int_0^t\beta(u)\,du\Big) \quad\Rightarrow\quad I(t)=\alpha(t)^{-1}$.

<br>

### b) Ito 미분

 $\boxed{d\!\big(I(t)\,x_t\big) =I(t)\,dx_t + x_t\,dI(t) + d\langle I,x\rangle_t}$.

$I(t)$: deterministic 함수. 따라서...

- $d\langle I,x\rangle_t=0$, 
- $dI(t)=\tfrac12\beta(t)\,I(t)\,dt$
  - Note) $I(t) = \exp\!\Big(\int \tfrac{1}{2}\beta(t)\,dt\Big)$

<br>

위로 인해,

$d\!\big(Ix\big) = I\Big(-\tfrac12\beta x\,dt+\sqrt{\beta}\,dW\Big) •	x\cdot\tfrac12\beta I\,dt = I\sqrt{\beta(t)}\,dW_t$.

<br>

양변을 $0\to t$ 적분하면

- $I(t)\,x_t = x_0 + \int_0^t I(s)\sqrt{\beta(s)}\,dW_s$,

<br>

결론:

$\boxed{ x_t = \alpha(t)\,x_0 \;+\; \int_0^t \frac{\alpha(t)}{\alpha(s)}\sqrt{\beta(s)}\,dW_s. }$.

(보통 **정규분포 잡음 한 덩어리**로 쓰려고)

$\boxed{x_t=\alpha(t)\,x_0+\sigma(t)\,z,\quad z\sim\mathcal N(0,I)}$.

형태로 표기!

<br>

## **(b) 분산 계산 (Itô isometry)**

Q) 아래에서 $\sigma^2(t)$가 어떻게 나오는지??

<br>

확률적 적분의 분산은 Itô 등가분산을 쓰면

$\mathrm{Var}\!\left[\int_0^t \frac{\alpha(t)}{\alpha(s)}\sqrt{\beta(s)}\,dW_s\right] =\int_0^t \left(\frac{\alpha(t)}{\alpha(s)}\right)^2\beta(s)\,ds$.

<br>

위 식에

$\alpha(t)=\exp\!\big(-\tfrac12\int_0^t\beta\big)$를 대입하면...

$\left(\frac{\alpha(t)}{\alpha(s)}\right)^2 =\exp\!\Big(-\!\!\int_s^t \beta(u)\,du\Big)$.

<br>

따라서

$\boxed{ \sigma^2(t) = \int_0^t \exp\!\Big(-\!\!\int_s^t \beta(u)\,du\Big)\,\beta(s)\,ds}$.

<br>

이 적분은 깔끔하게 닫힌형으로 정리.

Let $g(t) = \sigma^2(t)$

- $g(t)\;=\;\int_0^t \exp\!\Big(-\!\!\int_s^t \beta(u)\,du\Big)\,\beta(s)\,ds$.

<br>

라이프니츠 법칙으로 미분하면

- $g’(t)=\beta(t)-\beta(t)\,g(t)\;\Rightarrow\; g’(t)+\beta(t)g(t)=\beta(t),\quad g(0)=0$.

해는

$\boxed{g(t)=1-\exp\!\Big(-\!\!\int_0^t \beta(u)\,du\Big)=1-\alpha(t)^2}$.

즉

$\boxed{\sigma^2(t)=1-\alpha(t)^2}$.

<br>

## **(c) 최종 정리와 분산 보존**

요약하면

$\boxed{ x_t=\alpha(t)\,x_0\;+\;\sigma(t)\,z,\qquad \alpha(t)=\exp\!\Big(-\tfrac12\!\!\int_0^t\beta\Big),\quad \sigma^2(t)=1-\alpha(t)^2. }$.

분산은

$\mathrm{Var}[x_t]=\alpha(t)^2\,\mathrm{Var}[x_0]+\sigma^2(t) =\alpha(t)^2\,\mathrm{Var}[x_0]+(1-\alpha(t)^2).$.

<br>

특히 **데이터를 단위분산으로 정규화**(\mathrm{Var}[x_0]=1)했다면

$\mathrm{Var}[x_t]=1\quad(\forall t)$.

$\therefore$ **Variance Preserving**이 됨!

<br>

# 2. VE-SDE (Variance Exploding SDE) 증명

**정의**: $dx_t=\sqrt{\frac{d\sigma^2(t)}{dt}}\,dW_t$

- 드리프트 0
- 확산계수만 $t$의 함수

<br>

**해**: 드리프트가 없으므로

$\boxed{ x_t = x_0 + \int_0^t \sqrt{\frac{d\sigma^2(s)}{ds}}\,dW_s}$.

<br>

**분산**

$\mathrm{Var}\!\left[\int_0^t \sqrt{\frac{d\sigma^2(s)}{ds}}\,dW_s\right] =\int_0^t \frac{d\sigma^2(s)}{ds}\,ds =\boxed{\sigma^2(t)}$.



따라서

$\boxed{ x_t = x_0 + \sigma(t)\,z,\quad z\sim\mathcal N(0,I),\qquad \mathrm{Var}[x_t]=\mathrm{Var}[x_0]+\sigma^2(t)}$

$\rightarrow$ $t$가 커질수록 분산이 **계속 증가(Exploding)**

<br>

# 3. Summary

| **SDE**                                                      | **해**                                                       | **분산**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **VP-SDE** $dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW$ | $x_t=\alpha(t)x_0+\displaystyle\int_0^t \frac{\alpha(t)}{\alpha(s)}\sqrt{\beta(s)}\,dW_s$ | $\sigma^2(t)=1-\alpha(t)^2 → \mathrm{Var}[x_t]=\alpha(t)^2\mathrm{Var}[x_0]+(1-\alpha^2(t))$ |
| **VE-SDE** $dx=\sqrt{d\sigma^2/dt}\,dW$                      | $x_t=x_0+\displaystyle\int_0^t \sqrt{d\sigma^2/ds}\,dW_s$    | \mathrm{Var}[x_t]=\mathrm{Var}[x_0]+\sigma^2(t)              |

이렇게 해서 두 과정의 해가 각각 저 꼴이 되는 이유(적분인자/Itô 등가분산)와, “preserving vs exploding”이라는 이름이 **분산의 시간 거동**에서 자연스럽게 나온다는 것을 확인할 수 있습니다.



