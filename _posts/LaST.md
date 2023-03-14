## LaST

Latent Seasonal-Trend representations for time series forecasting



use **SEASONAL & TREND** characteristics for **disentanglement learning**

( can be easily extended to adapt to situations that have more than two components to dissociate )

<br>

### (1) Problem definition. 

- TS dataset $\mathcal{D}$ 
  - $X_{1: T}^{(i)}=\left\{x_1^{(i)}, x_2^{(i)}, \cdots, x_t^{(i)}, \cdots, x_T^{(i)}\right\}$, where $i \in\{1,2, \ldots, N\}$
    - $x_t^{(i)}$ : uni/multi-variate value
- Output of model : $Z_{1: T}$, 
  - which is suitable for predicting future sequences $Y=\hat{X}_{T+1: T+\tau}$. 

<br>

Representation $Z$ :

- $P(X, Y)=P(Y \mid X) P(X)=\int_Z P(Y \mid Z) P(Z \mid X) d Z \int_Z P(X \mid Z) P(Z) d Z$.

<br>

## (2) Variational Inference 

Likelihood $P(X \mid Z)$ :

- introduce variational distn $Q_\phi(Z \mid X)$ 

- evidence lower bound (ELBO) :

  $\begin{aligned}
  \log P_{\Theta}(X, Y) \geq & \log \int_Z P_\psi(Y \mid Z) Q_\phi(Z \mid X) d Z \\&+\mathbb{E}_{Q_\phi(Z \mid X)}\left[\log P_\theta(X \mid Z)\right] \\
  & -\mathbb{K} \mathbb{L}\left(Q_\phi(Z \mid X) \| P(Z)\right) \\&=\mathcal{L}_{E L B O},
  \end{aligned}$.

  - where $\Theta$ is composed of $\psi, \phi$, and $\theta$ denotes learned parameters.

<br>

Faces an entanglement problem & cannot clearly extract complicated temporal patterns. 

<br>

Solution : incorporate the ***decomposition strategy***

- learns a couple of **disentangled representations** to depict SEASONAL & TREND dynamics

<br>

Decomposition :

- $X=X^s+X^t$ 

  - formulate the temporal signals $X$ and $Y$ as the sum of seasonal and trend components

- $Z$ : factorized into $Z^s$ and $Z^t$

  - assumed to be independent of each other 

    ( $P(Z)=P\left(Z^s\right) P\left(Z^t\right)$ )

<br>

<img src="/Users/LSH/Library/Application Support/typora-user-images/image-20230307181422873.png" alt="image-20230307181422873" style="zoom:50%;" />.

- (a) Representations learning :
  -  producing **disentangled seasonal-trend representations**
- (b) Prediction :
  - based on learned representations $Z$

<br>

BEFORE :

$\begin{aligned}
\mathcal{L}_{E L B O} &=  \log \int_Z P_\psi(Y \mid Z) Q_\phi(Z \mid X) d Z \\&+\mathbb{E}_{Q_\phi(Z \mid X)}\left[\log P_\theta(X \mid Z)\right] \\
& -\mathbb{K} \mathbb{L}\left(Q_\phi(Z \mid X) \| P(Z)\right),
\end{aligned}$.

<br>

Theorem 1. With the decomposition strategy …..

$\begin{aligned}
\mathcal{L}_{E L B O} &=\log \int_{Z^s} \int_{Z^t} P_\psi\left(Y \mid Z^s, Z^t\right) Q_{\phi^s, \phi^t}\left(Z^s, Z^t \mid X\right) d Z^s d Z^t \quad \text { (predictor) } \\
\\& +\mathbb{E}_{Q_{\phi^s\left(Z^s \mid X\right)}}\left[\log P_{\theta^s}\left(X^s \mid Z^s\right)\right]+\mathbb{E}_{Q_{\phi^t}\left(Z^t \mid X\right)}\left[\log P_{\theta^t}\left(X^t \midZ^t\right)\right] \quad \text { (reconstruction) } \\
& -\mathbb{KL}\left(Q_{\phi^s}\left(Z^s \mid X\right) \| P\left(Z^s\right)\right)-\mathbb{K} \mathbb{L}\left(Q_{\phi^t}\left(Z^t \mid X\right)|| P\left(Z^t\right)\right)\\
&
\end{aligned}$.

<br>

Interpretation : ELBO is split into 3 main units

- a) Predictor
- b) Reconstruction 
- c) KL Divergence

<br>

### a) Predictor  

$\log \int_{Z^s} \int_{Z^t} P_\psi\left(Y \mid Z^s, Z^t\right) Q_{\phi^s, \phi^t}\left(Z^s, Z^t \mid X\right) d Z^s d Z^t$.

- makes forecasting and measures the accuracy
- regarded as the sum of two independent parts
  - (1) $\log \int_{Z^s} P_{\psi^s}\left(Y^s \mid Z^s\right) Q_{\phi^s}\left(Z^s \mid X\right) d Z^s$
  - (2) $\log \int_{Z^t} P_{\psi^t}\left(Y^t \mid Z^t\right) Q_{\phi^t}\left(Z^t \mid X\right) d Z^t$
- a-1) Seasonal Predictor : 
  - input : $Z^s \in \mathbb{R}^{T \times d}$ 
  - step 1) employs the discrete Fourier transform (DFT) algorithm to detect the seasonal frequencies
    - $Z_{\mathcal{F}}^s=\operatorname{DFT}\left(Z^s\right) \in \mathbb{C}^{F \times d}$ , where $F=\left\lfloor\frac{T+1}{2}\right\rfloor$  ( due to the Nyquist theorem )
  - step 2) inverse the frequencies back to the temporal domain
    - $\tilde{Z}^s=\operatorname{iDFT}\left(Z_{\mathcal{F}}^s\right) \in \mathbb{R}^{\tau \times d}$.
- a-2) Trend Predictor : feed forward network (FFN)
  - input : $Z^t \in \mathbb{R}^{T \times d}$ 
- a-3) Final Prediction : $\hat{Y} = \hat{Y^s} + \hat{Y^t}$

<br>

### b) Reconstruction 

$\mathbb{E}_{Q_{\phi^s\left(Z^s \mid X\right)}}\left[\log P_{\theta^s}\left(X^s \mid Z^s\right)\right]+\mathbb{E}_{Q_{\phi^t}\left(Z^t \mid X\right)}\left[\log P_{\theta^t}\left(X^t \mid Z^t\right)\right]$.

- regularization term (1)
- cannot be directly measured owing to the unknown $X^s$ and $X^t$
- *Theorem 2

<br>

### c) KL divergence 

$\mathbb{KL}\left(Q_{\phi^s}\left(Z^s \mid X\right) \| P\left(Z^s\right)\right)-\mathbb{K} \mathbb{L}\left(Q_{\phi^t}\left(Z^t \mid X\right)|| P\left(Z^t\right)\right)$.

- regularization term (2)
- can be easily estimated by Monte Carlo sampling 
- priors (for both) : $\mathcal{N}(0, I)$ for efficiency ( Appendix C. )

<br>

Theorem 2

- under the **Gaussian distribution** assumption….
- reconstruction loss $\mathcal{L}_{\text {rec }}$ : 
  - can be estimated without leveraging $X^s$ and $X^t$,

- $\mathcal{L}_{\text {rec }}=-\sum_{\kappa=1}^{T-1}\left\|\mathcal{A}_{X X}(\kappa)-\mathcal{A}_{\hat{X}^s \hat{X}^s}(\kappa)\right\|^2+\operatorname{CORT}\left(X, \hat{X}^t\right)-\left\|\hat{X}^t+\hat{X}^s-X\right\|^2$.
  - where $\operatorname{CORT}\left(X, \hat{X}^t\right)=\frac{\sum_{i=1}^{T-1} \Delta X_i^t \Delta \hat{X}_i^t}{\sqrt{\sum_{i=1}^{T-1} \Delta X^t} \sqrt{\sum_{i=1}^{T-1} \Delta \hat{X}^t}}$.
    - CORT = temporal correlation coefficient
    - $\Delta X_i=X_i-X_{i-1}$ : the first difference
    - Appendix A.2
  - where $\mathcal{A}_{X X}(\kappa)=\sum_{i=1}^{T-\kappa}\left(X_t-\bar{X}\right)\left(X_{t+\kappa}-\bar{X}\right)$.
    - autocorrelation coefficient with lagged value $\kappa$ 
    - Appendix B.2

<br>

Drawbacks: 

- (1) tends to **narrow down the distance between posterior & prior**

  ​	$\rightarrow$ modeling choice tends to sacrifice the variational inference vs. data fit

- (2) The disentanglement of the seasonal-trend representations is boosted **indirectly** by the separate reconstruction

  ​	$\rightarrow$ need to impose a **direct** constraint on the representations themselves.

<br>

Solution : alleviate these limitations by introducing additional **Mutual Information (MI) regularization terms**

- (1) **increase** MI between $Z^s, Z^t$ and $X$ 
  - to alleviate the divergence narrowing problem [44, 45]
- (2) **decrease** MI between $Z^s$ and $Z^t$ 
  - to further dissociate their representations

<br>

Objective function of LaST ( maximize )

- $\mathcal{L}_{L a S T}=\mathcal{L}_{E L B O}+I\left(X, Z^s\right)+I\left(X, Z^t\right)-I\left(Z^s, Z^t\right)$.
  - 2 mutual information terms are untraceable 

<br>

## Mutual information bounds for optimization

address the traceable MI bounds, 

- maximizing $I\left(X, Z^s\right)$ and $I\left(X, Z^t\right)$
- minimizing $I\left(Z^s, Z^t\right)$ in Eq. (8)

and provide lower and upper bounds for the model optimization.

<br>

### (1) Lower bound for $I\left(X, Z^s\right)$ or $I\left(X, Z^t\right)$

- ( omit the superscript $s$ or $t$ )
- $I(X, Z) \geq \mathbb{E}_{Q_\phi(X, Z)}\left[\gamma_\alpha(X, Z)\right]-\log \mathbb{E}_{Q(x) Q_\phi(z)}\left[e^{\gamma_\alpha(X, Z)}\right]=I_{\mathrm{MINE}}$, 
  - where $\gamma_\alpha$ is a learned normalized critic with parameters $\alpha$. 
- However, this bound suffers from the biased gradient owing to the parametric logarithmic term (see Appendix A.3 for proof). Inspired by [47], we substitute the logarithmic function by its tangent family to ameliorate the above biased bound:
  - $\begin{aligned}
    I_{\mathrm{MINE}} & \geq \mathbb{E}_{Q_\phi(X, Z)}\left[\gamma_\alpha(X, Z)\right]-\left(\frac{1}{\eta} \mathbb{E}_{Q(x) Q_\phi(z)}\left[e^{\gamma_\alpha(X, Z)}\right]+\log \eta-1\right) \\
    & \geq \mathbb{E}_{Q_\phi(X, Z)}\left[\gamma_\alpha(X, Z)\right]-\frac{1}{\eta} \mathbb{E}_{Q(x) Q_\phi(z)}\left[e^{\gamma_\alpha(X, Z)}\right],
    \end{aligned}$,
    - where $\eta$ denotes the different tangent points. 

- First Inequality : concave negative logarithmic function
  - values on the curve are upper bounds for that on the tangent line, and is tight when the tangent point overlaps the independent variable, i.e., the true value of $\mathbb{E}_{Q(x) Q(z)}\left[e^{\gamma(X, Z)}\right]$. The closer the distance between tangent point and independent variable, the greater the lower bound. Therefore, we set $\eta$ as the variational term $\mathbb{E}_{Q(x) Q_\phi(z)}\left[e^{\gamma_\alpha(X, Z)}\right]$ that estimates the independent variable to obtain as great lower bound as possible. 
- Second inequality, $\gamma_\alpha(x, z)$ - a critic function activated by Sigmoid - is limited within $[0,1]$ and thus $-(\log \eta-1) \geq 0$. This inequality is tight only if $\mathbb{E}_{Q(x) Q_\phi(z)}\left[\gamma_\alpha(X, Z)\right]=1$, which means $\gamma_\alpha$ can discriminate whether a pair of variables $(X, Z)$ is sampled from the joint distribution or marginals. Similarly to MINE, this consistency problem can be addressed by the universal approximation theorem for neural networks [52]. Thus, Eq. (9) provides a flexible and scalable lower bound for $I(X, Z)$ with an unbiased gradient.

For the evaluation, we exploit a traceable manner $[53,51]$ that draws joint samples $\left(X^{(i)}, Z^{(i)}\right)$ by $Q\left(Z^{(i)} \mid X^{(i)}\right) P_{\mathcal{D}}\left(X^{(i)}\right)$. As for the marginal $Q_\phi(Z)$, we randomly select a datapoint $j$ and then sample it from $Q_\phi\left(Z \mid X^{(j)}\right) P_{\mathcal{D}}\left(X^{(j)}\right)$. Details of the optimization are shown in Algorithm 1 .

<br>

### (2) Upper bound for $I\left(Z^s, Z^t\right)$

- introduce an energy-based variational family for $Q\left(Z^s, Z^t\right)$ 
  - uses a normalized critic $\gamma_\beta\left(Z^s, Z^t\right)$ to establish a traceable upper bound
  - incorporate the critic $\gamma_\beta$ into the upper bound $I_{\text {CLUB }}$ to obtain a traceable Seasonal-Trend Upper Bound (STUB) for $I\left(Z^s, Z^t\right)$
  - $\begin{aligned}
    I\left(Z^s, Z^t\right) & \leq \mathbb{E}_{Q\left(Z^s, Z^t\right)}\left[\log Q\left(Z^s \mid Z^t\right)\right]-\mathbb{E}_{Q\left(Z^s\right) Q\left(Z^t\right)}\left[\log Q\left(Z^s \mid Z^t\right)\right]=I_{\text {CLUB }} \\
    & =\mathbb{E}_{Q_{\phi^s, \phi^t}}\left(Z^s, Z^t\right)\left[\gamma_\beta\left(Z^s, Z^t\right)\right]-\mathbb{E}_{Q_{\phi^s}\left(Z^s\right) Q_{\phi^t}\left(Z^t\right)}\left[\gamma_\beta\left(Z^s, Z^t\right)\right]=I_{\text {STUB }}  \\
    &
    \end{aligned}$.
  - (Appendix D.2)
  - inequality is tight only if $Z^s$ and $Z^t$ are a pair of independent variables
  -  [55]. This is exactly a sufficient condition for $I_{\mathrm{STUB}}$, since MI and Eq. (11) are both zeros on the independent situation, which is our seasonaltrend disentanglement optimal objective. The critic $\gamma_\beta$, similar to $\gamma_\alpha$, takes on the discriminating responsibility but provides converse scores, constraining the MI to a minimum. However, Eq. (11) may get negative values during the learning of parameter $\beta$, resulting an invalid upper bound for MI. To alleviate this problem, we additionally introduce a penalty term $\left\|I_{S T U B}^{n e g}\right\|^2$ to assist the model optimization, which is an L2 loss of the negative parts in $I_{\text {STUB }}$.

For the evaluation, we take the same sampling manner as the one in the lower bound and optimization details are also shown in Algorithm 1.