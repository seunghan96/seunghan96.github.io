TimeVAE

- Requires a user-defined distribution for its probabilistic process

<br>

Probabilistic TS generator originated from diffusion models

- More flexible 
- Focus on TSGM (Lim et al., 2023) 
  - first and only work on this novel design. 

<br>

## (1) Problem Formulation

- MTS: $\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$, 
- Goal: synthesize $\boldsymbol{x}_{1: T}^0$ 
  - by generating observation $\boldsymbol{x}_t^0$ at time point $t \in[2, T]$ with the consideration of its previous historical data $x_{1: t-1}^0$. 
  - Correspondingly, the target distribution is the conditional density $q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t-1}^0\right)$ for $t \in[2, T]$, and the associated generative process involves the recursive sampling of $x_t$ for all time points in the observed period. Details about the training and generation processes will be discussed in the next subsection.



## (2) TSGM

### Conditional score-based time series generative model (TSGM)

( Only work to study the TS generation based on the diffusion )

- Conditionally generate each TS observation based on the past generated observations
- Includes three components
  - (1) Encoder
  - (2) Decoder
  - (3) Conditional score-matching network
    - used to sample the hidden states, which are then converted to the TS samples via the decoder

- Input) MTS $\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$
- Mapping) $\boldsymbol{h}_t^0=\mathbf{E n}\left(\boldsymbol{h}_{t-1}^0, \boldsymbol{x}_t^0\right), \quad \hat{\boldsymbol{x}}_t^0=\mathbf{D e}\left(\boldsymbol{h}_t^0\right)$
  - $\hat{\boldsymbol{x}}_t^0$ : Reconstructed TS at $t$ step
  - Recursive process 
    - En & De : constructed with the RNN
- Objective function (for both En & De) $\mathcal{L}_{E D}$ 
  - $\mathcal{L}_{E D}=\mathbb{E}_{\boldsymbol{x}_{1: T}^0}\left[\left\|\hat{\boldsymbol{x}}_{1: T}^0-\boldsymbol{x}_{1: T}^0\right\|_2^2\right] $.

<br>

Conditional score matching network $s_{\boldsymbol{\theta}}$

- Designed based on the SDE formulation of diffusion models
- Based on U-net
- Focuses on the **generation of hidden states** rather than producing the TS directly

<br>

Generation of hidden states

- Instead of applying the diffusion process to $\boldsymbol{x}_t^0$ ....

  Hidden states $\boldsymbol{h}_t^0$ is diffused to a Gaussian distribution by the following forward SDE

  - $\mathrm{d} \boldsymbol{h}_t=f\left(k, \boldsymbol{h}_t\right) \mathrm{d} k+g(k) \mathrm{d} \boldsymbol{\omega}$.
  - where $k \in[0, K]$ refers to the integral time. 

<br>

With the diffused sample $\boldsymbol{h}_{1: t}^k$,  $s_{\boldsymbol{\theta}}$ learns the gradient of the conditional log-likelihood function

Loss function: $\mathcal{L}_{\text {Score }}=\mathbb{E}_{\mathbf{h}_{1: \mathbf{T}}^{\mathbf{0}}, \mathbf{k}} \sum_{t=1}^T[\mathcal{L}(t, k)]$

- $\mathcal{L}(t, k)=\mathbb{E}_{\boldsymbol{h}_t^k}\left[\delta(k)\left\|s_{\boldsymbol{\theta}}\left(\boldsymbol{h}_t^k, \boldsymbol{h}_{t-1}, k\right)-\nabla_{\boldsymbol{h}_t} \log q_{0 k}\left(\boldsymbol{h}_t \mid \boldsymbol{h}_t^0\right)\right\|^2\right] $.

<br>

### Training

(1) En & De

- pre-trained using the objective $\mathcal{L}_{E D}$. 

- ( can also be trained simultaneously with the network $s_{\boldsymbol{\theta}}$, but Lim et al. (2023) showed that the pre-training generally led to better performance )

<br>

(2) Score-matching network

- Hidden states are firstly obtained through inputting the entire TS $\boldsymbol{x}_{1: T}^0$ into the encoder
- Objective function $\mathcal{L}_{\text {Score }}$. 

<br>

### Sampling

- achieved by sampling hidden states & applying the decoder
  - analogous to solving the solutions to the time-reverse SDE.

- SOTA sampling quality and diversity

<br>

Limitation: more computationally expensive than GANs.