## [ Paper review 23 ]

# Unbiased Implicit Variational Inference

### ( Michalis K. Titsias, Francisco J. R. Ruiz, 2019 )



## [ Contents ]



# 1. Key Idea

Gradient of ELBO : $\nabla_{\theta} \mathcal{L}(\theta)=\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta}\left(\log p\left(x, f_{\theta}(\varepsilon)\right)-\log q_{\theta}\left(f_{\theta}(\varepsilon)\right)\right)\right]$

- (1) model term : (with MC approximation)

  $\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta} \log p\left(x, f_{\theta}(\varepsilon)\right)\right] \approx \frac{1}{S} \sum_{s=1}^{S} \nabla_{\theta} \log p\left(x, f_{\theta}\left(\varepsilon^{(s)}\right)\right), \quad \varepsilon^{(s)} \sim q(\varepsilon)$

- (2) entropy term : 

  $\nabla_{\theta} \log q_{\theta}\left(f_{\theta}(\varepsilon)\right)=\nabla_{z} \log q_{\theta}(z) \times \nabla_{\theta} f_{\theta}(\varepsilon)+\underbrace{\left.\nabla_{\theta} \log q_{\theta}(z)\right|_{z=f_{\theta}(\varepsilon)}}_{=0(\text { in expectation })}=\nabla_{z} \log q_{\theta}(z) \times \nabla_{\theta} f_{\theta}(\varepsilon)$

  but $\nabla_{z} \log q_{\theta}(z) $ is not available !



use "UNBIASED MC estimator" of $\nabla_{z} \log q_{\theta}(z) $

- using density ratio (X)
- using lower bound of ELBO (X)
- directly optimize ELBO (O)



Key idea: as a form of... $\nabla_{z} \log q_{\theta}(z)=\mathbb{E}_{\text {distrib }(\cdot)}[\text { function }(z, \cdot)]$



# 2. UIVI

![image-20201218140003285](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201218140003285.png)



$\begin{aligned}\nabla_{z} \log q_{\theta}(z)&=\mathbb{E}_{q_{\theta}\left(\varepsilon^{\prime} \mid z\right)}\left[\nabla_{z} \log q_{\theta}\left(z \mid \varepsilon^{\prime}\right)\right]\\\\&
\approx \nabla_{z} \log q_{\theta}\left(z \mid \varepsilon^{\prime}\right), \quad \varepsilon^{\prime} \sim q_{\theta}\left(\varepsilon^{\prime} \mid z\right) \end{aligned}$   ( use MC estimation)



gradient of ELBO

$\begin{aligned}\nabla_{\theta} \mathcal{L}(\theta)&=\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta}\left(\log p\left(x, f_{\theta}(\varepsilon)\right)-\log q_{\theta}\left(f_{\theta}(\varepsilon)\right)\right)\right] \\
&=\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta} \log p\left(x, f_{\theta}(\varepsilon)\right)\right]-\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta} \log q_{\theta}\left(f_{\theta}(\varepsilon)\right)\right]\\&= \mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta} \log p\left(x, f_{\theta}(\varepsilon)\right)\right]-\mathbb{E}_{q(\varepsilon)}\left[\nabla_{z} \log q_{\theta}(z) \times \nabla_{\theta} f_{\theta}(\varepsilon)\right]\\
&\approx\mathbb{E}_{q(\varepsilon)}\left[\nabla_{\theta} \log p\left(x, f_{\theta}(\varepsilon)\right)\right]-\mathbb{E}_{q(\varepsilon)}\left[\nabla_{z} \log q_{\theta}\left(z \mid \varepsilon^{\prime}\right)\times \nabla_{\theta} f_{\theta}(\varepsilon)\right]\quad \varepsilon^{\prime} \sim q_{\theta}\left(\varepsilon^{\prime} \mid z\right)\\\end{aligned}
$

$\nabla_{\theta} \mathcal{L}(\theta)=\mathbb{E}_{q(\varepsilon) q(u)}\left[\left.\nabla_{z}\left(\log p(x, z)-\log q_{\theta}(z)\right)\right|_{z=h_{\theta}(u ; \varepsilon)} \times \nabla_{\theta} h_{\theta}(u ; \varepsilon)\right]$



## 2.1 Full Algorithm

Estimate the gradient based on samples :

- 1) sample$\epsilon \sim q(\epsilon)$ , $u\sim q(u)$ ( standard Gaussian )
- 2) set $z=h_{\theta}(\varepsilon ; u)=\mu_{\theta}(\varepsilon)+\Sigma_{\theta}(\varepsilon)^{1 / 2} u$
- 3) evaluate $\nabla_{z} \log p(x, z)$ and $\nabla_{\theta} h_{\theta}(u ; \varepsilon)$
- 4) sample $\varepsilon^{\prime} \sim q_{\theta}\left(\varepsilon^{\prime} \mid z\right)$
- 5) approximate $\nabla_{z} \log q_{\theta}(z) \approx \nabla_{z} \log q_{\theta}\left(z \mid \varepsilon^{\prime}\right)$

( How to do step 4 \& step 5? )



## 2.2 Reverse Conditional

in "step 4) sample $\varepsilon^{\prime} \sim q_{\theta}\left(\varepsilon^{\prime} \mid z\right)$"...

- conditional : $q_{\theta}(z\mid \epsilon)$

- reverse conditional : $q_{\theta}\left(z \mid \varepsilon^{\prime}\right)$

  

sample from reverse conditional using HMC

- $q\left(\varepsilon^{\prime} \mid z\right) \propto q\left(\varepsilon^{\prime}\right) q_{\theta}\left(z \mid \varepsilon^{\prime}\right)$  ( unnormalized density )

- but HMC is slow

  Thus, start with a GOOD STARTING(INITIAL) POINT , which is $\epsilon$

[proof]

$(\varepsilon, z) \sim q_{\theta}(\varepsilon, z)=q(\varepsilon) q_{\theta}(z \mid \varepsilon)=q_{\theta}(z) q_{\theta}(\varepsilon \mid z)$

Thus, $\varepsilon$ is a sample from $q_{\theta}(\varepsilon \mid z)$

To accelerate sampling $\varepsilon^{\prime} \sim q\left(\varepsilon^{\prime} \mid z\right),$ initialize $\mathrm{HMC}$ at $\varepsilon$

( after few iterations, the correlation between $\epsilon$ and $\epsilon^{'}$ will decrease! )



# 3. SIVI vs UIVI

![image-20201218143020480](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201218143020480.png)



# 4. VAE Experiments with UIVI

Model  :

- $p_{\phi}(x, z)=\prod_{n} p\left(z_{n}\right) p_{\phi}\left(x_{n} \mid z_{n}\right)$

Amortized variational distribution :

-  $q_{\theta}\left(z_{n} \mid x_{n}\right)=\int q\left(\varepsilon_{n}\right) q_{\theta}\left(z_{n} \mid \varepsilon_{n}, x_{n}\right) d \varepsilon_{n}$

Goal: 

- Find model parameters $\phi$ and variational parameters $\theta$

![image-20201218143135836](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201218143135836.png)

