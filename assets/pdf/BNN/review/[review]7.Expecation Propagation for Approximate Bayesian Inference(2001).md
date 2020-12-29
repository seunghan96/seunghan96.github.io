## [ Paper review 7 ]

# Expecation Propagation for Approximate Bayesian Inference

### ( Thomas P Minka, 2001 )



## [ Contents ]

0. Abstract
1. Introduction
2. ADF
3. Expectation Propagation



# 0. Abstract

Expectation Propagation : (1) + (2)

- (1) ADF ( Assumed-Density Filtering)
- (2) Loopy belief propagation



approximates the belief states by only retaining expectations ( such as mean, variance )

and iterates until these expectations are consistent!



Lower computational cost than ...

- Laplace's method, variational bayes, MC



# 1. introduction

Bayesian Inference : require large computational expense

Expectation Propagation (EP):

- "one-pass, sequential" method for computing an approximate posterior distribution

- weakness of ADF : information discarded previously may turn out to be important later!



# 2. ADF (Assumed-Density Filtering)

goal : to compute "approximate posterior"

also called..."online Bayesian Learning", "moment matching", "weak marginalization"...



applicable when we have postulated a joint pdf $p(D,x)$

( $D$ : observed , $x$ : hidden )

find out posterior over $x$ ( = $P(x \mid D)$ ) and evidence ( = $P(D)$ )



### Example

have observation from Gaussian distribution ( embedded in a sea of unrelated clutter )

( $w$ : ratio of clutter )

$\begin{aligned}
p(\mathbf{y} \mid \mathbf{x}) &=(1-w) \mathcal{N}(\mathbf{y} ; \mathbf{x}, \mathbf{I})+w \mathcal{N}(\mathbf{y} ; \mathbf{0}, 10 \mathbf{I}) \\
\mathcal{N}(\mathbf{y} ; \mathbf{m}, \mathbf{V}) &=\frac{\exp \left(-\frac{1}{2}(\mathbf{y}-\mathbf{m})^{\mathrm{T}} \mathbf{V}^{-1}(\mathbf{y}-\mathbf{m})\right)}{|2 \pi \mathbf{V}|^{1 / 2}}
\end{aligned}$



$d$ dimensional vector $x$ has Gaussian prior :

-  $p(\mathbf{x}) \sim \mathcal{N}\left(\mathbf{0}, 10 \mathbf{I}_{d}\right)$



joint pdf of $x$ and $D$ (  where $D=\left\{\mathbf{y}_{1}, \ldots, \mathbf{y}_{n}\right\}$   ) : 

- $p(D, \mathbf{x})=p(\mathbf{x}) \prod_{i} p\left(\mathbf{y}_{i} \mid \mathbf{x}\right)$  



How does it work?

[STEP 1] to apply ADF, re-express the joint-pdf as below

- $p(D, \mathbf{x})=\prod_{i} t_{i}(\mathbf{x})$,  where $t_{0}(\mathbf{x})=p(\mathbf{x})$ and $t_{i}(\mathbf{x})=p\left(\mathbf{y}_{i} \mid \mathbf{x}\right) .$



[STEP 2] choose an approximating family

- $q(\mathbf{x}) \sim \mathcal{N}\left(\mathbf{m}_{x}, v_{x} \mathbf{I}_{d}\right)$
- ( choose a "spherical Gaussian")



[STEP 3] incorporate the terms $t_i$ into the approximate posterior

- initial $q(x) = 1$

- at each step, move from old $q^{\backslash i}(\mathbf{x})$ to a new $q(\mathbf{x})$   
- Incorporating the prior term is trivial ! $\hat{p}(\mathbf{x})=\frac{t_{i}(\mathbf{x}) q^{\backslash i}(\mathbf{x})}{\int_{\mathbf{x}} t_{i}(\mathbf{x}) q^{\backslash i}(\mathbf{x}) d \mathbf{x}}$
- ( each step produces normalizing factor. 
  in this case, $Z_{i}=(1-w) \mathcal{N}\left(\mathbf{y}_{i} ; \mathbf{m}_{x}^{\backslash i},\left(v_{x}^{\backslash i}+1\right) \mathbf{I}\right)+w \mathcal{N}\left(\mathbf{y}_{i} ; \mathbf{0}, 10 \mathbf{I}\right)$ )



[STEP 4] minimize KL-Divergence

- $D(\hat{p}(\mathbf{x}) \| q(\mathbf{x}))$
- subject to the constraint that $q(x)$ is in the approximating family



ADF Algorithm

![image-20201129170744518](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201129170744518.png)


Intuitive Interpretation

- for each data point, compute $r$ ( = probability of not being clutter )
- make a update to $x(m_z)$
- change our confidence ! ( in the estimate $v_x$ )

BUT, depends on the order in which data is processed ( because the clutter probability $r$ depends on the current estimate of $x$ )



# 3. Expectation Propagation

novel interpretation of ADF

(original ADF) treat each observation term $t_i$ exactly $\rightarrow$ approximate posterior that includes $t_i$

(new interpretation) approximate $t_i$ with $\tilde{t}_i$  $\rightarrow$ using an exact posterior with $\tilde{t}_i$



define approximation term $\tilde{t}_i$ as...

- ratio of NEW \& OLD posterior
- $\tilde{t}_{i}(\mathbf{x})=Z_{i} \frac{q(\mathbf{x})}{q^{\backslash i}(\mathbf{x})}$   ( multiplying this with OLD posterior gives $q(x)$ )

- ( still in the same family! ( exponential family) )



EP Algorithm

![image-20201129171431265](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201129171431265.png)


![image-20201129171440624](C:\Users\LSH\AppData\Roaming\Typora\typora-user-images\image-20201129171440624.png)




