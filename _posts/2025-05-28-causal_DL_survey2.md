---
title: Causal Inference Meets Deep Learning; A Comprehensive Survey - Part 2
categories: [ML, TS]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference Meets Deep Learning: A Comprehensive Survey

<br>

# 4. Standard Paradigms of Causal Intervention

## (1) Do-operator

- Represents an active intervention
  - $do(T = t)$ : Sets variable $T$ to value $t$
- Differs from conditioning!!
  - Conditioning = passively observes $T = t$ without altering the system
- Why does it differ?
  - $do(T = t)$ ***enforces a change*** $\rightarrow$ Allowing the study of $T$’s causal impact on other variables
- Core tool in causal inference frameworks like SCM

<br>

## (2) Back-door adjustment

**Back-door path**

- Path between nodes $X$ and $Y$ that begins with an **"arrow pointing to $X$"**
- i.e.,  $X \leftarrow Z \rightarrow Y$

<br>

![figure2](/assets/img/tab/img76.png)

<br>

**Back-door criterion**

- Helps identify **valid adjustment sets** to estimate causal effects

- If the set of variables $Z$ satisfies ...

  - a) Does not contain descendant nodes of $X$
  - b) Blocks every path between $X$ and $Y$ that contains a path to $X$

  $\rightarrow$ $Z$ is said to satisfy the **back-door criterion of $(X, Y)$**

- Example
  - Before) Figure 4(a)
  - Intervention) 
    - To investigate the true causal relationship
    - Intervention to eliminate the causal links of confounders to treatment method $X$
  - After) Figure 4(b)

<br>

**Back-door adjustment**

- Blocks spurious paths from $X$ to $Y$ **"by conditioning on $Z$"**

- Isolates the true causal effect from $X$ to $Y$ :)

  ( ***Prevent distortion from confounders*** )

<br>

Observational vs. Interventional

- Observational conditioning: $P(Z = z \mid Y = y)$ 
- Interventional probability: $P(Z = z | do(Y = y))$ 

(Note) **Do-operator modifies** the data distribution by actively setting $Y = y$

<br>

### Four rules for causal inference

### Rule 1)

If the variables $W$ and $Y$ are unrelated, then:

$P(Y \mid d o(X), Z, W)=P(Y \mid d o(X), Z)$.

<br>

Example) Once we determine the state of the mediator $Z$ (Smoke) ...

$\rightarrow$ The variable $W$ (Fire) becomes independent of $Y$ (Alarm).

<br>

### Rule 2)

If the variable $Z$ blocks all back-door paths between $(X, Y)$, then:

$P(Y \mid d o(X), Z)=P(Y \mid X, Z)$.

<br>

Example) If $Z$ satisfies the back-door criteria from $X$ to $Y$, 

$\rightarrow$ Then conditional on $Z, d o(X)$ is equivalent to $X$.

<br>

### Rule 3)

If there is no causal path from $X$ to $Y$, then:

$P(Y \mid d o(X))=P(Y)$.

<br>

Example) If there is no causal path from $X$ to $Y$, 

$\rightarrow$ Intervening on $X$ does not impact the probability distribution of $Y$.

<br>

### Rule 4) 

If there is no confounders between $X$ and $Y$, then:

$P(Y \mid d o(X))=P(Y \mid X)$.

<br>

Example) If there are no confounders among the variables

$\rightarrow$ The intervention does not change the probability distribution

<br>

### Formula for the corresponding back-door adjustment

$P(Y=y \mid d o(X=x))=\sum_b P(Y=y \mid X=x, Z=z) P(Z=z)$.

- where $Z$ meets the back-door criterion of ( $X, Y$ ).

