---
title: A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities
categories: [TAB]
tags: []
excerpt: arxiv 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>

# A Closer Look at TabPFN v2: Understanding Its Strengths and Extending Its Capabilities

```
Ye, Han-Jia, Si-Yang Liu, and Wei-Lun Chao. "A closer look at tabpfn v2: Strength, limitation, and extension." arXiv preprint arXiv:2502.17361 (2025).
```

arxiv: https://arxiv.org/pdf/2502.17361

<br>

# Contents

1. 

<br>

# Abstract

TabPFN v2

- Advancement in **"tabular foundation models"**
- Unprecedented **"ICL performance"** across diverse downstream datasets

<br>

This paper = ***Closer look at TabPFN v2***

- How it effectively handles **"heterogeneity"** and achieves high predictive accuracy
- How its limitations in **"high-dimensional, many-category, and large-scale tasks"** can be mitigated

<br>

Findings

- TabPFN v2 can infer attribute relationships even when provided with randomized attribute token inputs, ***eliminating the need to explicitly learn dataset-specific attribute embeddings*** to address heterogeneity. 
- TabPFN v2 can be ***transformed into a feature extractor***, revealing its ability to construct a highly separable feature space for accurate predictions
- TabPFN v2’s limitations can be addressed through ***a test-time divide-and-conquer strategy***, enabling scalable inference without requiring re-training. 

<br>

# 1. Introduction

### Tabular Prior-Fitted Network v2 (TabPFN v2)

- Model) Transformer 
- Pretraining) Synthetic datasets
- ICL) Applied to diverse downstream tasks without additional tuning. 
- Details
  - Input = Labeled training set and an unlabeled test instance
  - Output = Test label
- Experiments: SoTA on both classification and regression

<br>

This paper = understand the mechanisms behind its success

- (1) How it effectively handles dataset heterogeneity and achieves high predictive accuracy
- (2) How to overcome its current limitations
  - Suggested data regime: of no more than 10,000 samples, 500 dimensions, and 10 classes

<br>

Major insights

- (1) TabPFN v2 ***internalizes attribute token learning*** to handle data **heterogeneity**

  - Raw input = $$d$$ attributes

  - Embedding: Each attribute Into fixed-dimensional tokens

  - Transformer: Handle variability in $$d$$

  - Prior works vs. TabPFN v2

    - Prior works : Rely on known attribute semantics (e.g., word vectors) or learn dataset-specific attribute tokens

    - TabPFN v2:  Employs randomized attribute tokens (= resampled at each inference)

      $$\rightarrow$$ Allows TabPFN v2 to be directly applied to new downstream datasets with varying dimensionalities and attribute meanings without additional tuning

  - Question) How does it still make accurate predictions? 
  - Findings) Regardless of the randomness, TabPFN v2 can consistently infer attribute relationships through in-context learning, essentially integrating attribute token learning into the inference itself. 
  - Summary) TabPFN v2 unifies representation learning and prediction within a single forward pass.

- (2) TabPFN v2 can be **repurposed as a feature extractor** for downstream tasks.
  - Produces instance-level feature representations that are highly discriminative
  - Reveal that TabPFN v2 effectively maps tabular instances into a nearly linearly separable embedding space. 
    - Remarkably, training a linear model on these features yields accuracy comparable to that of TabPFN v2’s in-context learner, highlighting its potential as a powerful feature encoder

- (3) **Test-time divide-and-conquer** effectively mitigates TabPFN v2’s limitations.
  - TabPFN v2 faces challenges when applied to high-dimensional, many-category, or large-scale datasets. 
  - These limitations can be effectively addressed through carefully designed post-hoc divide-and-conquer strategies, reminiscent of test-time scaling techniques developed for LLMs

<br>

Remark

- In-depth investigation into TabPFN v2
- Contribution lies in the novel analysis and principled extension of TabPFN v2

<br>

# 2. Related Work

## (1) Learning with Tabular Data

Skip

<br>

## (2) Tabular Foundation Models

Challenges of tabular data in foundation models = Heterogeneity

- Variations in attribute spaces, dimensionalities, and label distributions

Solution: 

- (1) Leverage the semantic meanings of attributes
  - e.g., “Age is 25, height is 165, ...” & apply LLM

- (2) Improve transferability by pre-computing attribute tokens based on semantic embeddings 
- (3) TabPFN family
- (4) Meta-learning 
  - Generate model weights tailored for downstream tabular tasks with limited data
- (5) Others:
  - Rely on lightweight fine-tuning to adapt to variations in attribute and label spaces 

<br>

TabPFN family

- Leverages the ICL capabilities of transformers to directly predict labels by contextualizing test instances among training examples. 
- Inspired subsequent pre-trained tabular models such as [49, 15, 58]
- TabPFN v1 
  - Pads attribute vectors to a fixed dimension
- TabPFN v2
  - Introduces a specialized attribute tokenizer to handle heterogeneous input spaces. 

<br>

## (3) Variants of TabPFN

TabPFN’s success = Stems from its pre-training on ***massive synthetic datasets***

$$\rightarrow$$ Strong ICL performance on small-scale classification tasks

<br>

Improving scalability of TabPFNs

- By addressing TabPFN’s sensitivity to context size [17, 74]. 
- Further strategies to enhance downstream performance include context adaptation with nearest neighbor [66], partial fine-tuning [18, 47], pre-training on real-world datasets [49], scalable ensemble [47], and more powerful and efficient pre-training on synthetic data [58]. 
- Most of these variants remain restricted to classification tasks due to limitations in TabPFN v1.

<br>

TabPFN v2 [29] 

- Extends TabPFN to support regression tasks and accommodate larger context sizes. 

<br>

This paper = Comprehensive evaluation of TabPFN v2

<br>

# 3. Background

## (1) Learning with a single tabular dataset
Notation

-  $$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$$: dataset
- $$N$$  training examples = $$N$$ rows
- $$x_i$$: $$d$$  features or attributes (i.e., columns in the table)
  - $$d$$ varies across datasets. 
- $$y_i $$: label
  - Belongs to  $$\{1, \dots, C\}$$  for a classification task
  - Numerical value for a regression task

<r>

Assumption = All attributes of an instance are numerical (continuous)

- Categorical (discrete) attributes: Transformed using ordinal or one-hot encoding

<br>

## (2) TabPFN

- Transformer + ICL
- Training and test instances are first **zero-padded** to a fixed dimension $$k^{'}$$  (e.g., 100)
- $$x_i$$  and  $$y_i$$  are linearly projected to  $$\tilde{x}_i \in \mathbb{R}^k$$  and $$ \tilde{y}_i \in \mathbb{R}^k$$
- Input & Output 
  - Input = Labeled training set and an unlabeled test instance
  - Output =  Test label
- Input ( = Prompt / Context )
  - $$\mathcal{C} = \{ \tilde{x}_1 + \tilde{y}_1, \dots, \tilde{x}_N + \tilde{y}_N, \tilde{x}^* \} \in \mathbb{R}^{(N+1) \times k}$$.
  - Consisting of  $$N+1$$ tokens
  - Processed by multiple transformer layer with variable length inputs ( = variable $$N$$ )
- Output token 
  - Passed through a MLP

<br>

## (3) TabPFN v2

Several key modifications. 

- Each of the $$d$$ attributes in $$\mathcal{D}$$ is embedded into a  $$k$$-dimensional space

- (1) Random perturbations added to differentiate attributes

- (2) Modification to input?

  - Training instance $$\tilde{x}_i$$

    - $$d+1$$ tokens with dimension $$k$$ .... with Label embedding $$\tilde{y}_i \in \mathbb{R}^k$$

  - Test instance $$x^*$$

    - Label is unknown ... Employ dummy label to generate the label embedding $$\tilde{y}^*$$

      (e.g., the average label of the training set) 

  - Shape of input = $$ (N + 1) \times (d + 1) \times k $$

- (3) Two types of self-attention

  - Over samples (among the \( $$N + 1$$ \) instances)
  - Over attributes (among the \( $$d + 1$$ \) dimensions)

- Output token ( = test instance’s dummy label $$\tilde{y}^{*}$$ )

  - Extracted and mapped to a ...
    - (1) $$C$$-class logit for classification
    - (2) Single-value logit for regression.

![figure2](/assets/img/tab/img105.png)

<br>

### Pretraining

Pre-trained on diverse synthetic datasets generated using **structural causal models (SCMs)**

<br>

## (4) Remark

(Note that many technical details were omitted from the main paper)

Example) Use of randomized tokens was documented in the supplementary material and code

<br>

# 4. Comprehensive Evaluation of TabPFN v2

Extend TabPFN v2’s evaluation (beyond the original set of datasets) to over 300!

- Much broader range of domains, attributes, scales, dimensionalities, and tasks

<br>

<br>

## (1) Setups

Benchmark from [76]

- 120 binary classification
- 80 multi-class classification
- 100 regression

<br>

Issues

- Mislabeled data
- Redundancies from overlapping dataset

<br>

Split 

- Out of the 300 datasets,
- 27 belong to the validation set used for checkpoint selection in TabPFN v2

$$\rightarrow$$ To avoid evaluation bias, we exclude these datasets and report results on the remaining 273 datasets.

<br>

Each dataset is randomly split into training, validation, and test partitions in a 64%/16%/20% ratio!

<br>

TabPFN v2 

- Predicts test set labels directly using ICL

  (w/o any additional parameter or hyperparameter tuning)

<br>

Metrics

- 15 random seeds ( report the average performance )
- Classification  = Acc.
  - with more than 10 classes: Error-Correcting Output Codes (ECOC) 
- Regression = RMSE

<br>

## (2) Strengths of TabPFN v2

- Baseline: 26 representative tabular methods

- Test: Statistical significance test (Wilcoxon-Holm test) .. Figure 1 (right)

- Results: Consistently outperforms both ..

  - (1) Tree-based methods

  - (2) Deep tabular models

<br>

## (3) Limitations of TabPFN v2

Above evaluation = Focuses on ***small- to medium-scale datasets***

- Fewer than **10,000 examples**

<br>

**Computational complexity** of transformers

$$\rightarrow$$ Constrains TabPFN v2's ability to scale effectively to datasets with larger sample sizes or higher dimensionality!

<br>

Large scale datasets: Conduct additional evaluations on ...

- (1) 18 high-dimensional datasets with $$d \geq 2000$$
- (2) 18 large-scale datasets where $$N \times d > 1,000,000$$

<br>

(1) High-dimensional datasets

- Follow the same protocol as before

(2) Large-scale datasets

- Due to the prohibitive cost of hyperparameter tuning...

  $$\rightarrow$$ Default hyperparameters are used for all methods

<br>

![figure2](/assets/img/tab/img106.png)

<br>

Results: ***Performance degrades on both large-scale and high-dimensional datasets***

(1) High-dimensional datasets

- Ranks below both CatBoost and RealMLP

(2) Large-scale datasets

- Falls behind the simple Logistic Regression (LR) model. 

(3) More than 10 categories

- ECOC strategy currently used by TabPFN v2 appears ineffective in achieving high accuracy

- Pre-trained exclusively on small- to medium-scale synthetic datasets with fewer than 10 categories,

  $$\rightarrow$$ Leading to a mismatch when applied to larger or more complex real-world data!

<br>

# 5. How Does TabPFN v2 Effectively Handle Data Heterogenity

- Examine the mechanisms behind its strengths
- Methods to overcome its limitations

<br>

## (1) Diving into TabPFN v2's Mechanisms for Heterogeneous Input

### a) Revisiting the problem

Robust tabular foundation model 

$$\rightarrow$$ Must handle heterogeneity effectively

$$\rightarrow$$ Enable to learn from diverse pre-training datasets & transfer its capabilities to new downstream tasks

<br>

### b) Tokenization as a feasible solution

Token-based methods

- (From) $$d$$-dimensional instance \( $$x \in \mathbb{R}^d$$ \) 

- (To) Set of $$d$$ fixed-dimensional tokens (each of dimension  $$k$$)

  - One token per attribute

- Enables the use of transformer architectures

  ( & Accommodate variability in $$d$$ across datasets )

<br>

Embedding

- Embed each attribute into a shared $$k$$-dim space
- (Previous works) Either uses ..
  - (1) pre-defined semantic embeddings (e.g., word vectors of attribute names)
  - (2) Dataset-specific embeddings
- Notation: 
  -  $$d$$ attribute-specific tokens \( $$[r_1, \dots, r_d] \in \mathbb{R}^{d \times k}$$ \) .
  - (Before) $$x_i \in \mathbb{R}^d$$
  - (After) $$[x_i^1 \cdot r_1, \dots, x_i^d \cdot r_d] \in \mathbb{R}^{d \times k}$$.
    - where $$x_i^j$$ denotes the $$j$$-th element of $$x_j$$

<br>

### c) Difficulty in direct generalization

Aforementioned methods: Face a notable challenge when applied to downstream tasks!!

$$\rightarrow$$ Attribute names or semantics are not always accessible!!

<br>

### d) TabPFN v2's mechanisms

Also builds on (prior) token-based methods  

- Represent each instance $$x_i$$ as a sequence of tokens

<br>

Rather than assigning a deterministic token to each attribute ...

$$\rightarrow$$ TabPFN v2 samples **random tokens** at inference time!

<br>

***Adding random tokens?***

- Learns a shared vector $$u \in \mathbb{R}^k$$ that lifts each element of $$x_i$$ into a $$k$$-dimensional space
- To distinguish attributes, TabPFN v2 adds a random perturbation to each one
- Details
  - $$j$$-th attribute (i.e., $$x_i^j$$), the representation becomes ...
  - $$x_i^j \cdot u + r_j$$  where $$ r_j = W p_j$$
    - $$p_j \in \mathbb{R}^{k'}$$: Randomly generated vector
    - $$W \in \mathbb{R}^{k \times k'}$$: Learned projection matrix that conditions the perturbation
  - Full instance $$x_i$$
    - $$[x_i^1 \cdot u + r_1, \dots, x_i^d \cdot u + r_d, \tilde{y}_i] \in \mathbb{R}^{k \times (d+1)}$$.
      - where the last token $$\tilde{y}_i $$ encodes the label information

<br>

## (2) TabPFN v2 Internalizes Attribute Token Learning

***"Randomized"*** tokenization scheme 

$$\rightarrow$$ Eliminates the need to define attribute- or dataset-specific tokens across tasks!

$$\rightarrow$$ Enabling direct application of the pre-trained model to any dataset

<br>

(At first glance)

- May appear to disregard the valuable semantic meaning of attributes. 

(However)

- Show that through ICL, TabPFN v2 can consistently infer relationships among attributes within a dataset

<br>

Analysis

- From three perspectives
- Using two representative downstream datasets

![figure2](/assets/img/tab/img107.png)

<br>

### Remark

TabPFN v2 can reliably infer meaningful attribute relationships through ICL

Although ***input embeddings are randomized***...

$$\rightarrow$$ Consistently **differentiate attributes across instances!**

<br>

# 6. TabPFN v2 can be Transformed into an Effective Feature Encoder

In Section 5, we show that TabPFN v2’s in-context learning process infers meaningful attribute

relationships. Here, we examine whether TabPFN v2 also produces separable instance representations.



## (1) Naive Feature Extraction Fails

As shown in Figure~\ref{fig:1} (left), TabPFN v2 makes predictions based on the output token corresponding to the (dummy) label embedding \( \tilde{y}^* \) of the test instance. This output token can thus be interpreted as the instance embedding for the test example. A natural extension to obtain embeddings for the training instances is to extract the output tokens corresponding to the training label embeddings \( \{ \tilde{y}_i \}_{i=1}^N \).

However, as shown in Figure~\ref{fig:4} (b), this naive approach leads to surprisingly discrepant feature distributions between training (darker cross) and test (lighter circle) examples. As a result, a linear classifier trained on these embeddings performs poorly on the test set. We attribute this discrepancy to the distinct roles of labeled training data and unlabeled test data in TabPFN v2’s in-context learning process. Specifically, the label embeddings for the training instances are derived from true labels, whereas those for the test instances rely on dummy labels. This mismatch renders the resulting output embeddings \emph{non-comparable} between training and test instances.

<br>

## (2) Leave-on-fold-out Feature Extraction

![figure2](/assets/img/tab/img108.png)

To address this challenge, we propose a leave-one-fold-out strategy that enables the extraction of comparable embeddings for training and test data. In the TabPFN v2 framework, we treat examples with true labels as the support set \( \mathcal{S} \), and those with dummy labels as the query set \( \mathcal{Q} \). Under the standard configuration, \( \mathcal{S} \) corresponds to the labeled training set and \( \mathcal{Q} \) to the unlabeled test instances. To extract comparable embeddings for the training examples, they must also be included in \( \mathcal{Q} \) with dummy label embeddings. This, however, creates a dilemma: effective in-context learning relies on maximizing the size of \( \mathcal{S} \) to ensure sufficient knowledge transfer to \( \mathcal{Q} \). Including training examples in \( \mathcal{Q} \) thus competes with the need to keep \( \mathcal{S} \) as large as possible.

To overcome this dilemma, we partition the training set into multiple folds, e.g., 10. In each round, one fold serves as \( \mathcal{Q} \)—with dummy labels used for embedding extraction—while the remaining folds form \( \mathcal{S} \) with true labels. This setup preserves sufficient label supervision in \( \mathcal{S} \) while enabling the extraction of embeddings for training instances in \( \mathcal{Q} \). Results in Figure~\ref{fig:4} (c)–(f) show that embeddings extracted by this strategy (with 10 folds) more faithfully capture dataset structure. We observe that TabPFN v2 simplifies the original tabular data distributions, transforming datasets into nearly linearly separable embedding spaces—especially after intermediate transformer layers.

<br>

## (3) Validation of Embedding Quality

![figure2](/assets/img/tab/img109.png)

To validate the quality of the extracted embeddings, we train a logistic regression on embeddings

derived from the training set and evaluate it on test set embeddings.

The average rank across 29 classification

datasets from the tiny benchmark2 in [76]

is reported in Table 2. Remarkably, training

a simple linear classifier on the extracted

embeddings achieves performance compa-

rable to that of TabPFN v2’s in-context

learner. Furthermore, concatenating em-

beddings from multiple layers (e.g., both

output and intermediate representations)

can sometimes lead to even better results.

These findings underscore TabPFN v2’s potential as a strong and versatile 

<br>

# 7. Improving TabPFN v2 via Test-Time Divide-and-Conquer

This section addresses the limitations discussed in Section 4.3, aiming to extend TabPFN v2’s

applicability beyond the boundaries. Specifically, we propose post-hoc divide-and-conquer strategies

inspired by Chain-of-Thought (CoT) prompting [71], which decompose challenging tasks into simpler

subtasks that TabPFN v2 can effectively handle.

<br>

## (1) High Dimension Datasets

High-dimensional datasets~\cite{datasetbench2022} present a unique challenge due to the quadratic complexity of TabPFN v2 with respect to the number of dimensions. To mitigate this, we propose subsampling the feature space into smaller subsets, processing each subset independently, and combining the predictions in an ensemble (bagging) fashion, similar to random forests~\cite{breiman2001random}.

In detail, we iteratively sample \( m \) subsets, each containing \( d' < d \) randomly selected attributes. For each subset, we leverage TabPFN v2’s ability to handle lower-dimensional data to obtain predictions. We denote this divide-and-conquer and then ensemble strategy as TabPFN v2*, which aggregates outputs using averaging (for regression) or majority voting (for classification).

Figure~\ref{fig:5} (left) summarizes the results on 18 high-dimensional \textit{classification} datasets. A variant that utilizes PCA to reduce the dimensionality, together with bagging, resolves the dimensionality issue to some extent. TabPFN v2* with \( d' = 500 \) and \( m = \lceil d / d' \rceil \) significantly increases the mean accuracy (to the highest), effectively extending TabPFN v2’s scalability to datasets with \( d \geq 2000 \).

<br>

## (2) Multi-Class Problems with More Than 10 Classes

To extend TabPFN v2 to tasks with more than 10 categories, we propose a decimal encoding approach that decomposes multi-class problems into multiple 10-class subproblems, ensuring compatibility with TabPFN v2’s constraints.

For a task with \( C > 10 \) classes, we encode each label \( y \in [C] \) as a \( t \)-digit decimal representation, where \( t = \lceil \log_{10} C \rceil \). For each digit position \( j \in \{1, \dots, t\} \), we train a separate TabPFN v2 model \( f_j \) to predict the \( j \)-th digit. During inference, the predicted digits are reconstructed to obtain the final class label. This strategy is also developed in~\cite{tabpfn_decoding2024}, and we denote it as TabPFN v2-DPT. As the decimal encoding inherently introduces artificial correlations among classes—classes that share the same digit at a given position are grouped together, even if they are semantically unrelated—to mitigate this effect, our TabPFN v2* randomly permutes the class-to-digit mapping \( \sqrt{C} \) times, leading to different groupings in each run, and the prediction results are ensembles to improve robustness.

As shown in Figure~\ref{fig:5} (middle), this approach achieves the second-best mean accuracy on 12 datasets with more than 10 classes while preserving computational efficiency.

<br>

## (3) Large-Scale Datasets

![figure2](/assets/img/tab/img110.png)

For large-scale datasets, we randomly sample 10,000 training examples from the full training set

as the support set and treat the remaining training examples and test instances as the query set. We

extract their embeddings to form a new tabular dataset, on which a logistic regression classifier is

trained to make predictions on the test set embeddings. This process is repeated four times, and the

final predictions are aggregated. We denote this version as TabPFN v2∗-SQ.

We also investigate integrating TabPFN v2 with decision trees to handle large-scale tasks. We note

that a similar strategy was mentioned in [29] to handle within-dataset heterogeneity for a drastically

different purpose. Specifically, we sample 32 subsets from the original training set, each containing

60% of the original data (sampled without replacement). For each subset, we first train a shallow

decision tree by setting the minimum number of samples required to split an internal node to 10,000.

The decision tree partitions the training set into smaller, more manageable subsets. During inference,

a test instance is first passed through each of the shallow decision tree to a leaf node and then predicted

by the corresponding TabPFN v2 model. The predictions from all 32 models are aggregated. We

denote this extension as TabPFN v2∗-DF.

Figure 5 (right) shows the average rank results, including TabPFN v2∗-SQ and TabPFN v2∗-DF

alongside variants using bagging and KMeans-based sampling. We observe that all variants improve

upon the vanilla TabPFN v2 on large-scale datasets, with TabPFN v2∗-DF and TabPFN v2∗-SQ

achieving the most significant improvement.



# 8. Conclusion

We present a timely investigation into TabPFN v2, a groundbreaking foundation model for tabular

tasks. Our analysis uncovers the core mechanism behind TabPFN v2’s strong performance across

heterogeneous tabular datasets: it can infer attribute relationships on-the-fly—even from randomly

initialized token inputs—without relying on pre-defined semantics or learning dataset-specific repre-

sentations. We also demonstrate that TabPFN v2 can be repurposed as a powerful feature encoder,

enabling broader applications such as data visualization and diagnostic analysis. To address its

limitations in more complex data regimes, we introduce post-hoc divide-and-conquer strategies that

extend TabPFN v2’s utility without requiring model re-training. Together, these contributions offer

fresh insights into advancing the development and application of foundation models for tabular data.



