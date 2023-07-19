# UnDiMix: Hard Negative Sampling Strategies for Contrastive Representation Learning



# Abstract

challenges in CL

=  selection of appropriate hard negative examples

<br>

Random sampling or importance sampling methods based on feature similarity often lead to suboptimal performance. 

### UnDiMix

- a hard negative sampling strategy 

- considers anchor similarity, model uncertainty, and diversity

<br>

# 1. Introduction

Impact of negative sampling strategies. 

- increasing the number of negative samples results in learning better representations
- but a few of the hardest negative samples tend to have the same label as the anchor, which hampers the learning process

$\rightarrow$ selecting informative hard negative examples is crucial

<br>

### Negative selection mechanisms

most CL : either

- uniformly sample negatives
- compute importance scores based on feature similarity / uncertainty 

$\rightarrow$ no clear notion of "informativeness"

<br>

### Challenges of Hard Negative example selection

![image-20230717164029934](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717164029934.png)

<br>

Efficient sampling should avoid FN examples

= ones that are from the same class as the anchor

<br>

<br>

Methods that only consider similarity with anchor

- tend to select same-class negatives,

<br>

Diverse negative examples

- assist in learning global representations of data distribution.

![image-20230717164406942](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717164406942.png).

<br>

***Beneficial to consider (1) diversity, in addition to (2) uncertainty and (3) anchor similarity***

<br>

# UnDiMix ( Uncertainty and Diversity Mixing )

combines importance scores that capture 

- (1) model uncertainty
- (2) diversity
- (3) anchor similarity

to select informative negative examples.

<br>

Details : utilizes ... 

- uncertainty to select influential examples
- pairwise distance among negatives to select diverse examples in a computationally efficient manner. 

<br>

Evaluation : various benchmark datasets 

- involving visual, text, and graph data

<br>

# 2. Related Works

## Hard Negative Sampling

AdCo 

- maintains a **separate global set for negative examples** that is actively updated using the contrastive loss gradients with respect to each negative example.
- However, the set of negative examples **remains the same for all the anchors**

<br>

MMCL

- formulate the **contrastive loss function as an SVM objective**
- utilize the **support vectors as hard negatives**
- THUS resorting to approximations to solve a **computationally expensive** quadratic equation for each anchor. 

<br>

HCL, Mochi, FNC

- rely on **feature-based anchor similarity** when selecting negative examples

- However, considering only anchor similarity results in **assigning more importance to the same-class negatives**

  ( = most likely false negatives )

<br>

Motivated by Mixup [52]

- Some methods create synthetic examples either by ...
  - interpolating instances at an image/pixel or latent representation level
  - interpolating virtual labels

<br>

Other methods use either ...

- texture-based and patch-based non-semantic augmentation techniques
- asynchronously-updated approximate nearest neighbor index of corpus

<br>

Use of two hyperparameters to create a ring of negatives around the anchor

<br>

Ma et al. selects negative examples with high model uncertainty

<br>

### UnDiMix

***Jointly consider anchor similarity, uncertainty, and diversity***. 

- [UNCERTAINTY] design an **auxiliary pseudo-labeling task** & leverage the gradients of the last layer as a model-based uncertainty measure. 

- [ANCHOR SIMILARITY] compute **gradient similarity** in order to assign more importance to negative samples **that are more influential to the anchor**

- [DIVERSITY] to capture other equally useful negative sampling properties

<br>

# 3. Method

## (1) Three properties

- P1) Anchor Vicinity Property

  ***Hard negative examples share similarities with the anchor***

  - feature representations of the hardest negative examples lie close to the anchor in the embedding space. 

- P2) Influence Property

  ***The selected negative examples should be influential and more informative than other examples.***

  - (SL) examples that are closer to the class decision boundary are considered the most informative. 

  - (SSL) label is not available

    $\rightarrow$ auxiliary pseudo-labeling task

    - involves predicting the alignment of differently augmented views of the same anchor

- P3) Diversity Property

  ***Informative negative examples should also be diverse***

<br>

![image-20230717165649762](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717165649762.png)

<br>

## (2) UnDiMix Description

Selects hard negative samples based on **importance scores**

- computed by combining the three components

<br>

(1) a ***feature-based*** component ( P1 )

- utilizes the **instance similarity** property in the feature space to select informative negative samples that satisfy P1,

(2) a ***model-based*** component  ( P2 )

- calculates the **influence of each negative example on the anchor** by using the model gradients with respect to each negative sample as a measure of uncertainty. 
- This component approximates P2 by calculating the gradient similarity with respect to the anchor.

(3) a ***density-based*** component ( P3 )

- promotes the selection of diverse negative samples by assigning more weight to negative examples **that are further away from other negative examples in the batch** and satisfies P3

<br>

### P1 ( Anchor Vicinity )

utilize instance similarity in the embedding space (= inner product )

![image-20230717170239110](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717170239110.png)

<br>

### P2 ( Influence )

**gradient-based uncertainty metric**

(1) pseudo-labeling

- implicitly minimizes entropy

(2) gradient-based uncertainty

- smaller gradient norm corresponds to higher model confidence

are established in semi-supervised and active learning

<br>

![image-20230717170445700](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717170445700.png)

### P3 (Diversity)

compute the average distance of a negative example from all other negatives

![image-20230717170632252](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717170632252.png)

<br>

### Importance Score

![image-20230717170643564](/Users/seunghan96/Library/Application Support/typora-user-images/image-20230717170643564.png)

