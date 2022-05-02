포기

# STFGNN ( Spatial-Temporal Fusion GNN ) for Traffic Flow Forecasting

# 0. Abstract

Background of Traffic Flow

- **complicated spatial dependencies** & **dynamical trends of temporal pattern** between different roads.

<br>

Limitations of previous works

- **limited representations** of given spatial graph structure with **incomplete adjacent connections** may restrict effective spatial-temporal dependencies learning of those models

- usually utilize **separate modules for spatial and temporal correlations

<br>

### STFGNN

data-driven method of generating “temporal graph” is proposed

- effectively learn **hidden spatial-temporal dependencies** by a novel **fusion** operation of various spatial and temporal graphs, 

- by integrating **fusion graph module** & novel **gated convolution module** into a unified layer...

  $\rightarrow$ can **handle long sequences**

<br>

# 1. Introduction

Limitation 1 : lack of **informative graph construction**

- ex) distant road still may have correlation!

- solution 1 : **Mask matrix(2020), Self-adaptive matrix (2019)**

  $\rightarrow$ but both lack of correlations representation ability

- solution 2 : **Temporal self-attention module(2020)**

  $\rightarrow$ face overfitting

<br>

Limitation2 : ineffective to capture dependencies between local & global corrrelations

- RNN/LSTM : time consuming & gradient vanishing/explosion
- CNN-based methods : need to stack layers for **global correlations**
- STGCN & GraphWaveNet : lose **local information**, if dilation rate increases
- **STSGCN** : novel localized spatial-temporal subgraph
  - synchronously capture local correlations
  - only designed locally ( ignore global info )

<br>

# 2. Related Works

## (1) GCN

skip

<br>

## (2) Spatial Temporal Forecasting

( mostly use GCN nowadays )

(1) DCRNN (2017) : 

- utilizes the bi-directional RW, to model spatial iformation
- captures temporal dynamics with GRU

(2) Transformer

- spatial & temporal attention modules

(3) STGCN, GraphWaveNet

- employ **GCN** on spatial domain
- employ **1-D conv** on temporal domain

<br>

## (3) Similarity of Temporal Sequences

3 categories of measuring similarities of TS

- (a) **timestep-based**
  - ex) Euclidean
- (b) **shape-based**
  - ex) DTW
- (c) **change-based**
  - ex) GMM

<br>

# 3. Preliminaries

Notation

- $\mathcal{G}=\left(V, E, A_{S G}\right)$.

  - $A_{S G} \in \mathbb{R}^{N \times N}$ : spatial adjacency matrix

- $X_{\mathcal{G}}^{(t)} \in \mathbb{R}^{N \times d}$ : spatial graph information $\mathcal{G}$ at time step $t$

  ( observed $d$ traffic features (e.g., the speed, volume) of each sensor )

<br>

Goal : 

- from previous $T$ speed observations,

  predict next $T^{'}$ traffic speed from $N$ correlated sensors

- $\left[\mathbf{X}_{\mathcal{G}}^{(t-T+1)}, \cdots, \mathbf{X}_{\mathcal{G}}^{t}\right] \stackrel{f}{\rightarrow}\left[\mathbf{X}_{\mathcal{G}}^{t+1}, \cdots, \mathbf{X}_{\mathcal{G}}^{t+T^{\prime}}\right]$.

<br>

# 4. STFGNN

![figure2](/assets/img/gnn/gnn432.png)

<br>

Consists of...

- (1) input layer ( = FC layer )
- (2) stacked **STFG** Neural Layers
- (3) output layer ( = FC layer )

<br>

Stacked **STFG** Neural Layers

- constructed by ..
  - (2-1)several **STFG Neural modules**
  - (2-2) **Gated CNN** module ( =2 parallel 1d-dilated convolution )

<br>

## (1) STFG Construction

Goal of **generating temporal graph**

- get more accurate dependency & relation than spatial graph

- incorporate **temporal graph** into a novel **STF** graph

  $\rightarrow$ make model **light weight**

  ( $\because$ fusio graph already has correlation information of each node with (1) spatial neighbours, (2) nodes with similar temporal pattern, (3) own previous/later situation )

<br>

Generating temporal graph with DTW?

- complexity : $O(n^2)$

- thus, restrict its **search length** $T$ $\rightarrow$ $O(Tn)$

  ( = ***fast-DTW*** )

  - $\omega_{k}=(i, j), \quad|i-j| \leq T$.

![figure2](/assets/img/gnn/gnn431.png)

<br>

![figure2](/assets/img/gnn/gnn433.png)

- Consists of 3 kinds of $N \times N$ matrix

  - (1) spatial graph ( given )
  - (2) temporal graph ( by alg 1 )
  - (3) temporal connectivity graph ( element is non-zero, iif previous & next time steps is the same node )

  $\rightarrow$ $A_{S T F G} \in \mathbb{R}^{3 N \times 3 N}$

  





$A_{S T F G} \in \mathbb{R}^{3 N \times 3 N}$, and taken $A_{T G}$ within a red circle in Figure 3(b) for instance. It denotes the connection between same node from time step: 2 to 3 (current time step $t=2$ ). For each node $l \in\{1,2, \cdots, N\}, i=(t+1) * N+l=3 N+l$ and $j=t * N+l=2 N+l$, then $A_{S T F G(i, j)}=1$. To sum up, Temporal Connectivity graph denotes connection of the same node at proximate time steps.

Finally, Spatial-Temporal Fusion Graph $A_{S T F G} \in$ $\mathbb{R}^{K N \times K N}$ is generated. Altogether with the sliced input data of each STFGN Module:
$$
h^{0}=\left[X_{\mathcal{G}}^{(t)}, \cdots, X_{\mathcal{G}}^{(t+K)}\right] \in \mathbb{R}^{K \times N \times d \times C}
$$
It is sliced iteratively from total input data:
$$
X=\left[X_{\mathcal{G}}^{(t)}, \cdots, X_{\mathcal{G}}^{(t+T)}\right] \in \mathbb{R}^{T \times N \times d \times C}
$$
$X_{\mathcal{G}}^{(t)}$ is high-dimension feature of original data $\mathbf{X}_{\mathcal{G}}^{(t)} \cdot C$

![figure2](/assets/img/gnn/gnn429.png) 

![figure2](/assets/img/gnn/gnn430.png) 



## (2) STFG Neural Module

## (3) Gated Convolution Module

