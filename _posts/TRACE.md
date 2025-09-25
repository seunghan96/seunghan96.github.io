# TRACE: Temporal Grounding Video LLM via Causal Event Modeling

<br>

# 0. Abstract

Problem: Video LLMs struggle with **Video Temporal Grounding (VTG)** 

$\because$ Their outputs are unstructured text mixing timestamps and captions.

<br>

Proposal: TRACE

- Idea: **"Causal event modeling"**
  - Represent a video response = **"Sequence of events"**
  - Each event = **Timestamps** + **Salient score** + **Caption**
- Model: **TRACE**
  - **Task-interleaved** video LLM 
    - with **separate encoders/heads** for time, score, and text
    - with an **adaptive head-switching** decode
- Result: 
  - Strong zero-shot gains on YouCook2, Charades-STA, QVHighlights
  - Competitive after fine-tuning against task-specific systems. 

<br>

# **1. Introduction**

**Video Temporal Grounding (VTG)** 

- Existing video LLMs generate **pure natural language**

  $\rightarrow$ Ignore video’s inherent **structured nature** (time + saliency + text)

- Goal: Close the gap by ...

  - Modeling **event structure** explicitly 
  - Decoding in a causally consistent order

<br>

# **2. Related Work**

[1] Prior VTG: 

- Non-generative, Task-specific
- Good performance, but no zero-shot and poor flexibility.

<br>

[2] Prior video LLMs with time tokens 

- (e.g., Vid2Seq, TimeChat, VTimeLLM, VTG-LLM, LITA, Momentor, Hawkeye) 
- Still treat outputs as natural text, not structured events

<br>

TRACE adds a **formal causal factorization** and **task-interleaving** to align decoding with video structure. 

<br>

# **3. Methodology (Detailed)**

![figure2](/assets/img/llm/img868.png)

<br>

## **3.1 Causal Event Modeling (formalism)**

### a) Notation

- (1) Instruction tokens: $I$
- (2) Visual inputs (frames): $F$
- (3) Output: **Ordered event set** $R=\{e_1,\dots,e_K\}$, with each event $e_k=(t_k, s_k, c_k)$
  - $t_k$: timestamps (start/end in seconds)
  - $s_k$: salient score(s)
  - $c_k$: textual caption
    - $R=\{(t_k,s_k,c_k)\mid 1\le k\le K\}$.

<br>

### b) **Causal factorization (event-wise autoregression).**

- $P(e_k \mid e_{1:k-1}, I, F) = P(t_k \mid e_{1:k-1}, I, F)\; P(s_k \mid t_k, e_{1:k-1}, I, F)\; P(c_k \mid s_k, t_k, e_{1:k-1}, I, F)$.
- Next event depends on...
  - (1) **past events**
  - (2) **instructions**
  - (3) **visuals**

<br>

## (2) TRACE: Task-Interleaved Video LLM

### a) Separated multi-task processing

- **Backbone LLM:** Mistral-7B-v0.2.
- **Special tokens:** 
  - `⟨sep⟩` (end of timestamp/score)
  - `⟨sync⟩` (end of current task / head switch)
- **Timestamps & scores encoders/heads**
  - A small tokenizer with 13 symbols: `⟨0⟩…⟨9⟩, ⟨.⟩, ⟨sep⟩, ⟨sync⟩`
    - Embeddings initialized from the LLM
  - **Fixed numeric layout** to stabilize decoding
    - Timestamps: Use **4 integer digits + dot + 1 fractional digit** (e.g., 0125.4).
    - Scores: Use **1 integer digit + dot + 1 fractional digit**
  - **Example tokenization** of two timestamps `[10.23, 125.37]`:
    - `⟨0⟩⟨0⟩⟨1⟩⟨0⟩⟨.⟩⟨2⟩⟨sep⟩ ⟨0⟩⟨1⟩⟨2⟩⟨5⟩⟨.⟩⟨4⟩ ⟨sync⟩`
- **Visual frames encoder:**
  - Step 1) CLIP ViT-L/14 produces **576 visual tokens per frame**
  - Step 2) **Slot-Based Compression** reduces to **8 tokens/frame**
  - Step 3) **Time encoder** converts the frame timestamp into **6 time tokens** 
  - Step 4) Concatenate: 6 time tokens + 8 visual tokens ⇒ **14 tokens/frame** 
  - Step 5) Feed to LLM 

<br>

### b) Task-interleaved sequence modeling

![figure2](/assets/img/llm/img867.png)

[1] **Inter-event order:** 

- **Visual tokens** $F$ $\rightarrow$ **Instruction tokens** $I$ $\rightarrow$ Events in **chronological order** $e_1, e_2, \dots.$.

<br>

[2] **Intra-event order:** 

- **Time tokens** $t_k$ → **Score tokens** $s_k$ → **Text tokens** $c_k$

<br>

Yields a **specialized autoregressive sequence** aligned with video structure!

<br>

### c) Adaptive head-switching for generation

- Decode uses **three heads** (**time**, **score**, **text**) 
  - Switches **whenever ⟨sync⟩ is generated** (i.e., cycling **time → score → text**)


![figure2](/assets/img/llm/img869.png)

<br>

## (3) Training strategy & data

**Two stages**

- **Stage-1 (module init):** 
  - Freeze CLIP & LLM
  - Train **vision compression**, **time/score encoders+heads**, and **text head** 
  - Dataset: ~**1.9M** samples

- **Stage-2 (instruction tuning):** 
  - Fine-tune **LLM + task modules** (freeze CLIP) 
  - Dataset: ~**0.9M** samples


- **Sampling & hyper-params:** 128 frames/video

<br>

# **4. Experiments**

## **4.1 Setup**





- **Tasks & metrics:**

  

  - **Dense video captioning:** YouCook2, ActivityNet-Captions; CIDEr, METEOR, SODA_c; averaged over IoU ∈ {0.3, 0.5, 0.7, 0.9}; also **F1** for timestamp accuracy.
  - **Moment retrieval:** Charades-STA; **R@1@IoU=0.5/0.7**, **mIoU**.
  - **Highlight detection:** QVHighlights; **mAP@{0.5,0.75}**, **HIT@1**. Baselines include VideoChat/Valley/Video-LLaMA, TimeChat, VTimeLLM, Momentor, Hawkeye, VTG-LLM, LITA. 

  







## **4.2 Zero-shot results (TRACE-7B)**





- **YouCook2:** **+3.1 CIDEr**, **+4.9 F1** vs SOTAs among video LLMs.
- **Charades-STA:** **+6.5** (R@1@0.5), **+3.7** (R@1@0.7).
- **QVHighlights:** **+10.3 mAP**, **+9.2 HIT@1**.
- Outperforms **HawkEye** and **VTimeLLM-13B** despite smaller LLM. (See **Table 2, p.8**.) 







## **4.3 Ablations**





- **Causal event modeling matters:** removing it degrades YouCook2 and Charades-STA notably.
- **Separate encoders/heads are necessary:** replacing them with shared text tokenizer **fails to follow instructions**.
- **More frames help:** 8→64→128 frames steadily improve results. (See **Table 3 & Fig. 5, p.9**.) 







## **4.4 Fine-tuning**





- After 3 epochs: TRACE sets new SOTAs on YouCook2 (no audio) and is competitive with strong **non-generative** models on Charades-STA. (See **Table 5, p.10**.) 





------





# **5. Conclusion & Future Work**





- TRACE aligns decoding with **video event structure**, improving VTG zero-shot and fine-tuned performance while preserving general video understanding.
- Future: integrate **causality discovery graphs** as inputs or intermediate steps; augment datasets with timestamped QA pairs. (Discussion in **Appendix C, p.22–25**.) 









## **Methodology at a glance (equations & tokens)**

- **Event set:** $R=\{(t_k,s_k,c_k)\}_{k=1}^K$
- **Causal factorization:** $P(e_k\!\mid\!e_{1:k-1},I,F)=P(t_k\!\mid\!e_{1:k-1},I,F)\,P(s_k\!\mid\!t_k,e_{1:k-1},I,F)\,P(c_k\!\mid\!s_k,t_k,e_{1:k-1},I,F)$
- **Numeric tokenization:** digits 0–9, dot, ⟨sep⟩, ⟨sync⟩; timestamps fixed-width (e.g., 0125.4), scores compact (e.g., 3.8)
- **Interleave order:** $F \rightarrow I \rightarrow [\,t_1 \!\rightarrow\! s_1 \!\rightarrow\! c_1\,] \rightarrow \cdots \rightarrow [\,t_K \!\rightarrow\! s_K \!\rightarrow\! c_K\,]$

