# Accelerating Batch Active Learning Using Continual Learning Techniques

<br>

# 0. Abstract

Problem of Active Learning (AL) = high training costs

- since models are typically retrained from scratch after every query round. 

<br>

Standard AL on NN with warm stating ..

- Fails to accelerating training
- Fails to avoid catastrophic forgetting

when fine-tuning over AL query rounds

<br>

### Proposal) Continual Active Learning



 by biasing further training towards previously labeled sets. We accomplish this by employing existing, and developing novel, replay-based Continual Learning (CL) algorithms that are effective at quickly learning the new without forgetting the old, especially when data comes from an evolving distribution. We call this paradigm "Continual Active Learning" (CAL). We show CAL achieves significant speedups using a plethora of replay schemes that use model distillation and that select diverse/uncertain points from the history. We conduct experiments across many data domains, including natural language, vision, medical imaging, and computational biology, each with different neural architectures and dataset sizes. CAL consistently provides a $\sim 3 \mathrm{x}$ reduction in training time, while retaining performance and out-of-distribution robustness, showing its wide applicability.



# 1. Introduction

