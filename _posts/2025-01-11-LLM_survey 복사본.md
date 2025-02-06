---
title: Large Language Models; A Survey
categories: [MULT, LLM, NLP]
tags: []
excerpt: arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Large Language Models: A Survey

https://arxiv.org/pdf/2402.06196

<br>

### Contents



# Abstract

Recent trends in LLMs

- **Strong performance** on a wide range of **NLP tasks**
  - e.g., ChatGPT: 2022.11
- Training **billions** of model’s parameters & **massive** amounts of text data
  - feat. ***Scaling laws***

<br>

This paper: 

- (1) **3 LLM families** (GPT, LLaMA, PaLM)

- (2) Techniques to **build/augment LLMs**
- (3) **Popular datasets** 
- (4) LLM evaluation **metrics**

<br>

# 1. Introduction

### P1) Language modeling (LM)

- Pass

<br>

### P2) Transformer-based LLMs

- [Dataset] Pretrained on **Web-scale text corpora**
- [Example] ChatGPT and GPT-4 

LLM is the basic building block for ...

$\rightarrow$ The development of **general-purpose AI agents** or **artificial general intelligence (AGI)** !!

<br>

### P3) Challenges

- Challenging to figure out the best recipes to build **LLM-powered AI systems**

<br>

### P4) Four waves of recent success of LLMs

- (1) **Statistical** LM
- (2) **Neural** LM
- (3) **Pre-trained** LM
- (4) **Large** LM 

<br>

### P5) (1) Statistical language models (SLMs) 

- Text = Sequence of words

  $\rightarrow$ Probability of text = **Product of their word probabilities**

- e.g.) **Markov chain models**  ( = **"n-gram"** models )

  - ***Smoothing***: To deal with **"data sparsity"**
    - (i.e., assigning zero probabilities to **unseen** words or n-grams) 

- Limitation: **Sparsity**

  $\rightarrow$ Cannot fully capture the diversity of language!

<br>

### P6) (2) Neural language models (NLMs)

- Handle **"Data sparsity"** 
  - Map words to **embedding vectors**
- **Next word prediction**
  - Based on the aggregation of its **preceding words** using **NN**

<br>

### P7) (3) Pre-trained language models (PLMs)

- **Task-agnostic** 
- **Pre-training** & **fine-tuning** 
  - (1) **Pre-trained** 
    - On large scale dataset
    - For **general tasks** (e.g., word prediction)
  - (2) **Fine-tuned** 
    - To specific tasks 
    - Using small amounts of **(labeled) task-specific data**

<br>

### P8) (4) Large language models (LLMs) 

- (Mainly refer to) **Transformer-based** NLMs
- Contain tens to hundreds of **billions of parameters**
- Pretrained on **massive text data**
- Ex) ***PaLM , LLaMA , and GPT-4***
- (Compared to PLM) Much larger in model size & better performance
  - ***emergent abilities*** that are not present in smaller-scale language models
- ***Emergent abilities***
  - (1) **In-context learning**
    - Learn a new task from a small set of examples presented in the prompt **at inference time**
  - (2) **Instruction following**
    - (After instruction tuning) **Follow the instructions** for new types of tasks without using explicit examples
  - (3) **Multi-step reasoning**
    - Solve a complex task by **breaking down** that task into intermediate reasoning steps 

- **Augmentation**
  - LLMs can also be augmented by using **external knowledge and tools**
  - Effect
    - Can effectively **interact** with users and environment
    - Can  continually improve itself **using feedback** data collected through interactions (e.g. via RLHF)

<br>

![figure2](/assets/img/llm/img330.png)

<br>

### P9) AI agents

- LLMs can be deployed as so-called ***AI agents***

- AI agents?

  = Artificial entities that sense their **environment**, make **decisions**, and take **actions**

- AI agent researches

  - (Previous) Agents for **specific tasks** and domains
  - (Recent) **General-purpose AI agents** based on LLMs ( feat. Emergent abilities )

- LLM vs. AI Agent

  - **LLMs**: Trained to produce responses in **static settings**

  - **AI agents**: Need to take actions to interact with **dynamic environment**

    $\rightarrow$ $\therefore$ LLM-based agents often need to augment LLMs!

    - e.g., Obtain updated information from **external knowledge bases**
    - e.g., Verify whether a system action produces the expected result
    - e.g., Cope with when things do not go as expected

<br>

### P10) Section Introduction

- Section II: Overview of **SOTA LLMs** ( Three LLM families (GPT, LLaMA and PaLM) )
- Section III: **How LLMs are built**
- Section IV: **How LLMs are used**, and augmented for real-world applications
- Sections V and VI: **Popular datasets and benchmarks** for evaluating LLMs
- Section VII: **Challenges** and **future research** directions

<br>

# 2. LLM

1. **Review** of early pre-trained neural language models ( = base of LLMs )
2. Focus our discussion on **three families of LLMs** ( GPT, LlaMA, and PaLM )

![figure2](/assets/img/llm/img331.png)

<br>

## (1) Early Pre-trained Neural Language Models

### P1) History of NLMs

[13] : First **neural language models (NLMs)**

[14] : Applied NLMs to **machine translation**

[41] : **RNNLM** (an open source NLM toolkit) 

[42] : Popularize NLMs

[After] **NLMs based on RNNs (& variants)** were widely used 

- E.g., Machine translation, text generation and text classification 

<br>

### P2) Invention of Transformer

- Transformer = Allow for much more **parallelization** than RNNs

- Development of **Pre-trained** language models (PLMs) 

  ( + **Fine-tuned** for many downstream tasks )

- Three categories
  - (1) ***Encoder-only***
  - (2) ***Decoder-only***
  - (3) ***Encoder-decoder models***

<br>

### P3) Transformer: Encoder-only PLMs

#### P3-1) Encoder-only?

- Only consist of an encoder network
- Developed for **language understanding tasks**
  - e.g., Text classification
- e.g.) ***BERT & variants***
  - e.g., RoBERTa, ALBERT, DeBERTa, XLM, XLNet, UNILM

<br>

#### P3-2) BERT

BERT (Birectional Encoder Representations from Transformers)

- 3 modules 
  - (1) **Embedding module** 
    - Input text $\rightarrow$ Sequence of embedding vectors
  - (2) **Stack of Transformer encoders** 
    - Embedding vectors $\rightarrow$ Contextual representation vectors
  - (3) **Fully connected layer** 
    - Representation vectors $\rightarrow$ One-hot vectors
- 2 pretraining tasks
  - (1) **Masked language modeling (MLM)** 
  - (2) **Next sentence prediction (NSP)**
- Finetuning
  - Can be fine-tuned by adding a classifier layer
  - e.g., **Text classification, question answering to language inference** 

<br>

![figure2](/assets/img/llm/img332.png).

<br>

#### P3-3) RoBERTa, ALBERT, DeBERTa, ELECTRA, XLMs

**RoBERTa (A Robustly Optimized BERT Pretraining Approach)**

- Improves the **robustness** of BERT
- Key changes
  - (1) Modify a few key **hyperparameters**
  - (2) **Remove the NSP task**
  - (3) Train with much **larger mini-batches and learning rates**

<br>

**ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)**

- Two **parameter-reduction** techniques 

  - (1) Split the embedding matrix $\rightarrow$ Into **two smaller matrices**
  - (2) **Repeating layers** split among groups 

  $\rightarrow$ Lower memory consumption & increase the training speed of BERT

<br>

**DeBERTa (Decoding-enhanced BERT with disentangled attention)**

- Improves the BERT and RoBERTa models
- Two novel techniques 
  - **(1) Disentangled attention mechanism**
    - Each word = Two vectors that encode its (a) content & (b) position
    - Attention weights among words are computed using **disentangled matrices** on their contents and relative positions
  - **(2) Enhanced mask decoder**
    - To incorporate absolute positions in the decoding layer
- (During fine-tuning) Novel **virtual adversarial training** method
  - To improve models’ generalization. 

<br>

**ELECTRA**

- (New pre-training task) **Replaced Token Detection (RTD)**
- MLM vs. RTD
  - a) Target token
    - (MLM) Mask the input
    - (RTD) Corrupts it by **replacing some tokens** with plausible alternatives (sampled from a **small generator network**)
  - b) Prediction
    - MLM: Predicts the original identities of the corrupted tokens
    - RTD: **Discriminative model** is trained to predict whether a token in the corrupted input was replaced by a generated sample or not
- Effectivenss of RTD
  - More **sample-efficient** than MLM
    - RTD: Defined over **all input tokens**
    - MLM: Only small subset being masked out

![figure2](/assets/img/llm/img333.png)

<br>

#### P3-4) XLMs

- Extend BERT to **"cross-lingual"** language models
- Two methods
  - (1) **Unsupervised** method: Relies on **"monolingual"** data
  - (2) **Supervised** method: Leverages parallel data with a new **"cross-lingual"** language model objective
- SOTA results
  - E.g., Cross-lingual classification, unsupervised and supervised machine translation

![figure2](/assets/img/llm/img334.png)

<br>

#### P3-5) XLNet & UNILM

( = Encoder-only + **Advantages of decoder models** )

**XLNet**

- Based on Transformer-XL

- Pre-trained using a **generalized "autoregressive" method** 

  - Enables learning **bidirectional** contexts 

    by maximizing the expected likelihood over **"all permutations of the factorization order"**

<br>

**UNILM (UNIfied pre-trained Language Model) **

- Pre-trained using **three types** of language modeling tasks

  - (1) **Unidirectional** prediction
  - (2) **Bidirectional** prediction
  - (3) **Sequence-to-sequence** prediction

  $\rightarrow$ By employing a shared Transformer network & utilizing specific self-attention masks

  -  Mask: to control what context the prediction is conditioned on!

![figure2](/assets/img/llm/img335.png)

<br>

### P4) Transformer: Decoder-only PLMs

Example: **GPT-1** and **GPT-2** (by OpenAI)

$\rightarrow$ Foundation to more powerful LLMs (e.g., **GPT-3** and **GPT-4**)

<br>

#### P4-1) GPT1

- ***Decoder-only*** Transformer model
- [Pretrain] In a **SSL fashion** (e.g., Next word/token prediction)
- [Fine-tune] On each specific downstream task

![figure2](/assets/img/llm/img336.png)

<br>

#### P4-2) GPT2

- Shows that LMs are able to learn to perform specific NLP tasks **"w/o any explicit supervision"**
- Dataset: large **WebText** dataset (consisting of millions of webpages)
- GPT-1 + $\alpha$:
  - (1) **Layer normalization**: Moved to the input of each sub-block
  - (2) **Additional layer normalization**: Added after the final self-attention block
  - (3) **Initialization**: Modified to account for the accumulation on the residual path and scaling the weights of residual layers, 
  - (4) **Vocabulary size**: Expanded
  - (5) **Context size**: Increased from 512 to 1024 tokens.

<br>

### P5) Transformer: Encoder-Decoder PLMs

- Shows that almost all NLP tasks can be cast as a **"sequence-to-sequence"** generation task

- Unified model (as "Encoder-decoder framework")

  $\rightarrow$ Can an perform all **(1) natural language understanding** and **(2) generation tasks**

<br>

#### P5-1) T5 (Text-to-Text Transfer Transformer) & mT5

**T5: Unified framework** 

( Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer )

- All NLP tasks are cast as a **text-to-text generation** task
- Relative positional embedding (but slow)

![figure2](/assets/img/llm/img345.png)

![figure2](/assets/img/llm/img346.png)

<br>

**mT5: Multilingual variant of T5**

( mT5: A massively multilingual pre-trained text-to-text transformer)

- Pre-trained on a new Common Crawl-based dataset 
  - Consisting of texts in **101 languages**

<br>

#### P5-2) MASS

( MASS: Masked Sequence to Sequence Pre-training for Language Generation )

- [Pretraining task] Reconstruct a **sentence fragment** given the **remaining part of the sentence**

- Encoder & Decoder

  - **[Encoder]** Input = Masked sentence with randomly masked fragment
  - **[Decoder]** Predicts the masked fragment

  $\rightarrow$ Training: Jointly trains the encoder and decoder for **language embedding** and **generation**, respectively.

<br>

#### P5-3) BART

( BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension )

- **Sequence-to-sequence** translation model architecture
- [Pretraining task] **Corrupt** text with noise & **Reconstruct** the original text

<br>

#### P5-1)

<br>

## (2) LLM Families

### P1) LLM

- **Transformer**-based PLM ( ~ hundreds of billions of parameters )
- Stronger language **understanding** and **generation** (vs. PLM)
- **Emergent abilities** that are not present in smaller-scale models
- Three LLM families
  - **(1) GPT**
  - **(2) LLaMA**
  - **(3) PaLM**

![figure2](/assets/img/llm/img337.png)

<br>

### P2) The GPT Family

Definition = Family of ***decoder-only*** Transformer-based language models ( by **OpenAI** )

-  GPT-1, GPT-2, GPT-3, InstrucGPT, ChatGPT, GPT-4, CODEX, and WebGPT...
- **[1] Early models**
  - GPT-1 and GPT-2
  - ***Open*** Source
- **[2] Recent models**
  - GPT-3 and GPT-4
  - ***Closed*** source ( Only be accessed via APIs )

<br>

#### P2-1) GPT3

- **175 B** params
- Widely considered as the first LLM in that ..
  - (1) Much **larger** than previous PLMs, 
  - (2) First time demonstrates **emergent abilities** 
- Shows the emergent ability of **"in-context learning"**
  - (1) To any downstream tasks ***without any gradient updates or fine-tuning***
  - (2) Demonstrations specified purely ***via text interaction*** with the model. 

- Strong performance on **many NLP tasks**

<br>

![figure2](/assets/img/llm/img338.png)

- Function of the number of examples in **in-context prompts**

<br>

#### P2-2) CODEX

- Descendant of **GPT-3** (by OpenAI in March 2023)

- General-purpose **"programming" model**

  - Parse natural language & **Generate code**

- Fine-tuned for programming applications on code corpora collected from GitHub

  $\rightarrow$ CODEX powers Microsoft’s **GitHub Copilot**

<br>

#### P2-3) WebGPT

- Descendant of GPT-3

- Fine-tuned to answer open-ended questions using a **text-based web browser**

  $\rightarrow$ Facilitating users to search and navigate the web. 

- Trained in 3 steps
  - Step 1) Learn to mimic **human browsing behaviors** 
    - Using human demonstration data. 
  - Step 2) **Reward function** 
    - Learned to predict **human preferences**
  - Step 3) **Reinforcement learning & Rejection sampling**
    - Refined to optimize the reward function

<br>

#### P2-4) InstructGPT

- Goal: To follow expected **"human instructions"**
- Align LLM with user intent on a wide range of tasks by fine-tuning with **human feedback**
- **Reinforcement Learning from Human Feedback (RLHF)**

![figure2](/assets/img/llm/img339.png)

<br>

#### P2-5) ChatGPT

- Most important milestone of LLM development (November 30, 2022)
- **Chatbot**: Wide range of tasks (such as question answering, information seeking, text summarization...)
- Powered by **GPT-3.5** (and later by **GPT-4**) (= a sibling model to **InstructGPT**)

<br>

#### P2-6) GPT-4

- **Multimodal LLM** (Launched in March, 2023)
  - Input: ***Image*** and text
  - Output: Text outputs. 
- Exhibits ***human-level performanc***e on various benchmarks
- (Same as early GPT models)
  - Step 1) Pre-trained to predict next tokens
  - Step 2) Fine-tuned with **RLHF** to align model behaviors with human-desired ones

![figure2](/assets/img/llm/img340.png)

<br>

### P3) The LLaMA Family (by Meta)

***LLaMA models are "open-source"***

$\rightarrow$ Grows rapidly!

<br>

#### P3-1) LLaMA

( LLaMA: Open and Efficient Foundation Language Models )

https://arxiv.org/pdf/2302.13971

- [Date] February 2023
- [Size] Ranging from **7B to 65B** params
- [Data] Pre-trained on trillions of tokens ( from **publicly available** datasets )
- [Arch] **Transformer architecture of GPT-3 + $\alpha$**
- Difference?
  - (1) ReLU $\rightarrow$ **SwiGLU**
  - (2) Absolute positional embedding $\rightarrow$ **Rotary positional embeddings**
  - (3) Standard LN $\rightarrow$ **Root-mean-squared LN**
- Result: 
  - **(open-source) LLaMA-13B** > (proprietary) GPT-3 (175B) on most benchmarks

![figure2](/assets/img/llm/img343.png)

![figure2](/assets/img/llm/img344.png)

<br>

Rotary positional embedding

- https://www.youtube.com/watch?v=o29P0Kpobz0

![figure2](/assets/img/llm/img347.png)

![figure2](/assets/img/llm/img348.png)

<br>

#### P3-2) LLaMA-2

( LLaMA 2: Open Foundation and Fine-Tuned Chat Models )

https://arxiv.org/pdf/2307.09288

- by Meta (+ partner Microsoft )

- Include both 
  - (1) Foundation language models
  - (2) Chat models (finetuned for dialog) $\rightarrow$ **LLaMA-2 Chat**

<br>

![figure2](/assets/img/llm/img341.png)

- Step 1) Pretrain LLaMA-2 with **publicly available** online data
- Step 2) Supervised fine-tuning **(SFT)**
- Step 3) Iteratively refined using...
  - **(1) RLHF**
  - **(2) Rejection sampling** 
  - **(3) Proximal policy optimization (PPO)**

<br>

#### P3-3) Alpaca

- Fine-tuned from the (1) **LLaMA-7B** model with (2) **52K instruction-following demonstrations**
- 52K instruction-following demonstrations
  - Generated in the style of self-instruct using GPT-3.5 (**text-davinci-003**)
- Very **cost-effective** for training (especially for academic research)
- (Self-instruct evaluation set) Alpaca performs **similarly to GPT-3.5**, despite that Alpaca is **much smaller**



#### P3-4) Vicuna

- **Vicuna13B**: 13B chat model
  - By fine-tuning **LLaMA** on user-shared **conversations** collected from ShareGPT
- Evaluation
  - (Evaluator = GPT4) Vicuna-13B achieves more than 90% quality of OpenAI’s ChatGPT and Google’s Bard
- Training cost of Vicuna-13B is merely **$300**!

<br>

#### P3-5) Guanaco 

- Also finetuned **LLaMA** models using **instruction-following** data
- Efficient fine-tuning
  - Finetuning is done ***very efficiently*** using **QLoRA**
  - Finetuning a **65B** parameter model can be done on a **single 48GB GPU**
  - **QLoRA**: Back-propagates gradients through a frozen, 4-bit quantized PLM into LoRA
- Result: Best Guanaco model 
  - Outperforms all previously released models on the **Vicuna benchmark**
    - 99.3% of the performance level of ChatGPT ( with much lighter model )

<br>

#### P3-6) Koala

- Another **instruction-following** language model built on **LLaMA**
  - Specific focus on interaction data that include user inputs and responses generated by **highly capable closed-source chat models (e.g., ChatGPT)**

<br>

#### P3-7) Mistral-7B

- Mistral-7B: Engineered for **superior performance & efficiency**
- (1) **Grouped-query attention** $\rightarrow$ For faster inference
- (2) **Sliding window attention** $\rightarrow$ To effectively handle sequences of arbitrary length with a reduced inference cost
- Outperforms the best open-source 
  - 13B model (**LLaMA-2-13B**) across all evaluated benchmarks
  - 34B model (**LLaMA-34B**) in reasoning, mathematics, and code generation. 

<br>

#### P3-8) Summary of LLaMA

- LLaMA or LLaMA2, including Code LLaMA [66], Gorilla [67], Giraffe [68], Vigogne [69], Tulu 65B [70], Long LLaMA [71], and Stable Beluga2 [72], just to name a few.

<br>

### P4) The PaLM Family

**PaLM (Pathways Language Model)** ... by Google

First PaLM model: April 2022 

- [Size] 540B params
- [Dataset] High-quality text corpus consisting of 780B tokens
- [GPU] 6144 TPU v4 chips using the ***Pathways system***
  - Enables **highly efficient training across multiple TPU Pods**

<br>

#### P4-1) U-PaLM

- U-PaLM models of 8B, 62B, 540B 

  = Trained on PaLM with UL2R

  - A method of continue training LLMs on a few steps with **UL2’s mixture-of-denoiser objective**
  - Approximately **2x computational saving**

<br>

#### P4-1) Flan-PaLM

- Flan-PaLM = U-PaLM + **Instruction finetuning**
  - Finetuning: Much larger number of tasks, larger model sizes, and chain-ofthought data
- Result: Substantially outperforms previous instruction-following models
- Fine-tuning data: comprises 473 datasets, 146 task categories, and 1,836 total tasks

![figure2](/assets/img/llm/img342.png)

<br>

#### P4-1) PaLM-2

- More **compute-efficient** LLM 
- With better **multilingual and reasoning capabilities**

<br>

#### P4-1) Med-PaLM

- Domain-specific PaLM
  - Finetuned on PaLM using instruction prompt tuning
- Designed to provide high-quality answers to medical questions

<br>

## (3) Other Representative LLMs

#### P1)

Other popular LLMs 

- Which do not belong to those three model families

<br>

#### P2) FLAN

- Simple method for improving the zero-shot learning abilities

  $\rightarrow$ Showed that instruction tuning LMs on a collection of datasets substantially improves zero-shot performance

- Instruction-tuned model, "FLAN"

  - Pretrained LM with 137B params. 
  - Instruction tune it on over 60 NLP datasets 
    - verbalized via natural language instruction templates

![figure2](/assets/img/llm/img349.png)

<br>

#### P3) Gopher

- Analysis of Transformer-based LMs across a wide range of model scales
- Gopher = 280B params model
- Evaluated on 152 diverse tasks

![figure2](/assets/img/llm/img350.png)

<br>

#### P4) T0

- Easily mapping ***any*** natural language tasks into a ***human-readable*** prompted form

- Convert (a) $\rightarrow$ (b)

  - (a) Supervised datasets
  - (b) **Multiple prompts** with **diverse wording**

  $\rightarrow$These prompted datasets allow for benchmarking the ability of a model to perform completely **held-out tasks**

<br>

#### P5) ERNIE 3.0

- Unified framework for pre-training large-scale knowledge enhanced models
  - AR model + AE model
- Tailored for both natural language (1) understanding & (2) generation tasks 
- ERNIE 3.0 = 10B params + 4TB 

![figure2](/assets/img/llm/img351.png)

<br>

#### P6) RETRO (Retrieval Enhanced Transformer)

- Enhanced AR model

  - By conditioning on document chunks retrieved from a large corpus

    ( based on local similarity with preceding tokens )

- Frozen Bert retriever & Differentiable encoder & Chunked cross-attention mechanism

  $\rightarrow$ Predict tokens based on an order of magnitude with more data than what is typically consumed during training.

![figure2](/assets/img/llm/img352.png)

<br>

#### P7) GLaM (Generalist Language Model)

- Sparsely activated MoE
  - To scale the model capacity & Incurring substantially less training cost
- Largest GLaM = 1.2T parameters ( = 7x larger than GPT3 )
  - 1/3 of the energy used to train GPT-3
  - 1/2 of the computation FLOPs for inference

![figure2](/assets/img/llm/img353.png)

<br>

#### P8) LaMDA

- Transformer-based models specialized **for dialog**

  - Up to **137B** params & pre-trained on **1.56T** words of public dialog data and web text

- Findings: Fine-tuning with **annotated data** & enabling the model to consult **external knowledge sources**

  $\rightarrow$ Lead to significant improvements towards the two key challenges of **safety and factual grounding**

<br>

#### P9) OPT (Open Pre-trained Transformers)

- Decoder-only pre-trained transformers 
  - params: 125M ~ 175B 

![figure2](/assets/img/llm/img354.png)

<br>

#### P10) Chinchilla

- Investigated the optimal model size and number of tokens under a given compute budget
- Experimental settings
  - Over 400 language models (70M~16B parmas + 5~5B tokens)
- Findings: model size & number of training tokens should be scaled equally
- Chinchilla = Compute-optimal model
  - Same compute budget as Gopher but with 70B parameters and 4% more data

<br>

#### P11) Galactica

- LLM that can store, combine and reason about scientific knowledge
- Dataset: Large scientific corpus of papers, reference material, knowledge bases ...
- Experiments: Outperform ...
  - Chinchilla on mathematical MMLU: by 41.3% to 35.7%
  - PaLM 540B on MATH: with a score of 20.4% versus 8.8%

<br>

#### P12) CodeGen

- Family of LLMs up to 16.1B params

- Dataset: 

  - (1) Natural language
  - (2) Programming language data
  - (3) Open sourced the training library JAXFORMER

- Competitive with the previous SOTA on zero-shot Python code generation on HumanEval. 

- Multi-step paradigm for program synthesis

  = Single program is factorized into multiple prompts specifying sub-problems

- Constructed an open benchmark: Multi-Turn Programming Benchmark (MTPB)
  - Consisting of 115 diverse problem sets that are factorized into multi-turn prompts

<br>

#### P13) AlextaTM (Alexa Teacher Model)

- Demonstrated that **multilingual seq2seq models**, pre-trained on a mixture of denoising and Causal Language Modeling (CLM) tasks, are more efficient few-shot learners than **decoder-only models** on various task!

<br>

#### P14) Sparrow

- Information-seeking dialogue agent
  - More helpful, correct, and harmless compared to prompted language model baselines
- Use RLHF 

![figure2](/assets/img/llm/img355.png)

<br>

#### P15) Minerva

- Pretrained on general natural language data

- Further trained on technical content

  $\rightarrow$ To tackle previous LLM struggle with quantitative reasoning 

  ( e.g., mathematics, science, and engineering problems )

<br>

#### P16) MoD (Mixture-of-Denoisers)

- Unified perspective for self-supervision in NLP

- Findings

  - How different pre-training objectives can be cast as one another
  - How interpolating between different objectives can be effective

- Mixture-of-Denoisers (MoD)

  - Pretraining objective = Combines diverse pre-training paradigms

    $\rightarrow$ This framework is known as Unifying Language Learning (UL2)

![figure2](/assets/img/llm/img356.png)

<br>

#### P17) BLOOM

- [Model] Decoder-only Transformer
- [Size] 176B
- [Dataset] ROOTS corpus
  - Hundreds of sources in 46 natural and 13 programming languages (59 in total)

![figure2](/assets/img/llm/img357.png)

<br>

#### P18) GLM

- GLM-130B: Bilingual (English and Chinese) pre-trained LLM

<br>

#### P19) Pythia

- Suite of 16 LLMs  (70M ~ 12B params)
- Trained on public data seen in the exact same order
- Public access to 154 checkpoints for each one of the 16 models

<br>

#### P20) Orca

- 13B parameter model
- Imitate the reasoning process of large foundation models
- Learns from rich signals from GPT-4 
  - e.g., explanation traces, step-by-step thought processes, .. 

<br>

#### P21) StarCoder

StarCoder & StarCoderBase

- 15.5B parameter models with 8K context length

- Infilling capabilities

- Fast large-batch inference enabled by multi-query attention

- [Dataset]

  - StarCoderBase: 1T tokens sourced from The Stack

    ( = Large collection of permissively licensed GitHub repositories )

  - StarCoder: StarCoderBase + (fine-tune) 35B Python tokens

<br>

#### P22) KOSMOS

- Multimodal LLM (MLLM): Can perceive general modalities

- Trained from scratch

  - On web-scale **multi-modal** corpora

    ( including arbitrarily interleaved text and images, image-caption pairs, and text data )

- Impressive performance on ...

  - (1) Language understanding, generation, and even OCR-free NLP
  - (2) Perception-language tasks
    - e.g., Multimodal dialogue, image captioning, visual question answering
  - (3) Vision tasks
    - e.g., Image recognition with descriptions

<br>

#### P23) Gemini

- Multimodal LLM (MLLM): Can perceive general modalities

  $\rightarrow$ Promising. capabilities across image, audio, video, and text understanding

- Built on top of Transformer decoders
- Support 32k context length (via using efficient attention mechanisms).

- Three versions
  - (1) Ultra: for highly-complex tasks
  - (2) Pro: for enhanced performance and deployability at scale
  - (3) Nano: for on-device applications

<br>

#### P24) Overview of some of the most representative LLM frameworks

![figure2](/assets/img/llm/img358.png)

<br>

# 3. How LLMs Are Built

#### P1)

In this section, we first review the popular architectures used for LLMs, and then discuss data and modeling techniques ranging from data preparation, tokenization, to pre-training, instruction tuning, and alignment.

<br>

#### P2)

Once the model architecture is chosen, the major steps involved in training an LLM includes: data preparation (collection, cleaning, deduping, etc.), tokenization, model pretraining (in a self-supervised learning fashion), instruction tuning, and alignment. We will explain each of them in a separate subsection below. These steps are also illustrated in Fig 25.

<br>

## (1) Dominant LLM Architecture

#### P1)

The most widely used LLM architectures are encoder-only, decoder-only, and encoder-decoder. Most of them are based on Transformer (as the building block). Therefore we also review the Transformer architecture here.

<br>

#### P2) Arch: Transformer

1) Transformer: in a ground-breaking work [44], Vaswani et al. proposed the Transformer framework, which was originally designed for effective parallel computing using GPUs. The heart of Transformer is the (self-)attention mechanism, which can capture long-term contextual information much more effectively using GPUs than the recurrence and convolution mechanisms. Fig 26 provides a high-level overview of transformer work. In this section we provide an overview of the main elements and variants, see [44], [123] for more details.

<br>

#### P2-1)

The Transformer language model architecture, originally proposed for machine translation, consists of an encoder and a decoder. The encoder is composed of a stack of N = 6 identical Transformer layers. Each layer has two sub-layers. The first one is a multi-head self-attention layer, and the other one is a simple position-wise fully connected feed-forward network. The decoder is composed of a stack of 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder has a third sub-layer, which performs multi-head attention over the output of the encoder stack. The attention function can be described as mapping a query and a set of keyvalue pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Instead of performing a single attention function with dmodel dimensional keys, values and queries, it is found to be beneficial to linearly project the queries, keys and values h with different, learned linear projections to dk, dk and dv dimensions, respectively. Positional encoding is incorporated to fuse information about the relative or absolute position of the tokens in the sequence.

<br>

#### P3) Arch: Encoder-Only

2) Encoder-Only: For this family, at each stage, the attention layers can access all the words in the initial sentence. The pre-training of these models usually consist of somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence. Encoder models are great for tasks requiring an understanding of the full sequence, such as sentence classification, named entity recognition, and extractive question answering. One prominent encoder only model is BERT (Bidirectional Encoder Representations from Transformers), proposed in [24].

<br>

#### P4)

3) Decoder-Only: For these models, at each stage, for any word, the attention layers can only access the words positioned before that in the sentence. These models are also sometimes called auto-regressive models. The pretraining of these models is usually formulated as predicting the next word (or token) in the sequence. The decoder-only models are best suited for tasks involving text generation. GPT models are prominent example of this model category.

<br>

#### P1)

4) Encoder-Decoder: These models use both encoder and decoder, and are sometimes called sequence-to-sequence models. At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder only accesses the words positioned before a given word in the input. These models are usually pretrained using the objectives of encoder or decoder models, but usually involve something a bit more complex. For instance, some models are pretrained by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is then to predict the text that this mask word replaces. Encoder-decoder models are best suited for tasks about generating new sentences conditioned on a given input, such as summarization, translation, or generative question answering.

<br>

## (2) Data Cleaning

#### P1)

Data quality is crucial to the performance of language models trained on them. Data cleaning techniques such as filtering, deduplication, are shown to have a big impact on the model performance.

<br>

#### P2)

As an example, in Falcon40B [124], Penedo et al. showed that properly filtered and deduplicated web data alone can lead to powerful models; even significantly outperforming models from the state-of-the-art trained on The Pile. Despite extensive filtering, they were able to obtain five trillion tokens from CommonCrawl. They also released an extract of 600 billion tokens from our REFINEDWEB dataset, and 1.3/7.5B parameters language models trained on it. 27 shows the Refinement process of CommonCrawl data by this work.

<br>

#### P3) Data Filtering

1) Data Filtering: Data filtering aims to enhance the quality of training data and the effectiveness of the trained LLMs. Common data filtering techniques include:

<br>

#### P3-1)

Removing Noise: refers to eliminating irrelevant or noisy data that might impact the model’s ability to generalize well. As an example, one can think of removing false information from the training data, to lower the chance of model generating false responses. Two mainstream approaches for quality filtering includes: classifier-based, and heuristic-based frameworks.

<br>

#### P3-2)

Handling Outliers: Identifying and handling outliers or anomalies in the data to prevent them from disproportionately influencing the model.

<br>

#### P3-3)

Addressing Imbalances: Balancing the distribution of classes or categories in the dataset to avoid biases and ensure fair representation. This is specially useful for responsible model training and evaluation.

<br>

#### P3-4)

Text Preprocessing: Cleaning and standardizing text data by removing stop words, punctuation, or other elements that may not contribute significantly to the model’s learning.

<br>

#### P3-5)

Dealing with Ambiguities: Resolving or excluding ambiguous or contradictory data that might confuse the model during training. This can help the model to provide more definite and reliable answers.

<br>

#### P4) Deduplication

2) Deduplication: De-duplication refers to the process of removing duplicate instances or repeated occurrences of the same data in a dataset. Duplicate data points can introduce biases in the model training process and reduce the diversity, as the model may learn from the same examples multiple times, potentially leading to overfitting on those particular instances. Some works [125] have shown that de-duplication improves models’ ability to generalize to new, unseen data.

<br>

#### P4-1)

The de-duplication process is particularly important when dealing with large datasets, as duplicates can unintentionally inflate the importance of certain patterns or characteristics. This is especially relevant in NLP tasks, where diverse and representative training data is crucial for building robust language models. 

<br>

#### P4-2)

The specific de-duplication method can vary based on the nature of the data and the requirements of the particular language model being trained. It may involve comparing entire data points or specific features to identify and eliminate duplicates. At the document level, existing works mainly rely on the overlap ratio of high-level features (e.g. n-grams overlap) between documents to detect duplicate samples.

<br>

## (3) To
