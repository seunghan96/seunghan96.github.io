# 1. Introduction

## (1) Motivaton

- [Financial 도메인] **정확한 Information Retrieval(IR)**이 매우 중요

- [Financial 데이터] **복잡·동적**이며, 문서(10-K), 테이블, narrative 등 다양한 형태가 혼합

  $\rightarrow$ IR 난이도가 높음!

- [Financial 질의] 보통 **짧고, 모호하고, 약어·jargon·acronym**이 많이 포함

  - e.g., *“Recent CAGR in MS trading revenue”*처럼 회사명조차 축약된 경우가 많음.

<br>

## (2) 기존 dataset & 방법론

- [기존 QA datasets] 미리 제공된 고정 context를 기반

  $\rightarrow$ 실제 Financial QA의 retrieval 난이도를 반영하지 못함

- [LLM] Financial QA를 **closed-book**으로 처리할 경우 정확도가 매우 낮음

  - 단순히 **context window를 늘리는 방식**은 비효율적이고 비용이 큼.

<br>

## (3) FINDER

Solution: RAG이 Financial QA가 필수적!!

$\rightarrow$ **FINDER**는 이러한 문제를 해결하기 위해 설계된 dataset

- Ambiguous query
- Realistic financial search behavior
- Expert-grounded evidence & answers

<br>

## (4) Main Contributions

- 전문가가 만든 **5,703개의 Query–Evidence–Answer triplets** 제공
- 금융 QA benchmark 중 **가장 높은 수준의 query complexity** 포함
- SoTA retrievers·LLMs 성능 평가

<br>

# 2. Related Works

## (1) Financial QA Datasets

기존 Financial QA dataset

- 특정 reasoning task에는 강하지만, **retrieval 자체를 핵심 문제로 다루지 않음!**

- **핵심 문제**: 대부분의 datasets는

  1. **잘 정제된 질문**

  2. **명확한 context**,

  3. **모호성 없는 쿼리**

     를 기반으로 만들어져 **real-world 금융 IR 난이도**를 반영하지 못함!

<br>

Proposal: FINDER

- **Ambiguous, brief, acronym-heavy** real search queries 사용
  - e.g., Ambiguous: *“AAPL segment margin YoY?”*
  - e.g., Brief: *“TSLA delivery numbers”*
  - e.g., Acronym-heavy (약어·도메인 jargon 엄청 많음): MS=Morgan Stanley, EPS=Earnings Per Share
- Annotated **ground-truth evidence**를 제공
- Retrieval 난이도를 dataset 설계의 중심으로 둠.

<br>

## (2) RAG in Finance

- **[RAG]** LLM에 **external documents**를 retrieval하여 context로 제공

  $\rightarrow$ Hallucination, outdated knowledge 문제를 완화하는 핵심 기술.

- **[Financial 도메인]** **정보 업데이트 속도가 빠르고**, 전문 용어가 많아 LLM 단독으로는 신뢰성 부족

  → Retrieval 단계 품질이 금융 QA 성능을 크게 좌우함

- 최근 연구들의 주요 주제?

  - **Document indexing**: 검색 효율을 높이기 위해 문서를 구조화해 빠르게 조회할 수 있는 형태로 저장하는 과정
  - **Chunking**: 긴 문서를 검색 가능한 작은 단위(문단·섹션 등)로 분할해 retrieval 성능을 높이는 기법
  - **Reranking**: 1차 retrieval 결과를 더 정교한 모델(LLM 등)로 재정렬해 가장 관련성 높은 문서를 상위에 올리는 단계
  - **Query expansion**: 원래 질의를 synonym·관련 용어·도메인 knowledge로 확장해 retrieval 정확도를 높이는 기법.
  - **Embedding-based retrieval**: 문서와 질의를 벡터 공간에 임베딩해 유사도 기반으로 관련 문서를 검색하는 방식.

  $\rightarrow$ 모두 RAG의 성능을 결정하는 critical pipeline.

- Retrieval 성능이 낮으면 Generation 모델도 실패!!

  → 금융 QA에서는 **retrieval이 곧 성능의 upper bound**.

<br>

FINDER

- RAG 평가를 위해 만들어진 최초의 규모 있는 **domain-specific benchmark** 
- Retrieval 모델이 ambiguous query를 어떻게 파싱·해석하는지 평가할 수 있도록 설계

- e.g., Figure 1)

  

  - “MS trading revenue”처럼 ambiguous query를 요구하며,

  - System이 먼저 **MS → Morgan Stanley**를 해석해야 올바른 paragraph를 찾을 수 있음.

    → 기존 datasets에서 등장하지 않는 고난도 retrieval reasoning.

  





------



원하면 **Section 3. FINDER Dataset** 요약으로 이어갈게!

계속하려면 **“다음”**이라고 해줘.