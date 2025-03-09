# 0. 자기소개

연세대학교 통계데이터사이언스학과 통합과정 10학기에 진학예정이고, 이제 오는 8월에 졸업을 앞둔 이승한이라고 합니다.

main은 시계열을 위주로 연구를 진행했으나, 꼭 해당 도메인에만 국한되어서 관심을 가지지는 않습니다.

```
시계열 + LLM의 접목 머릿속으로 가져놓기
```



# 1. 지원 동기



# 1.LLM to VLM

지원한 동기를 말씀드리자면, 

1. VLM뿐만 아니라, 여태까지 DL 아키텍처들이 발전되어왔던 방향을 생각해보면

   - 최대한 (1) simple & (2) unified & (3) diverse (=multimodal)한 데이터를 궁극적인 목표로 발전해왔음

2. 

   - 그래서 그러한 방향을
     현재 이 직무에 적용시켜 생각해보자면, **다양한 모달리티의 데이터를 (성능의 희생을 최대한 적게하면서) unified되고 simple한 아키텍처**를 만들고자함

   - 그리고, **LLM의 발전은 지금까지 많이 이뤄졌으니**, 이제 **Vision을 거기에 적용시킨 VLM을 발전**시키고 싶은게 트렌드가 아닌가 싶습니다.

     즉, **그 visual feature**라는 거를 얼마나 **지금까지 잘 발전되어온 LLM잘 adapt 시킬 것인가**가 아마 핵심이라고 파악.

     그게 본 직무에서 추구하는 바라고 느꼈다.

<br>

# 2. VLM의 네 가지 발전 방향

1. 아키텍처
2. pretraining task
3. 다양한 downstream task, 특히 dense prediction task들에 어떻게 

<br>

여기서 아마 1.아키텍처가

## (1) 아키텍처

- 거시적) multi -> uni

- 흐름 2) fusion

  - 대부분 LLM decoder 혹은 encoder-decoder기반을 통과하게 될 것인데, 어떻게 다른 modality의 데이터를 LLM에 align될 수 있는 토큰으로 만들지

  - 간단한거는 concat도 있겠지만

    - (1) cross attention으로써 주입

    - (2) 가벼운 adapter를 사용

    - (3) Q-former, perceiver, adapter

    - 최근에 좀 인상깊었던 것은: **EVE였나**, Fuyu였나

      (밑에 글 참고)

## (2) pretraining task

- CL -> ITM, MLM, MIM, captioning loss, image-text-matching,



## (3) knowledge distillation, transfer learning

- coarse뿐만 아니라 dense



## (4) 데이터

- CLIP이 noisy-web data로써 학습이 되었으니까
  - 더 많은 데이터
  - 혹은 더 적은 데이터를 efficient하게
  - 혹은 스스로 정제를 해가며 filtering
  - 혹은 caption을 스스로 generate해서 둘 다 사용하던지

<br>

# 3. EVE, Fuyu

***본질은 같지만 발현되는 것만 다를 뿐***

- LCM (Large Concept Model) 메타에서 작년에 공개한 것에서 영감

<br>

***이 말에 전적으로 동의.***

<br>

사실 우리눈에 보이는 현상들은, **"(1) 어떠한 본질 + (2) 어떠한 형태"**로써 발현

- 과연 딥러닝이 포착하기 위해서 노력하는 것이 무엇일까?를 생각해봤을때

- 피상적인 (2)보다는 (1)일 것.

- 이는 곧 파라미터 수, 및 해당 특징을 잘 포착하기 위한 모듈.알고리즘을 고안하는것과 직결

  즉, 생각보다 발현되는 양상/형태/모달리티는 그렇게 어려운 일이 아니다

  ( 물론 어렵다는게, 저희가 연구하는데에 있어서 어렵긴 하겠지만, 이제 모델의 입장에서 봤을 때 )

  우리가 cross attention이든, adapter등 다양한 방법을 쓰는 것은, 그것을 효과적으로 fuse하는 법을 아직 모르기 때문에 복잡한 형태로 발현되는 sub-optimal의 결과물이라고 생각이 듭니다 (물론, 제가 많이 해보지는 않아서....)

$\rightarrow$ ***Modality별 최소한의 방법을 통한 adaptation***

<br>

Q-former, perceiver, adapter등 modality를 align하는 것에는 생각보다 큰 노력이 필요하지 않을 거라는 가정 하에 연구를 해보고 싶었음.

<br>

# 4. 시계열에서도 이러한 연구

예를 들자면,

오래된 논문

- Domain adaptation (LN param을 domain 별로)

<br>

Channel normalization

- 채널이라는 개념에 대해 설명들지마ㅕㄴ,,

- Layer norm의 affine transformation
- Domain adaptation (LN param을 domain 별로)







ICML에 submission한것도 TS라 사실 본질적으로 다르지만,

- 영감을 받았던거는 조금 오래 된 연구
- 2010년대 후반에나왔떤걸로 기억: domain adaptation -> domain간에 서로 다른 layer norm 파라미터 (affine transformation)
- time series에도 채널이라는 개념이 존재
  
  



LCM (Large Concept Model)



다른 도메인이라 정량적으로,정성적으로 비교할 수 없지만,

- 생각보다 



시계열의 다른 채널

- 다른 본질 -> 같은 형식으로 발현

멀티모달

- 같은 본질 -> 다른 형식으로 발현



# 자기 고백

## (1) 고백

제 논문쓰고 conference에 내던 main 연구분야는 VLM이 아니었다보니,

어쩌면 그러면 프로젝트, 대회에서 구르면서 고생해본 경험이 주는 그 가치

- 그것도 많이 중요하다 생각하는데

혹시 그 부분이 우려가되시는 측면이 있으시면, 



## (2) 개인적인 바램

사실 딥러닝 판에서, 하나의 도메인 (혹은 모달리티)를 깊게 파서 그쪽의 전문가가 되는 것도 물론 대단하고 좋은 거라고 생각하지만,

한편으로는

- 빠르게 바뀌고 
- 다양한 모달리티들이 서로 유기적으로 연관지어서 발전되어오고 있는 

최근의 트렌드를 감안해보면, 

다양한 분야에 대해서 꾸준히 팔로업하면서, 필요에 따라서 modality transfer를 할 수도 있는 것이 좋을 것 같다고 생각이 들어서,

저 개인적으로도 이번 인턴은 좋은 성장의 기회가 될 것 같기도 합니다.



## (3) 



# 끝



Superpod 자원을 활용한 Multimodal Backbone의 Vision 능력 추가를 위한 모델 학습 관련 과제

Superpod을 통한 Backbone 생산을 위한 FW 도입과 관련된 Engineering





**Superpod**는 NVIDIA의 **DGX SuperPOD**를 의미할 가능성이 크다.
 이는 **대규모 AI 및 HPC(고성능 컴퓨팅) 워크로드를 처리하기 위한 데이터센터급 GPU 클러스터**다.

### **SuperPOD 특징**

- **DGX 시스템 기반**: 여러 대의 **NVIDIA DGX** 서버(DGX H100, DGX A100 등)로 구성
- **NVLink 및 NVSwitch 지원**: GPU 간 초고속 통신 가능
- **멀티모달 모델 학습에 최적화**: Vision과 Language 등 다양한 데이터 유형을 병렬로 처리

### **Multimodal Backbone의 Vision 능력 추가 관련**

SuperPOD를 활용하면 대규모 데이터를 처리하는 **멀티모달(예: 텍스트+이미지) 백본 모델의 비전 모듈**을 학습하는 것이 가능하다.
 즉, SuperPOD의 강력한 GPU 연산 능력을 활용해 **이미지 인식, 시각적 표현 학습, VLM(Visual Language Model) 최적화** 등의 과제를 수행할 수 있다.



**SuperPOD 기반 대규모 학습 환경 구축**

- NVIDIA DGX SuperPOD에서 멀티모달 AI 백본(Backbone)을 학습할 수 있도록 **분산 학습 시스템** 설계
- **GPU, NVLink, InfiniBand** 등을 활용한 고속 데이터 전송 및 병렬 처리 최적화

**FW(Framework) 도입 및 최적화**

- PyTorch, DeepSpeed, Megatron-LM 등 **대규모 모델 학습용 프레임워크**를 도입
- SuperPOD 환경에서의 **효율적인 메모리 관리, Mixed Precision 활용** 등 최적화
- 모델 체크포인트 관리 및 장애 대응

**멀티모달 AI 백본(Backbone) 설계 및 실험**

- VLM(Visual Language Model) 등의 **Vision 능력을 강화하는 AI 모델 개발**
- 학습 데이터 파이프라인 구축 및 효율적인 데이터 로딩 방식 연구

**엔지니어링 및 유지보수**

- SuperPOD 기반 학습 인프라의 **성능 모니터링 및 유지보수**
- **컨테이너화(Docker, Kubernetes)**를 활용한 배포 및 확장성 확보



![image-20250307200931192](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250307200931192.png)