---
title: (code review 4) BERT for ABSA (Aspect Based Sentiment Analysis)
categories: [NLP,HBERT]
tags: [NLP,ABSA,HBM]
excerpt: AE (Aspect Extraction), ASC (Aspect Sentiment Classification)
---

# Code Review for "BERT for ABSA"

Reference :

- https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/absa_data_utils.py
- https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/run_ae.py
- https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/run_asc.py
- https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/run_ae.py
- https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/run_asc.py

<br>

( AE vs ASC )

Aspect Extraction (AE): 

- given a **(1) review sentence** ("The retina display is great."), find aspects("retina display");

Aspect Sentiment Classification (ASC):

- given an **(1) aspect** ("retina display") and a **(2) review sentence** ("The retina display is great."), detect the polarity of that aspect (positive).

<br>

# 1. BERT for ASC(Aspect Sentiment Classification)

`bert_forward` :

- 3개의 embedding, 12개의 encoder, pooling 다 거친 결과물

`adv_attack` :

- gradient 방향 noise 껴서 adversarial attack 데이터 생성하기 ( = perturbed sentence )

`adversarial_loss` :

- perturbed sentence ( + mask )를 input으로 넣어서 Cross Entropy loss 계산

`forward` :

- 과정 1) `bert_forward`
- 과정 2) prediction하기 ( = dropout & classifier로 logit값 생성 )
- 과정 3) 
  - (case 1) Y값 있는 경우 
    - training 과정 X 경우 : loss 계산
    - training 과정 O 경우 : loss 계산 & adversarial loss 계산
  - (case 2) Y값 없는 경우
    - logit값 반환

```python
class BertForABSA(BertModel):
    def __init__(self, config, num_classes=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_classes = num_classes 
        self.epsilon = epsilon
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        pooled, embedded = self.bert_forward(input_ids, token_type_ids, 
                                             attention_mask=attention_mask, 
                                             output_all_encoded_layers=False)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        # Case 1) Y값 (O) 경우
        if labels is not None:
            loss_ = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            
            # (training 모드)
            if pooled.requires_grad: 
                perturbed_snt = self.adv_attack(embedded, loss_)
                perturbed_snt = self.replace_cls_token(embedded, perturbed_snt)
                adv_loss = self.adversarial_loss(perturbed_snt, attention_mask, labels)
                return loss_, adv_loss
            return loss_
        
        # Case 2) Y값 (X) 경우
        else:
            return logits

    #--------- Adversarial Attack 데이터 생성 ------------#
    def adv_attack(self, emb, loss):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_snt = emb + self.epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_snt
    #---------------------------------------------------#
    def replace_cls_token(self, emb, perturbed):
        condition=torch.zeros_like(emb)
        condition[:, 0, :] = 1
        perturbed_snt = torch.where(condition.byte(), emb, perturbed)
        return perturbed_snt
    
    #--------BERT 통한 최종 Embedding 결과----------------#
    def bert_forward(self, input_ids, token_type_ids=None, 
                            attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        mask3d = attention_mask.unsqueeze(1).unsqueeze(2)
        mask3d = mask3d.to(dtype=next(self.parameters()).dtype) 
        mask3d = (1.0 - mask3d) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, mask3d,output_all_encoded_layers=output_all_encoded_layers)
        pooled = self.pooler(encoded_layers[-1])
        return pooled, embedding_output
    #---------------------------------------------------#

    def adversarial_loss(self, perturbed, attention_mask, labels):
        #------ (1) mask 생성 --------#
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask3d = attention_mask.unsqueeze(1).unsqueeze(2)
        mask3d = mask3d.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        mask3d = (1.0 - mask3d) * -10000.0
        
        #------ (2) Encoding 하기--------#
        encoded_layers = self.encoder(perturbed, mask3d,output_all_encoded_layers=False)
        layers_wo_last = self.pooler(encoded_layers[-1])
        layers_wo_last = self.dropout(layers_wo_last)
        
        #------ (3) Predicton & Loss 계산--------#
        logits = self.classifier(layers_wo_last)
        adv_loss = self.loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
        return adv_loss
```

<br>

# 2. BERT for AE(Aspect Extraction)

## [ AE & ASC 차이점 ]

### 차이점 1) 목표

- AE : **모든 단어별로** aspect 여부 check
- ASC : **aspect 단어만**을 대상으로 감정 분석

<br>

### 차이점 2) tokenize

- AE : **sub-word** tokenization
- ASC : **word** tokenization

```python
for (ex_index, example) in enumerate(examples):
    if mode!="ae":
        tokens_a = tokenizer.tokenize(example.text_a)
        
    else: #only do subword tokenization.
        tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )
```

***( AE의 "subword_tokenize"를 자세히 들여다보면...  )***

```python
class ABSATokenizer(BertTokenizer):     
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map
```

<br>

### 차이점 3) bert_forward의 결과로 나오는 output

- (1) ASC : **pooled된 output** & embedded
- (2) AE : **sequence output** & embedded

<br>

```python
class BertForABSA(BertModel):
    def __init__(self, config, num_classes=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, embedded = self.bert_forward(input_ids, 
                                                token_type_ids, 
                                                attention_mask, 
                                                output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss_ = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            if sequence_output.requires_grad: 
                perturbed_sentence = self.adv_attack(embedded, loss_, self.epsilon)
                adv_loss = self.adversarial_loss(perturbed_sentence, attention_mask, labels)
                return _loss, adv_loss
            return _loss
        else:
            return logits

    def adv_attack(self, emb, loss, epsilon):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_sentence

    def adversarial_loss(self, perturbed, attention_mask, labels):
        #------ (1) mask 생성 --------#
        mask3d = attention_mask.unsqueeze(1).unsqueeze(2)
        mask3d = mask3d.to(dtype=next(self.parameters()).dtype)
        mask3d = (1.0 - mask3d) * -10000.0
        
        #------ (2) Encoding 하기--------#
        encoded_layers = self.encoder(perturbed, mask3d,output_all_encoded_layers=False)
        encoded_layers_last = encoded_layers[-1]
        encoded_layers_last = self.dropout(encoded_layers_last)
        
        #------ (3) Predicton & Loss 계산--------#
        #### 유의점 : (ingore_index=-1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1) 
        logits = self.classifier(encoded_layers_last)
        adv_loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
        return adv_loss

    #--------BERT 통한 최종 Embedding 결과----------------#
    def bert_forward(self, input_ids, token_type_ids=None, 
                        attention_mask=None, output_all_encoded_layers=False):
        #-------------------------------------#
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        #-------------------------------------#
        mask3d = attention_mask.unsqueeze(1).unsqueeze(2)
        mask3d = mask3d.to(dtype=next(self.parameters()).dtype) 
        mask3d = (1.0 - mask3d) * -10000.0
        #-------------------------------------#
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, mask3d, output_all_encoded_layers=output_all_encoded_layers)
        layers_wo_last = encoded_layers[-1]
        return layers_wo_last, embedding_output
```

