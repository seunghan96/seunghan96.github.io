---
title: (code) Triplet Loss for Time Series
categories: [CL, TS]
tags: []
excerpt:
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Triplet Loss for Time Series

```
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Advances in neural information processing systems 32 (2019).
```

<br>

references :

- https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
- https://arxiv.org/pdf/1901.10738.pdf

<br>

# 1. Triplet Loss ( ver 1 )

length of **positive sample** = length of **positive sample**

<br>

Notation :

- `B` : batch size
- `K` : number of negative samples
- `neg_samples` : negative samples
- Length
  - `T` : total length of TS
  - `length` : maximum length of subseries
  - `        length_pos_neg` : length of pos/neg subseries
  - `length_anchor` : length of anchor
- Indices
  - `start_anchor` : START index of anchor
  - `start_pos`: START index of positive sample	
  - `end_pos` : END index ~
  - `start_neg` : START index of negative sample
- Embedding vectors
  - `Z_anchor` : size = (`B`, `dim_Z`)
  - `Z_pos` : size = (`B`, `dim_Z`)
  - `Z_neg` : size = (`B`, `dim_Z`) ……..  make it $$K$$ Times

```python
class TripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length, K, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.K = K 
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        B = batch.size(0) # batch size
        T = train.size(0) # total length
        length = min(self.compared_length, train.size(2)) # MAX length of subseries

        neg_samples = np.random.choice(T, size=(self.K, B))
        neg_samples = torch.LongTensor(neg_samples)

        length_pos_neg = np.random.randint(1, length+1) # length of subseries

        length_anchor = np.random.randint(length_pos_neg, length + 1) 
        start_anchor = np.random.randint(0, length - length_anchor + 1, B)

        # (1,B) pos & (K,B) neg
        start_pos = np.random.randint(0, length_anchor - length_pos_neg + 1, B)  
        start_pos = start_anchor + start_pos # positive sample START INDEX
        end_pos = start_pos + length_pos_neg # positive sample END INDEX

        start_neg = np.random.randint(0, length - length_pos_neg + 1, (self.K, B)) # negative sample START INDEX

        Z_anchor = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                start_anchor[j]: start_anchor[j] + length_anchor
            ] for j in range(B)]))

        Z_pos = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_pos[j] - length_pos_neg: end_pos[j]
            ] for j in range(B)])) 

        dim_Z = Z_anchor.size(1)
        
        #===============================================================#
        # Positive loss 
        #===============================================================#
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            Z_anchor.view(B, 1, dim_Z),
            Z_pos.view(B, dim_Z, 1)
        )))

        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del Z_pos
            torch.cuda.empty_cache()
        
        #===============================================================#
        # Negative loss 
        #===============================================================#
        multiplicative_ratio = self.negative_penalty / self.K
        
        for i in range(self.K):
            Z_neg = encoder(
                torch.cat([train[neg_samples[i, j]: neg_samples[i, j] + 1][
                    :, :,
                    start_neg[i, j]:
                    start_neg[i, j] + length_pos_neg
                ] for j in range(B)]))
            
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    Z_anchor.view(B, 1, dim_Z),
                    Z_neg.view(B, dim_Z, 1))))

            if save_memory and i != self.K - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del Z_neg
                torch.cuda.empty_cache()

        return loss
```



# 2. Triplet Loss ( ver 2 )

length of **positive sample** $$\neq$$ length of **positive sample**

- lengths **among** positive samples **are also different**
- lengths **among** negative ~

<br>

Notation :

- `B` : batch size
- `K` : number of negative samples
- `neg_samples` : negative samples
- Length
  - `T` : total length of TS
  - `length` : maximum length of subseries
  - `        lengths_pos` : lengths of pos subseries
  - `        length_neg` : length of neg subseries
  - `length_anchor` : length of anchor
- Indices
  - `start_anchor` : START index of anchor
  - `start_pos`: START index of positive sample	
  - `end_pos` : END index ~
  - `start_neg` : START index of negative sample
- Embedding vectors
  - `Z_anchor` : size = (`B`, `dim_Z`)
  - `Z_pos` : size = (`B`, `dim_Z`)
  - `Z_neg` : size = (`B`, `dim_Z`) ……..  make it $$K$$ Times

```python
class TripletLossVaryingLength(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length, K, negative_penalty):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.K = K
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        B = batch.size(0)
        T = train.size(0)
        max_length = train.size(2)

        neg_samples = np.random.choice(T, size=(self.K, B))
        neg_samples = torch.LongTensor(neg_samples)

        with torch.no_grad():
            lengths_batch = max_length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()
            lengths_samples = np.empty((self.K, B), dtype=int)
            for i in range(self.K):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[neg_samples[i], 0]), 1
                ).data.cpu().numpy()

        #==================================================================#
        # lengths_anchor : (B)
        # lengths_pos : (B)
        # lengths_neg : (K,B)
        #------------------------------------------------------------------#
        lengths_pos = np.empty(B, dtype=int)
        lengths_neg = np.empty((self.K, B), dtype=int)
        for j in range(B):
            lengths_pos[j] = np.random.randint(
                1, min(self.compared_length, lengths_batch[j]) + 1)
            for i in range(self.K):
                lengths_neg[i, j] = np.random.randint(
                    1,min(self.compared_length, lengths_samples[i, j]) + 1)
        length_anchor = np.array([np.random.randint(
            lengths_pos[j], min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(B)])  
        
        #==================================================================#
        # start_anchor : (B)
        # start_pos : (B)
        # start_neg : (K,B)
        #------------------------------------------------------------------#
        start_anchor = np.array([np.random.randint(
            0, lengths_batch[j] - length_anchor[j] + 1
        ) for j in range(B)]) 

        start_pos = np.array([np.random.randint(
            0, high=length_anchor[j] - lengths_pos[j] + 1
        ) for j in range(B)])
        end_pos = start_pos + lengths_pos

        start_neg = np.array([[np.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(B)] for i in range(self.K)])

        #==================================================================#
        # Z_anchor : (B, dim_Z)
        # Z_pos : (B, dim_Z)
        ## Z_neg : BELOW
        #------------------------------------------------------------------#
        Z_anchor = torch.cat([encoder(
            batch[
                j: j + 1, :,
                start_anchor[j]: start_anchor[j] + length_anchor[j]
            ]
        ) for j in range(B)])  

        Z_pos = torch.cat([encoder(
            batch[
                j: j + 1, :,
                end_pos[j] - lengths_pos[j]: end_pos[j]
            ]
        ) for j in range(B)])  

        dim_Z = Z_anchor.size(1)
        
        #===============================================================#
        # Positive loss 
        #===============================================================#
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            Z_anchor.view(B, 1, dim_Z),
            Z_pos.view(B, dim_Z, 1)
        )))

        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del Z_pos
            torch.cuda.empty_cache()

        #===============================================================#
        # Negative loss 
        #===============================================================#
        multiplicative_ratio = self.negative_penalty / self.K
        for i in range(self.K):
            Z_neg = torch.cat([encoder(
                train[neg_samples[i, j]: neg_samples[i, j] + 1][
                    :, :,
                    start_neg[i, j]:
                    start_neg[i, j] + lengths_neg[i, j]]
            ) for j in range(B)])
            
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    Z_anchor.view(B, 1, dim_Z),
                    Z_neg.view(B, dim_Z, 1))))

            if save_memory and i != self.K - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del Z_neg
                torch.cuda.empty_cache()

        return loss
```

