```
import torch
import torch.nn as nn
import torch.nn.functional as F
```

# 1. MODEL

## (1) InformerStack

- `enc_in` : encoder의 input dimension
- `dec_in` : decoder의 input dimension
- `c_out` : number of channel output
- `label_len` : length of label output 
- `out_len` : 예측하고자 하는 (time)구간의 길이
- `d_model` : embedding할 dimension
- `d_ff` : Encoder layer / Decoder layer에 있는 첫 번째 Conv1d의 output dimension
- `e_layers` : 1개의 encoder에 쌓을 encoder layer의 개수
  - ex) [3,2,1] :
    - 1번째 encoder에는 : encoder layer 3개
    - 2번째 encoder에는 : encoder layer 2개
    - 3번째 encoder에는 : encoder layer 1개
- `d_layers` : 1개의 decoder에 쌓을 decoder layer의 개수
- Attention 관련 hyper 파라미터들
  - `factor` : 
  - `n_heads` : Attention head의 개수
- `dropout` : dropout rate
- `attn` : ( 'prob' 일 경우, **ProbSparse attention** 사용 )
- `embed` : temporal embedding 시
  - 1) `FixedEmbedding` 할 지 
  - 2) `nn.Embedding` 할 지
- `freq` : time feature embedding 시의 frequency
- `activation`  : activation function
- `output_attention` : attention score들을 반환할 지
- `distil` : distillation 할 지 말지

<br>

```python
class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
```

<br>

## (2) Informer

```python
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
```

<br>

# 2. ENCODER

EncoderStack

- Encoder
  - Encoder Layer
  - Conv Layer



## (1) `EncoderStack` : encoder 모음

```python
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders) # Encoder의 모음
        self.inp_lens = inp_lens # input의 길이들 ( 0,1,2,....)

    def forward(self, x, attn_mask=None):
        # 3차원의 X : [Batch, Length, Dimension]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
```

<br>

## (2) `Encoder`

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # 3차원의 X : [Batch, Length, Dimension]
        attns = []
        
        #-----------(1) Conv layer 사용하는 경우--------------#
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
            
        #-----------(2) Conv layer 사용 안하는 경우--------------#
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        #-----------(3) Norm layer 사용하는 경우--------------#
        if self.norm is not None:
            x = self.norm(x)

        return x, attns
```

<br>

## (3) Encoder Layer



```python
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        
        #--------(1) Attention을 거치고 ------------#
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask)
        
        #--------(2) Residual Connection & Layer Normalization 1 ------------#
        y = x = self.norm1(x + self.dropout(new_x))
        
        #--------(3) Convolution layer x 2 통과 ------------#
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

		#--------(4) Residual Connection & Layer Normalization 2 ------------#
        y = self.norm2(x+y)

        return self.norm2(x+y), attn
```

<br>

## (4) Conv Layer

```python
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
```

<br>

# DECODER

## (1) `Decoder`

```python
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
```

<br>

## (2) `DecoderLayer`

```python
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        #----------(1) Self Attention -------------#
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        
        #----------(2) Cross Attention -------------#
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        y = x = self.norm2(x)
        
        #-----------(3) Convolutional Layer ----------#
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
		y = self.norm3(x+y)
        return y
```

<br>

# 4. EMBEDDING

**Data** Embedding

- 1) **Positional** Embedding

- 2) **Token** Embedding

- 3-1) **Temporal** Embedding

  - **Fixed** Embedding 사용 O / X

- 3-2) **Time Feature** Embedding

  ( 3-1 or 3-2 중 선택 )

<br>

## (1) Positional Embedding ( 학습 X )

positional encoding은 학습 대상 X 

- `d_model` : Positional embedding을 할 dimension
- `max_len` : 문장의 최대 길이

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position_term = torch.arange(0, max_len).float().unsqueeze(1)
        divide_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position_term * divide_term)
        pe[:, 1::2] = torch.cos(position_term * divide_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
```

<br>

## (2) Token Embedding ( 학습 O )

Token encoding은 학습 대상 O

( 1D convolution으로 임베딩한다 )

- `c_in` 
  - encoder의 경우 : encoder의 input dimension
  - decoder의 경우 : decoder의 input dimension
- `d_model` : Positional embedding을 할 dimension

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
```

<br>

## (3) Fixed Embedding ( 학습 X )

- "positional encoding"과 유사 

- embedding input dimension은 

  - minute_size
  - hour_size
  - ....

  다양할 수 있음

```python
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
```

<br>

## (4) Temporal Embedding ( 학습 O/X )

- 월/일/요일/시/분을 각각 encoding한 이후, summation
- 임베딩 할 때..
  - 1) `FixedEmbedding` 사용 시 : 학습 X
  - 2) `nn.Embedding` 사용 시 : 학습 O

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        x_minute = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        x_hour = self.hour_embed(x[:,:,3])
        x_weekday = self.weekday_embed(x[:,:,2])
        x_day = self.day_embed(x[:,:,1])
        x_month = self.month_embed(x[:,:,0])
        return x_hour + x_weekday + x_day + x_month + x_minute
```

<br>

## (5) Time Feature Embedding ( 학습 O )

- `freq` : Frequency for time features encoding 

  (s : secondly, t : minutely, h : hourly, d : daily, b : business days, w : weekly, m : monthly)

```python
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)
```

<br>

## (6) Data Embedding

- Token Embedding
- Positional Embedding
- Temporal Embedding ( or Time Feature Embedding)

위 세 요소를 summation하여 반환

```python
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
```

<br>

# 5. Attention

(일반적인) `FullAttention`과 `PropAttention` 중에 선택!

`AttentionLayer`

- 위에서 선택한 Attention을 사용하여 layer를 만든 것

<br>

## (기타) Mask

`torch.triu` : upper triangle

```python
a = torch.randn(3, 3)
torch.triu(a, diagonal=1)
--------------------------------------------
tensor([[ 0.0000,  0.5207,  2.0049],
        [ 0.0000,  0.0000,  0.6602],
        [ 0.0000,  0.0000,  0.0000]])
```



### a) TriangularCausalMask

- 뒤에 cheating 방지를 위한 mask

- `torch.triu` : upper triangle

  ```
  a = torch.randn(3, 3)
  torch.triu(a, diagonal=1)
  --------------------------------------------
  tensor([[ 0.0000,  0.5207,  2.0049],
          [ 0.0000,  0.0000,  0.6602],
          [ 0.0000,  0.0000,  0.0000]])
  ```

```python
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
```

<br>

### b) ProbMask

```python
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
```



## (1) AttentionLayer

- `attention` : 사용할 attention의 종류
  - 후보 1) `FullAttention`
  - 후보 2) `ProbAttention`
- `d_model` : (Attention의 input으로 들어 갈) 이전에 embedding해서 나왔던 dimension
- `n_heads` : Attention head의 개수

```python
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_K=None, d_V=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_K = d_K or (d_model//n_heads)
        d_V = d_V or (d_model//n_heads)
        # (1) inner_attention : 사용할 attention 종류
        self.inner_attention = attention
        
        # (2) Wq,Wk,Wv : Q,K,V를 만들 projection weight matrix
        ## ( n_head=8개를 병렬적으로 동시에 수행한다 )
        self.Wq = nn.Linear(d_model, d_K * n_heads)
        self.Wk = nn.Linear(d_model, d_K * n_heads)
        self.Wv = nn.Linear(d_model, d_V * n_heads)
        self.Wo = nn.Linear(d_V * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, Q, K, V, attn_mask):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        H = self.n_heads
        
        # (1) Q,K,V 계산
        Q = self.Wq(Q).view(B, L, H, -1)
        K = self.Wk(K).view(B, S, H, -1)
        V = self.Wv(V).view(B, S, H, -1)
        
        # (2) Attention 수행 ( via Q,K,V )
        out, attn = self.inner_attention(Q,K,V,attn_mask)
		
        # (3) Output x Weight로 최종 출력값 계산
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
		out = self.Wo(out)
        return out, attn
```

<br>

## (2) FullAttention

- `scale` : softmax에 들어갈 값에 곱하는 scale

```python
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, Q, K, V, attn_mask):
        B, L, H, E = Q.shape
        _, S, _, D = V.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=Q.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, V)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
```



<br>

## (3) ProbAttention

```python
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                 attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # [ B(=Batch Size), H(=Head의 개수), L(=Length), D(=Dimension) ]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # (1) sampled Q_K 계산
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # (2) Top_k query with sparisty measurement 찾기
        ## torch.topk : (index=0) 값 & (index=1) 인덱스
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        ## (3) Top_K query만을 사용하여 Q*K 계산
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        ## (4) Q*K와 Top k 인덱스 반환
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2) # V_sum = V.sum(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, Q, K, V, attn_mask):
        B, L_Q, H, D = Q.shape
        _, L_K, _, _ = K.shape

        Q = Q.transpose(2,1)
        K = K.transpose(2,1)
        V = V.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(Q, K, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
            
        # get the context
        context = self._get_initial_context(V, L_Q)
        
        # update the context with selected top_k Q
        context, attn = self._update_context(context, V, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_K=None, d_V=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_K = d_K or (d_model//n_heads)
        d_V = d_V or (d_model//n_heads)

        self.inner_attention = attention
        self.Wq = nn.Linear(d_model, d_K * n_heads)
        self.Wk = nn.Linear(d_model, d_K * n_heads)
        self.Wv = nn.Linear(d_model, d_V * n_heads)
        self.Wo = nn.Linear(d_V * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, Q, K, V, attn_mask):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        H = self.n_heads

        Q = self.Wq(Q).view(B, L, H, -1)
        K = self.Wk(K).view(B, S, H, -1)
        V = self.Wv(V).view(B, S, H, -1)

        out, attn = self.inner_attention(
            Q,
            K,
            V,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.Wo(out), attn
```

