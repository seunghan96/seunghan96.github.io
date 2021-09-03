

# 1. `linear`

```python
def linear(in_dim, output_dim, bias=True, dropout: int = None):
    layer = nn.Linear(in_dim, output_dim, bias=bias)
    if dropout is not None:
        return nn.Sequential(nn.Dropout(dropout), layer)
    else:
        return layer
```

<br>

# 2. `linspace`

`len_back` : Length of backcast

`len_fore` : Length of 

```python
def linspace(len_back: int, len_fore : int, centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if centered:
        norm = max(len_back, len_fore )
        start = -len_back
        stop = len_fore  - 1
    else:
        norm = len_back + len_fore 
        start = 0
        stop = len_back + len_fore  - 1
    lin_space = np.linspace(start/norm, stop/norm, 
                            len_back+len_fore,dtype=np.float32)
    b_ls = lin_space[:len_back]
    f_ls = lin_space[len_back:]
    return b_ls, f_ls
```

```python
b_ls,f_ls=linspace(2,3,centered=True)

print(b_ls)
print(f_ls)
#-------------------------------------#
[-0.6666667  -0.33333334]
[0.         0.33333334 0.6666667 ]
```

```python
b_ls,f_ls=linspace(2,3,centered=False)

print(b_ls)
print(f_ls)
#-------------------------------------#
[0.  0.2]
[0.4 0.6 0.8]
```

<br>

# [ `NBEATS` ]

# 3-1. `NBEATSBlock`

```python
class NBEATSBlock(nn.Module):
    def __init__(
        self,hidden_dim,thetas_dim,
        num_block_layers=4,
        len_back=10,len_fore =5,
        share_thetas=False,dropout=0.1):
        
        #------------ 1) 기본 Setting ---------------------------------#
        super().__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.len_back = len_back
        self.len_fore  = len_fore 
        self.share_thetas = share_thetas
        #--------------------------------------------------------------#
        
		#------------ 2) Layer들 구성하기 --------------------------------#
        fc_stack = [ nn.Linear(len_back, hidden_dim),nn.ReLU() ]
        for _ in range(num_block_layers - 1):
            fc_stack.extend([linear(hidden_dim, hidden_dim, dropout=dropout), nn.ReLU()])
        self.fc = nn.Sequential(*fc_stack)
        #--------------------------------------------------------------#
        
        #------------ 3) Forward & Backward의 theta 공유 여부 -----------#
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(hidden_dim, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(hidden_dim, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(hidden_dim, thetas_dim, bias=False)
        #--------------------------------------------------------------#
        
    def forward(self, x):
        return self.fc(x)
```

<br>

# 3-2. `NBEATSSeasonalBlock`

- `s1_Wf` : forward season 1
- `s2_Wf` : forward season 2
- `b1_Wf` : backward season 1
- `b2_Wf` : backward season 2

```python
def season_param(freq,line):
    param = [np.cos(2 * np.pi * i * line) for i in freq]
    param = torch.tensor(param,dtype=torch.float32)
    return  
```

```python
def trend_param(thetas_dim,line):
    param = [line ** i for i in range(thetas_dim)]
    param = torch.tensor(param, dtype=torch.float32)
    return  
```



```python
class NBEATSSeasonalBlock(NBEATSBlock):
    def __init__(
        self,hidden_dim,thetas_dim=None,
        num_block_layers=4,
        len_back=10,len_fore =5,
        nb_harmonics=None,
        min_period=1,dropout=0.1):
        if nb_harmonics:
            thetas_dim = nb_harmonics
        else:
            thetas_dim = len_fore 
        self.min_period = min_period
        super().__init__(
            hidden_dim=hidden_dim,thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            len_back=len_back,len_fore =len_fore ,
            share_thetas=True,dropout=dropout)
        
		#------------------- Seasonal Parameter ( 학습 대상은 X ) ---------------#
        p1, p2 = (thetas_dim//2,thetas_dim//2) if thetas_dim%2 == 0 else (thetas_dim//2, thetas_dim//2 + 1)
        line_back, line_fore = linspace(len_back, len_fore , centered=True)
        
        s1_Wb = season_param(self.get_freq(p1),line_back) # H/2-1
        s2_Wb = season_param(self.get_freq(p2),line_back)
        s1_Wf = season_param(self.get_freq(p1),line_fore) # H/2-1
        s2_Wf = season_param(self.get_freq(p2),line_fore)
        self.register_buffer("S_backcast", torch.cat([s1_Wb, s2_Wb]))
        self.register_buffer("S_forecast", torch.cat([s1_Wf, s2_Wf]))

        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        #------------------ 1) NBEATS block 통과 ---------------------------#
        ### (아직 Forward / Backward 안 나뉨)
        x = super().forward(x)
        #------------------------------------------------------------------#
        
        #--------------- 2) Forecast & Backcast 생성 (with seasonal) -------------------#
        amplitudes_backward = self.theta_b_fc(x)
        amplitudes_forward = self.theta_f_fc(x)
        backcast = amplitudes_backward.mm(self.S_backcast)
        forecast = amplitudes_forward.mm(self.S_forecast)
        #-------------------------------------------------------------------------------#
        return backcast, forecast

    def get_freq(self, n):
        return np.linspace(0, (self.len_back + self.len_fore ) / self.min_period, n)
```

<br>

# 3-3. `NBEATSTrendBlock`

```python
class NBEATSTrendBlock(NBEATSBlock):
    def __init__(
        self,hidden_dim,thetas_dim,
        num_block_layers=4,
        len_back=10,len_fore =5,dropout=0.1):
        super().__init__(
            hidden_dim=hidden_dim,thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            len_back=len_back,len_fore =len_fore ,
            share_thetas=True,dropout=dropout)

        #------------------- Trend Parameter ( 학습 대상은 X ) ---------------#
        line_back, line_fore = linspace(len_back, len_fore , centered=True)
        norm = np.sqrt(len_fore  / thetas_dim)  # ensure range of predictions is comparable to input
        
        trend_Wf = trend_param(thetas_dim,line_fore)
        trend_Wb = trend_param(thetas_dim,line_back)
        self.register_buffer("T_forecast", trend_Wf * norm)
        self.register_buffer("T_backcast", trend_Wb * norm)        

        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        #------------------ 1) NBEATS block 통과 ---------------------------#
        ### (아직 Forward / Backward 안 나뉨)
        x = super().forward(x)
        #------------------------------------------------------------------#
        
        #--------------- 2) Forecast & Backcast 생성 (with trned) -------------------#
        backcast = self.theta_b_fc(x).mm(self.T_backcast)
        forecast = self.theta_f_fc(x).mm(self.T_forecast)
        #------------------------------------------------------------------------------#
        return backcast, forecast
```

<br>

# 3-4. `NBEATSGenericBlock`

```python
class NBEATSGenericBlock(NBEATSBlock):
    def __init__(
        self,hidden_dim,thetas_dim,
        num_block_layers=4,
        len_back=10,len_fore =5,dropout=0.1):
        super().__init__(
            hidden_dim=hidden_dim,
            thetas_dim=thetas_dim,
            num_block_layers=num_block_layers,
            len_back=len_back,
            len_fore =len_fore ,
            dropout=dropout,
        )
        
		#------------ 일반적인 Backward & Forward Parameter ( 학습 대상 O ) -----------#
        self.backcast_fc = nn.Linear(thetas_dim, len_back)
        self.forecast_fc = nn.Linear(thetas_dim, len_fore )

    def forward(self, x):
        #------------------ 1) NBEATS block 통과 ---------------------------#
        ### (아직 Forward / Backward 안 나뉨)
        x = super().forward(x)
        #------------------------------------------------------------------#

        #--------------- 2) Forecast & Backcast 생성 -------------------#
        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
		#------------------------------------------------------------------#
        return self.backcast_fc(theta_b), self.forecast_fc(theta_f)
```

