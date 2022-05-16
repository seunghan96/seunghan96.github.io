# MTHetGNN

## (1) 데이터셋

Exchange-Rate

- 소개 : The exchange rate data from eight countries, including UK, Japan, New Zealand, Canada, Switzerland, Singapore, Australia and China, ranging from 1990 to 2016.
- 모델 : 
  - 주요 모델 : MTHetGNN, DCRNN, Graph WaveNet, EnhanceNet, MLCNN, MTGNN, TEGNN, 
  - 기본 모델 : VAR, RNN-GRU, LSTNET



Traffic 1 :

- 소개 : The traffic highway occupancy rates measured by 862 sensors in San Francisco from 2015 to 2016 by California Department of Transportation.
- 모델 : 
  - 주요 모델 : MTHetGNN, DCRNN, Graph WaveNet, EnhanceNet, MLCNN, MTGNN, TEGNN, 
  - 기본 모델 : VAR, RNN-GRU, LSTNET



Solar-Energy : 

- 소개 : Continuous collected Solar energy data from the National Renewable Energy Laboratory, which contains the solar energy output collected from 137 photovoltaic power plants in Alabama in 2007.

- 모델 : 
  - 주요 모델 : MTHetGNN, DCRNN, Graph WaveNet, EnhanceNet, MLCNN, MTGNN, TEGNN, 
  - 기본 모델 : VAR, RNN-GRU, LSTNET



## (2) 메트릭

- RSE (Relative Squared Error)
- RAE (Relative Absolute Error)
- CORR (Empirical Correlation Coefficient)



## (3) 실험 세팅

- [1] train-val-test : 6/2/2
- [2] hyperparameter search :
  - validation에서, 위 3개의 메트릭에서 최고인 모델 구조 선택 ( with grid search )
- [3] window : 32
- [4] 기타 베이스라인 모델들 구조 :
  - RNN : 
    - hidden layer [10,20,50,100]
    - Drop-out : [0.1, 0.2, 0.3]
  - LSTNet:
    - hidden CNN/RNN layer : [20,50,100,200]
    - length of recurrent skip : 24
  - DCRNN :
    - enc & dec 둘 다 64 unit
  - Graph WaveNet :
    - 16 layers
    - dilation factors 1,2,4,8
  - EnhanceNet :
    - enhanced RNN 사용
    - memory dimension : 16
  - MTGNN :
    - mix-hop propagation depth : 2
    - activation saturation trate of GL layer : 3
  - TEGNN & MTHetGNN
    - hidden GCN : [5,10,15,..,100]

<br>

## (4) Test

forecasting future values ${X_t+3, X_t+6, X_t+12, X_t+24}$,

- Exchange-Rate data : 3,6,12,24 days

- Solar-Energy : 30, 60, 120, 240 minutes
- Traffic data : 3,6,12,24 hours

<br>

# A3T-GCN

## (1) 데이터셋

Traffic 3 : 

- 소개 : taxi trajectory dataset (SZ taxi) in Shenzhen City
  - taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015.
  - 156 major roads of Luohu District
- 모델 : 
  - 주요 모델 : GCN, GCN
  - 기본 모델 : historical average model (HA), ARIMA, SVR, GRU



Traffic 4 :

- 소개 : loop detector dataset (Los loop) in Los Angeles
  - total of 207 sensors along with their traffic speed from Mar. 1 to Mar. 7, 2012

- 모델 : 

  - 주요 모델 : GCN, GCN

  - 기본 모델 : historical average model (HA), ARIMA, SVR, GRU



## (2) 메트릭

- RMSE
- MAE
- Accuracy

- Coefficient of Determination ($R^2$ )
- Explained Variance Score ($var$ )



## (3) 실험 세팅

- [1] train-test : 8/2

- [2] hyperparameter search : X

- [3] window : 32

- [4] learning rate : 0.001

- [5] epoch : 5000

- [6] number of hidden unit

  - traffic 3 : 64
  - traffic 4 : 100

- [4] 기타 베이스라인 모델들 구조 : 언급 없음.

  - historical average model (HA)

  - ARIMA

  - SVR,

  - GCN

  - GRU

    

## (4) Test

- traffic information in the next 15, 30, 45, and 60 min is predicted.

<br>

## (5) 기타 : Perturbation analysis

- to test the robustness

- two types of random noises
  - $N \in\left(0, \sigma^{2}\right)$ ………$\sigma \in(0.2,0.4,0.8,1,2)$
  - $P(\lambda)$ ………. $\lambda \in(1,2,4,8,16)$



# ACGRN

## (1) 데이터셋

Traffic 5 : PeMSD4

- 소개 : 
  - traffic flow data in the San Francisco Bay Area.
  - 307 loop detectors selected within the period from 1/Jan/2018 to 28/Feb/2018.
- 모델 :
  - 주요 사용 모델 : DSANet, DCRNN, STGCN, ASTGCN, STSGCN
  - 기본 모델 : historical average model (HA), VAR, GRU-ED, 



Traffic 6 : PemSD8

- 소개 :
  - collected from 170 loop detectors on the San Bernardino area from 1/Jul/2016 - 31/Aug/2016

- 모델 :
  - 주요 사용 모델 : DSANet, DCRNN, STGCN, ASTGCN, STSGCN
  - 기본 모델 : historical average model (HA), VAR, GRU-ED, 



**Data Preprocess**

- missing values in the datasets are filled by linear interpolation
- aggregated into 5-minute windows, resulting in 288 data points per day. 
- normalize the dataset by standard normalization



## (2) 메트릭

- MAE
- RMSE
- MAPE



## (3) 실험 세팅

[1] train-val-test : 6/2/2

[2] optimizer : Adam

[3] epoch : 100 ( + early stopping patience = 15)

[4] hyperaparemter tuning : 

- validation data로 carefully 했다는게 전부네 ㅋㅋㅋㅋ

[5] 기타 베이스라인 모델들 구조 : 언급 없음.

- historical average model (HA)
- VAR
- GRU-ED
- DSANet
- DCRNN
- STGCN
- ASTGCN
- STSGCN

[6] 기타

- Although our method does not need a pre-defined graph, 
  we use the pre-defined graph for our baselines



## (4) Test

- use 1-hour historical data 

- to predict the next hour’s data

  ( i.e., 5 we organize 12 steps’ historical data as input and the following 12 steps data as output ) 



# Multi-scale temporal features extraction based GCN

## (1) 데이터셋

financial : TAIEX

- 소개 :
  - Taiwan Stock Exchange Capitalization Weighted Stock Index 
  - daily price of Taiwan Stock Index for the five years from 2000 to 2004 ( from yfinance )

- 모델 :
  - 주요 모델 : 
  - 기본 모델 : 

<br>

traffic 7 : Beijing

- 소개 : 

  - minute-level traffic speed data from six roads in Beijing, collected from April 1 to 30, 2016

  - obtained from (https://github.com/BuaaPercy/Traffic-DataSet-of-Beijing-Road)

- 모델 [RMSE, MAE] :
  - 주요 모델 : 제안1, 제안2
  - 기본 모델 : ANN, VAR, ARIMA, LSTM, LSTM-U, Naive
- 모델 [RMSE, MAE] :
  - 주요 모델 : 제안1, 제안2
  - 기본 모델 : KNN, SVM, VAR, RF, LSTM

<br>

medical : Chickenpox Cases in Hungary (CCH) dataset

- 소개 :
  - county-level time series, which describes the number of chickenpox cases reported by Hungarian physicians per week
  - collected for every week between January 2005 and January 2015
  - obtained from (Rozemberczki et al., 2021)
- 모델 :
  - 주요 모델 : 본인, GConvLSTM , GConvGRU, DyGrAE, STGCN, DCRNN, Evolve GCN-O, Evolve GCN-H
  - 기본 모델 : 

<br>

**Data Preprocess**

- min-max normalization

<br>





## (2) 메트릭

- RMSE
- MSE
- MAE

<br>

## (3) 실험 세팅

[1] train-val-test : 

<img src="/Users/LSH/Library/Application Support/typora-user-images/image-20220509114358296.png" alt="image-20220509114358296" style="zoom:30%;" />.

[2] EMD decomposition

-  number of IMFs decomposed from each dimensional sequence may be different

  $\rightarrow$ set to fixed value ( smallest에 맞추기 )

- TCN

  - (fixed) filter size=2, number of blocks=3, each block 2 conv, dilation = 1,2,4
  - (tuned) grid search 사용
    - sliding window : [2,3,4,…,12]
    - GCN의 $h_{hidden}$ : [2,3..6]
    - GCN의 $h_{output}$ : [2,3..6]

- GCN

  - sliding window

<br>

[5] 기타 베이스라인 모델들 구조 [Financial]

- 언급 x

<br>

[6] 기타 베이스라인 모델들 구조 [Traffic]

- ANN : layer=3, neuron=50
- VAR
- ARIMA
- LSTM : ( multi-input for MTS ) : layer=1, neuron=50
- LSTM-U : ( univaraite 각각 모델링 ) : layer=1, neuron=50

<br>

[7] 기타 베이스라인 모델들 구조 [CCH]

- 논문 참고

<br>

## (4) Test

<img src="/Users/LSH/Library/Application Support/typora-user-images/image-20220509114358296.png" alt="image-20220509114358296" style="zoom:30%;" />.

window size는 다양하게 해봐서 다 실험해봄.

<br>

# STSGCN

## (1) 데이터셋

Traffic 5 : PeMSD4

Traffic 7 : PeMSD8

Traffic 8 : PeMSD3

Traffic 9 : PeMSD7

***data preprocess*** : standard normalization

<br>

모델

- 주요 모델 : DCRNN, STGCN, ASTGCN(r), STG2Seq, GraphWaveNet, STSGCN
- 기본 모델 : VAR, SVR, LSTM

<br>

## (2) 메트릭

- MAE
- MAPE
- RMSE

<br>

## (3) 실험 세팅

[1] train-val-test : 6:2:2

- hyperparameters are determined by the model’s performance on the validation datasets.

[2] 그 결과, 최고의 구조 : ( 4개 데이터셋 전부에 대해 )

- 4 STSGCLs,
  - each STSGCM contains 3 graph convolutional operations (64-64-64)

<br>

## (4) Test

- use 1-hour historical data 

- to predict the next hour’s data

  ( i.e., 5 we organize 12 steps’ historical data as input and the following 12 steps data as output ) 

<br>

# STGCN

## (1) 데이터셋

traffic 10 : BJER4

- 소개 :  
  - major areas of east ring No.4 routes in Beijing City by double-loop detectors
  - 12 roads selected
  - traffic data are aggregated every 5 minutes
  - time period used is from 1st July to 31st August, 2014 except the weekends

- 모델 :
  - 주요 모델 : GCGRU, 제안1, 제안2
    - 제안 1: STGCN (Cheb)
    - 제안 2: STGCN (1st)
  - 기본 모델 : HA, LSVR, ARIMA, FNN, FC-LSTM



<br>

Traffic 11 : PeMSD7을 둘로 변형

- 소개 : randomly select a medium and a large scale among the District 7 of California,
   containing 228 and 1,026 stations

- 둘로 나눔

  - PeMSD7(M) : 228 stations

  - PeMSD7(L) : 1026 stations

- time range of PeMSD7 dataset is in the weekdays of May and June of 2012

- - 주요 모델 : GCGRU, 제안1, 제안2
    - 제안 1: STGCN (Cheb)
    - 제안 2: STGCN (1st)
  - 기본 모델 : HA, LSVR, ARIMA, FNN, FC-LSTM

<br>

**Data Preprocessing**

- adjacency matrix 생성 방법(공식) 적혀있음

<br>

## (2) 메트릭

- MAE
- MAPE
- RMSE

<br>

## (3) 실험 세팅

[1] train-val-test :

- first month of historical speed records as training set
- and the rest serves as validation and test set respectively. ( 뭔소리야 ㅋㅋㅋ )

[2] 선택된 구조

- ST-Conv block 내에, 3개의 layer ( 각 channel은 64-16-64 )
- GCN kernel size=3
- 제안 1) STGCN(Cheb)
- 제안 2) STGCN(1st) : Cheb의 K=1인 경우

[3] RMSprop + 50 epoch + 50batchsize

[4] lr $10^{-3}$ ( with weight decay of 0.7 after every 5 epochs )

<br>

## (4) Test

train-val-test :

- first month of historical speed records as training set
- and the rest serves as validation and test set respectively. ( 뭔소리야 ㅋㅋㅋ )

