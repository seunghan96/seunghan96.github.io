# ML Flow 5. Experiments & Runs

1. Experiments? Runs?

- Experiments (실험)
- Runs (실행)

실험은 일종의 "프로젝트"라고 볼 수 있고, 하나의 실험은 여러 개의 실행을 가질 수 있다.

<br>

1. 경로 이동

```
$ cd mlflow/examples
```

이번에 실습할 예제는 `sklearn_autolog`이다.

<br>

2. MLproject 실행

```
$ mlflow run sklearn_elasticnet_wine --no-conda
```

```
/Users/seunghan96/opt/anaconda3/lib/python3.9/site-packages/click/core.py:2309: FutureWarning: `--no-conda` is deprecated and will be removed in a future MLflow release. Use `--env-manager=local` instead.
  value = self.callback(ctx, self, value)
2022/05/06 21:40:30 INFO mlflow.projects.utils: === Created directory /var/folders/ln/bxrzt06d0r3fbxsdkgxb_dc80000gn/T/tmp4dukax80 for downloading remote URIs passed to arguments of type 'path' ===
2022/05/06 21:40:30 INFO mlflow.projects.backend.local: === Running command 'python train.py 0.5 0.1' in run with ID 'c4c898f021ac40f8ab53c96a43d43313' === 
Elasticnet model (alpha=0.500000, l1_ratio=0.100000):
  RMSE: 0.7460550348172179
  MAE: 0.576381895873763
  R2: 0.21136606570632266
2022/05/06 21:40:37 INFO mlflow.projects: === Run (ID 'c4c898f021ac40f8ab53c96a43d43313') succeeded ===
```

<br>

3. 구조 확인하기

위의 `mlflow run` 의 결과로, 아래와 같은 경로/파일들이 생성된 것을 알 수 있다.

![figure2](/assets/img/mlops/img154.png)

해석 :

- 실험 (Experiment) ID : `0`
- 실행 (RUN) ID : `c4c898f021ac40f8ab53c96a43d43313`

<br>

위의 실행을, (이번엔 파라미터 값을 지정해서) 한번 더 실행해보자.

( 여러 다른 파라미터값을 지정하여 실행하고 싶을때, 이와 같이 여러 실행을 돌리면 된다 )

```
$ mlflow run sklearn_elasticnet_wine -P alpha=0.5 --no-conda
```

<br>

구조를 확인해보면, **동일한 실험 하에** 하나의 RUN이 추가된 것을 알 수 있다.

![figure2](/assets/img/mlops/img155.png)

<br>

4. 실험 (Experiment) 생성 & 조회

실험 생성

- experiment name : `my_exp`
- experiment ID : `1`

```
$ mlflow experiments create -n "my_exp"
```

```
Created experiment 'my_exp' with id 1
```



실험 조회

```
$ mlflow experiments list
```

```
  Experiment Id  Name     Artifact Location
---------------  -------  ---------------------------------------------------------
              0  Default  file:///Users/seunghan96/Desktop/mlflow/examples/mlruns/0
              1  my_exp   file:///Users/seunghan96/Desktop/mlflow/examples/mlruns/1
```

<br>

5. Experiment 지정 후 실행

두  가지 방식으로 지정해 줄 수 있다.

- (1) experiment ID 사용 : `--experiment-id 1`
- (2) experiment name 사용 : `--experiment-name "my_exp"`

```
$ mlflow run sklearn_elasticnet_wine -P alpha=0.5 --no-conda --experiment-id 2
```

```
$ mlflow run sklearn_elasticnet_wine -P alpha=0.25 --no-conda --experiment-name "my_exp"
```