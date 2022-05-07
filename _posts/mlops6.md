# ML Flow 6. Tracking Server

1. Tracking이란?

- Tracking ( 트래킹 ) = ***"기록하기"***
- 즉, ML의 과정 & 결과를 기록한다!
- Tracking은 실험의 각 "실행"에서 발생한다.



2. Tracking에서 기록하는 것들

- (1) 코드 버전
- (2) 시작 & 종료 시간
- (3) 소스
- (4) 매개 변수
- (5) 메트릭

- (6) 아티팩트



**그렇다면, 위 과정 & 결과물들은 "어디에 기록"되는가?**

$\rightarrow$ ( default ) `./mlruns` 경로에 저장된다

<br>

3. 기록물들의 분류

앞서, 여러 가지 의 정보를 기록한다고 했다.

이것들은, 크게 아래와 같이 2가지로 구분할 수 있다.

- (1) Artifacts
  - 파일, 모델, 이미지 등
  - ex) `artifacts`
- (2) MLflow Entity
  - 실행, 매개 변수, 메트릭, 태그, 메모, 메타 데이터 등
  - Ex) `meta.yaml`, `metrics`, `params`, `tags`

<br>

앞선 예제들에서는, 이 두 분류 모두 다 하나의 경로 ( `./mlruns` )에 저장되었었다.

하지만, 필요에 따라 이 둘을 구분하여 다른 경로에 저장할 수 있다. 그러기 위해 필요한 것이 바로 ***Tracking Server (트래킹 서버)*** 인 것이다.



4. Tracking Server

앞서 말했듯, ML의 과정 및 결과를 기록하기 위한 별도인 서버가 존재할 수 있고, 이를 **Tracking Server**라 한다.

앞선 예제들에서는, `mlflow.log_params`, `mlflow.log_metrics` 등을 통해 `./mlruns`에 기록을 했었다면, 이번에는 **백엔드 서버**를 통해 저장할 것이다.

<br>

5. Tracking Server 띄우기

우선, 실습을 진행할 경로로 이동 부터 하자.

```
$ cd mlflow/examples
```

<br>

이 안에, 트래킹 서버를 위한 경로를 생성 & 이동하자

```
$ mkdir tracking-server
$ cd tracking-server 
```

<br>

Tracking Server를 띄우는 명령은 아래와 같다.

```
$ mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root $(pwd)/artifacts
```

![figure2](/assets/img/mlops/img156.png)

- 5000번 포트에 리스닝하고 있는 것을 알 수 있다.

  ( `http://127.0.0.1:5000`)

- 2개의 저장소

  - (1) `--backend-store-uri` : ML Entity를 저장하는 저장소
    - 파일저장소/DB 사용 가능
  - (2) `--default-artifact-root` : Artifacts를 저장하는 저장소
    - 파일 시스템/스토리지 사용 가능 
    - 외부 저장소 ex) AWS S3, GCS 등

<br>

6. Project & Tracking Server 통신 설정

우리의 ML 프로젝트가, 방금 띄운 백엔드 Tracking Server와 통신할 수 있도록 설정해준다.

```
$ export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

<br>

7. 프로젝트 실행

이번에 실행할 프로젝트는, 앞서 실행했던 프로젝트와 동일한 `sklearn_logistic_regression` 이다. 마찬가지로, 콘다환경을 사용하지 않을 것이다.

```
$ mlflow run sklearn_logistic_regression --no-conda
```

![figure2](/assets/img/mlops/img157.png)

<br>

8. 결과 확인하기

이번엔, `mlruns` 폴더가 생성되지 않았다. 이는, 우리가 별도의 트래킹 서버를 띄우고, 이를 우리의 저장 경로로 지정했기 때문이다

( `export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"` )

<br>

예상했겠지만, 우리가 원하는 결과물들은, 앞서 생성한 `tracking-server` 에 저장되어 있을 것이다. 한번 이동해서, 확인해보자.

```
$ tree tracking-server
```

![figure2](/assets/img/mlops/img158.png)

<br>

확인을 해보면, 두 개의 파일이 생성된 것을 알 수 있다.

- (1) `artifacts`
- (2) `mlflow.db`

<br>

이 결과들도, 마찬가지로 웹 UI를 통해서 확인할 수 있다.

우리의 트래킹 서버 경로인 `http://127.0.0.1:5000` 에 들어가서 확인해보자.

![figure2](/assets/img/mlops/img159.png)

<br>

우리가 Tracking한 내용들을 보다 자세히 확인하기 위해서, `models` 밑의 `sklearn`을 클릭해보자.

![figure2](/assets/img/mlops/img160.png)

<br>

간단 요약

- Tracking Server는 ML 프로젝트의 과정 & 결과를 기록해두는 서버
- Tracking Server의 웹 대시보드를 사용하여 쉽게 관리 가능