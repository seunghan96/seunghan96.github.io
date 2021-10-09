---
title: \[따배도\] 2-3.Docker Container 설치하기
categories: [DOCKER]
tags: [docker]
excerpt: 따라하며 배우는 도커
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[따배도] 2-3.Docker Container 설치하기

( 참고 : 따배도 https://www.youtube.com/watch?v=NLUugLQ8unM&list=PLApuRlvrZKogb78kKq1wRvrjg1VMwYrvi )



### Contents

1. Docker Container 설치를 위해 필요한 것
2. 실습으로 따라하기
   1. [ Step 1 ] **Virtual Box** 설치 - 네트워크 구성 - **VM** 생성
   2. [ Step 2-1 ] VM에 **Ubuntu 20.04** 설치 & 기본환경 구성
   3. [ Step 2-2 ] VM에 **CentOS 20.04** 설치 & 기본환경 구성
   4. [ Step 3 ] Ubuntu / CentOS 서버에 **Docker 설치하기**



## [ Step 3 ] Ubuntu / CentOS 서버에 **Docker 설치하기**

이번 단계에서는 앞서 설치한 Ubuntu와 CentOS에 Docker를 설치할 것이다.

그러기 위해 우선 XShell을 사용하여 각각에 접속할 것이다.

![figure2](/assets/img/docker/img37.png)

<br>

도커를 설치하는 방법에는 크게 3가지가 있다.

1. **Repository를 이용한 설치**
2. (원격 접속 불가한 경우) Download 후 직접 설치
3. Script를 이용한 설치

이 중, 우리는 1번 (Repository를 이용한 설치) 방법으로 설치할 것이다.

<br>

아래의 사이트에서 Docker를 다운받을 수 있다.

- https://docs.docker.com/get-docker/

![figure2](/assets/img/docker/img38.png)

<br>

### < Ubuntu >

- https://docs.docker.com/engine/install/ubuntu/

- Ubuntu상에서 Docker를 설치하는 방법/필요 조건들이 명시되어있다

- 순서

  - step 1) 요구하는 필수 프로그램들 설치

  ```bash
  sudo apt-get update
  sudo apt-get install \
      apt-transport-https \
      ca-certificates \
      curl \
      gnupg \
      lsb-release
  ```

  ![figure2](/assets/img/docker/img39.png)

  - step 2) 인증서 저장

  ```bash
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  ```

  

  - step 3) 도커 repository url을 Ubuntu 상에 등록하기

  ```shell
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  ```

  

  - step 4) docker engine 설치하기 ( apt-get )

  ```shell
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io
  ```

  ​		- `docker-ce` : docker 데몬

  ​		- `docker-ce-cli` : docker command line

  ​	

- Ubuntu 상에서 따로 서비스 데몬을 구동시키지 않아도, 설치만 하고나면 바로 사용가능하다!

- 도커가 잘 작동되는지 test해보자

  - client version
  - server version

  두 가지가 나옴을 알 수 있다.

  ```bash
  sudo docker version
  ```

  ![figure2](/assets/img/docker/img40.png)

<br>

### < CentOS > 

- https://docs.docker.com/engine/install/centos/

- Centos상에서 Docker를 설치하는 방법/필요 조건들이 명시되어있다

- Ubuntu와 CentOS의 Docker 설치 방법은, 명령어 외에는 전부 동일하다.

- 순서

  - step 1) 요구하는 필수 프로그램들 설치

    ( `su -`로 관리자로 변경한 뒤...)

  ```bash
  yum install -y yum-utils
  ```

  - ~~step 2) 인증서 저장~~ ( CentOS에서는 불필요하다 )

  - step 2) 도커 repository url을 CentOS 상에 등록하기

  ```shell
  yum-config-manager \
      --add-repo \
      https://download.docker.com/linux/centos/docker-ce.repo
  ```

  

  - step 3) docker engine 설치하기 ( apt-get )

  ```shell
  yum install docker-ce docker-ce-cli containerd.io
  ```

  ​		- `docker-ce` : docker 데몬

  ​		- `docker-ce-cli` : docker command line

  

  - step 4) service 데몬 start

    ( Ubuntu와는 달리, 따로 서비스 데몬을 구동시켜야한다. )

  ```shell
  systemctl start docker
  systemctl enable docker
  ```

  ​		\- enable : 다음에 재부팅시에도 enable되어있도록!

<br>

- 도커가 잘 작동되는지 test해보자

  - client version
  - server version

  두 가지가 나옴을 알 수 있다.

  ```bash
  docker version
  ```

<br>

### 권한 확인

Ubuntu, CentOS 모두 guru 사용자가 permission denied된 것을 확인할 수 있다.

- Ubuntu

  ![figure2](/assets/img/docker/img41.png)

- CentOS

  ![figure2](/assets/img/docker/img42.png)

<br>

### 사용자에게 권한 할당

- root가 guru 사용자에게 docker 관리자 권한을 할당한다

<br>

[ Ubuntu & CentOS 동일 ]

```
su -
usermod -a -G docker guru
su - guru
```

```
docker ps
```

- 권한이 할당됨을 알 수 있다
