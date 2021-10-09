---
title: \[따배도\] 2-2.Docker Container 설치하기 (CentOS)
categories: [DOCKER]
tags: [docker]
excerpt: 따라하며 배우는 도커
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[따배도] 2-2.Docker Container 설치하기 (CentOS)

( 참고 : 따배도 https://www.youtube.com/watch?v=NLUugLQ8unM&list=PLApuRlvrZKogb78kKq1wRvrjg1VMwYrvi )



### Contents

1. Docker Container 설치를 위해 필요한 것
2. 실습으로 따라하기
   1. [ Step 1 ] **Virtual Box** 설치 - 네트워크 구성 - **VM** 생성
   2. [ Step 2-1 ] VM에 **Ubuntu 20.04** 설치 & 기본환경 구성
   3. [ Step 2-2 ] VM에 **CentOS 20.04** 설치 & 기본환경 구성
   4. [ Step 3 ] Ubuntu / CentOS 서버에 **Docker 설치하기**



## Q1. Docker Container 설치를 위해 필요한 것

도커도 일종의 application이다. **컴퓨터 & 도커 프로그램**만 있으면 사용할 수 있다.

\* ***세부 조건***

- CPU : 2 core 이상
- Memory : 2GB 이상

- 운영체제 : 리눅스 ( Windows / Mac OS 도 가능하긴 함 )

<br>

## Q2. 실습으로 따라하기

[ Step 1 ] **Virtual Box** 설치 - 네트워크 구성 - **VM** 생성

- Hypervisor인 Virtual Box를 설치한다
- VM들 간의, VM과 외부와의 network를 구성한다

[ Step 2-1 ] VM에 **Ubuntu 20.04** 설치 & 기본환경 구성

[ Step 2-2 ] VM에 **CentOS 20.04** 설치 & 기본환경 구성

[ Step 3 ] Ubuntu / CentOS 서버에 **Docker 설치하기**

<br>

***실습 과정은 아래의 강좌를 참고***

- https://www.youtube.com/watch?v=PqgWp7rbqws&list=PLApuRlvrZKogb78kKq1wRvrjg1VMwYrvi&index=4



## [ Step 1 ]

**Virtual Box** 설치 - 네트워크 구성 - **VM** 생성

<br>

### 1-1) Virtual Box 설치

대표적인 HyperVisor인 **Virtual Box**를 아래의 링크를 통해 설치한다.

- https://www.virtualbox.org/



### 1-2) Virtual Box 내의 Network 구성

**NAT 네트워크** 추가하기

( for 외부와의 통신 & VM 간의 통신 )

- network 이름 : `localNetwork`
- network CIDR : `10.100.0.0/24`
- DHCP 지원
- 포트포워딩
  - docker 1 ( Ubuntu )
    - 호스트 IP : `127.0.0.1`
    - 호스트 포트 : `105`
    - 게스트 IP : `10.100.0.105`
    - 게스트 포트 : `22`
  - docker 2 ( CentOS )
    - 호스트 IP : `127.0.0.1`
    - 호스트 포트 : `106`
    - 게스트 IP : `10.100.0.106`
    - 게스트 포트 : `22`

![figure2](/assets/img/docker/img2.png)

![figure2](/assets/img/docker/img3.png)

<br>

### 1-3) VM (Virtual Machine) 생성

2개의 VM을 생성할 것

- 1) Ubuntu
  - 이름 : `docker-ubuntu`
  - CPU(2core), Memory(2GB), network(localNetwork), disk(20GB)
- 2) CentOS
  - 이름 : `docker-centos`
  - CPU(2core), Memory(2GB), network(localNetwork), disk(20GB)

<br>

1) Ubuntu

![figure2](/assets/img/docker/img4.png)

![figure2](/assets/img/docker/img5.png)

![figure2](/assets/img/docker/img6.png)

![figure2](/assets/img/docker/img7.png)



2) CentOS

![figure2](/assets/img/docker/img8.png)

![figure2](/assets/img/docker/img5.png)

![figure2](/assets/img/docker/img6.png)

![figure2](/assets/img/docker/img7.png)

<br>

## [Step 2-2]

### 2-2-1) CentOS7 다운로드 & 설치

VM에 **CentOS 20.04** 설치 & 기본환경 구성

![figure2](/assets/img/docker/img16.png)

![figure2](/assets/img/docker/img17.png)

- http://mirror.navercorp.com/centos/7.9.2009/isos/x86_64/

<br>

![figure2](/assets/img/docker/img18.png)

![figure2](/assets/img/docker/img19.png)

<br>

[ Time Zone ]

- Seoul로

![figure2](/assets/img/docker/img20.png)

<br>

[ Software Selection ]

- gnome desktop으로

![figure2](/assets/img/docker/img21.png)

<br>

[ Installation Destination ]

- 그냥 done누르기 ( 자동으로 partitioning )

![figure2](/assets/img/docker/img22.png)

<br>

[ Network & Hostname ]

![figure2](/assets/img/docker/img23.png)

- hostname을 `docker-centos.example.com`으로 바꿔주기

![figure2](/assets/img/docker/img24.png)

- static ip로 바꿔주기

![figure2](/assets/img/docker/img25.png)

- Ethernet 켜고 Done눌러서 저장하기

<br>

위의 세 가지 변경사항을 적용한 뒤, **Start Installation**

<br>

[ 설치 시작 ]

![figure2](/assets/img/docker/img26.png)

- 설치되는 과정에서,
  - 1) Root의 password를 지정할 수 있고
  - 2) User를 새롭게 생성할 수 있다

설치 완료!

<br>

### 2-2-2) 기본 구성 변경해주기

우측 상단의 환경설정 들어가기

![figure2](/assets/img/docker/img34.png)

- `Devices` > `Displays` > `Resolution` : 1280x960

- `Region & Languages` : +키 누르고, more에서 Korean 추가하기
- `Privacy` : Auto Screen Lock OFF로 끄기
- `Power` : Power Saving > Blank screen > NEVER로
- `Network` : 톱니바퀴 > 잘 구성된 것을 확인할 수 있음

<br>

Terminal 접속

( Ctrl + Shift + +/-로 확대/축소 가능)



`ip addr`

- ip address 확인
  - 확인해보면, virbr0라는 가상머신 안에 있는 hypervisor가 기본으로 켜져있음을 알 수 있다. 이를 꺼준다
  - `systemctl stop libvirtd`
    - 지금 중단
  - `systemctl disable libvirtd`
    - 앞으로도 중단

`vi /etc/hostname`

- "docker-centos.example.com"으로 잘 등록되어있음을 알 수 있다

`vi /etc/hosts`

- Ubuntu와 마찬가지로 추가해준다.
  - 10.100.0.105 docker-ubuntu.example.com docker-ubuntu
  - 10.100.0.106 docker-centos.example.com docker-centos

`ping -c 3 8.8.8.8`

- 구글에 잘 접속됨을 확인한다

`systemctl set-default multi-user.target`

- text mode를 default로

`yum install -y tree`

- tree 설치하기

![figure2](/assets/img/docker/img35.png)



Restart하기

![figure2](/assets/img/docker/img36.png)



이제 XShell을 사용하여 원격 login을 할 것이다.

<br>

### 2-2-3) Xshell

Xshell을 사용하여 가상머신(Ubuntu)에 연결한다!

[ 새로만들기 ]

[ 연결 ]

- 이름 : `docker-centos`

- 호스트 : `127.0.0.1`
- 포트 번호 : `106`

![figure2](/assets/img/docker/img14.png)

[ 사용자 인증  ]

![figure2](/assets/img/docker/img15.png)

![figure2](/assets/img/docker/img30.png)

<br>

잘 접속된 것을 확인할 수 있다

![figure2](/assets/img/docker/img31.png)

<br>

`su - `

- 루트 사용자로

- `exit` 하면 다시 guru사용자로

`ip addr`

- ip address 정보 확인하기

`cat /etc/os-release`

- 설치되어있는 OS 정보 확인하기

`free -h`

- 현재 시스템의 메모리 사용 정보를 인간(human) readable하게

`uname -r`

- 설치되어있는 kernel 버전 확인하기

`sudo systemctl isolate graphical.target`

- gui로 모드로 변경하고 싶으면!

<br>

### 2-2-4) 스냅샷

- 우클릭 - `스냅샷`
  - 현재 시점을 다시 roll back할 수 있음
- `찍기`
  - 스냅샷 이름 : `os-install`
  - 스냅샷 설명 :
    - guru : 비밀~
    - root : 비밀~
    - ipaddr : 10.100.0.106
    - Centos 7 