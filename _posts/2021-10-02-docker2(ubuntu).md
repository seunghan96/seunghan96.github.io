---
title: \[따배도\] 2-1.Docker Container 설치하기 (Ubuntu)
categories: [DOCKER]
tags: [docker]
excerpt: 따라하며 배우는 도커
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[따배도] 2-1.Docker Container 설치하기 (Ubuntu)

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

## [Step 2-1]

VM에 **Ubuntu 20.04** 설치 & 기본환경 구성

순서 : 

- 1) Ubuntu 20.04 다운로드 & 설치
- 2) 기본 구성
- 3) 원격 로그인 가능하도록 구성

<br>

### 2-1-1) Ubuntu 20.04 다운로드 & 설치

`ubuntu-20.04.1-desktop-amd64.iso`파일 다운받기

- https://ubuntu.com/#download

![figure2](/assets/img/docker/img9.png)



설치 시작

![figure2](/assets/img/docker/img10.png)

![figure2](/assets/img/docker/img0.png)

![figure2](/assets/img/docker/img11.png)

- iso파일넣어주는거 스샷



**시작** 버튼 눌러서 부팅하기!



설치 진행 과정 : 

- 언어 : ( 영어 or 한국어 ) 선택
- 키보드 레이아웃 : 기본 설정 대로
- 업데이트 및 기타 소프트웨어 : 계속
- 설치 형식 : 계속
- 파티션 : 계속
  - 자동으로 파티션이 만들어진다
- TimeZone : 서울
- 계정 정보 입력 
  - 이름 : `guru`
  - 비번 : `****`
- 설치 완료 후 Rebooting & 로그인하기

<br>

### 2-1-2) 기본 구성 변경해주기

우측 상단에 **설정** 클릭

[ 디스플레이 ]

- `디스플레이` - `해상도` - **1280x960**

  ( 업데이트 skip )

[ 네트워크 ]

- `네트워크` - 유선 : ''연결됨'' 오른쪽에 있는 **설정** 클릭
  - IPv4를 **'자동'에서 '수동'으로 변경**하기
  - 주소 : `10.100.0.105`
  - 네트마스크 : `24`
  - 게이트웨이 : `10.100.0.1`
  - 네임서버 DNS : `10.100.0.1`



터미널 시행하기

- Ctrl + Shift + +/-로 글자 크기 변경가능

- `ip addr` : ip address 확인

  - 10.100.0.105 확인

- `hostname`

  - docker-ubuntu 확인

- `sudo vi /etc/hostname` 파일 열어서 수정 가능

  - (구) docker-ubuntu
  - (신) docker-ubuntu.example.com
    - esc & :wq로 저장

  ![figure2](/assets/img/docker/img12.png)

- `sudo vi /etc/hosts/`

  - docker-ubuntu / docker-centos의 **ip address와 hostname을 등록**
    - esc & :wq로 저장

  ![figure2](/assets/img/docker/img13.png)

- `ping -c 3 8.8.8.8`

  - 구글에 잘 접속되는지 확인!

- `sudo passwd root`

  - root 패스워드 설정하기

- `sudo passwd guru`

  - 현재 이용자(guru)의 패스워드 변경하기

- `su - root`

  - (guru에서) root로 계정 전환하기

  ![figure2](/assets/img/docker/img27.png)

  

- `systemctl set-default multi-user.target`

  - 하드웨어 리소스를 적게 사용하기 위해 gui에서 text mode로 변경

- `apt-get update`

  - 최신 repository 업데이트

- `apt-get install -y openssh-server curl vim tree`

  - openssh-server, curl, vim, tree 설치하기
    - ssh : 원격 접속 가능케하기 위해!
  - 설치과정에서 모두 yes 누르도록

  ![figure2](/assets/img/docker/img28.png)

  

- `systemctl status sshd`

  - 현재 잘 실행됨을 확인할 수 있다 ( active (running) )

- `ssh guru@localhost`

  - guru 사용자로 접속됨을 확인 가능
  - `exit` 으로 나오기

  ![figure2](/assets/img/docker/img29.png)

- `reboot`

  - 시스템 리부팅하기

<br>

### 2-1-3) Xshell

Xshell을 사용하여 가상머신(Ubuntu)에 연결한다!

[ 새로만들기 ]

[ 연결 ]

- 이름 : `docker-ubuntu`

- 호스트 : `127.0.0.1`
- 포트 번호 : `105`

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

![figure2](/assets/img/docker/img32.png)

<br>

`free -h`

- 현재 시스템의 메모리 사용 정보를 인간(human) readable하게

`uname -r`

- 설치되어있는 kernel 버전 확인하기

`sudo systemctl isolate graphical.target`

- gui로 모드로 변경하고 싶으면!

![figure2](/assets/img/docker/img33.png)

<br>

### 2-1-4) 스냅샷

- ( memory 사이즈를 4096에서 2048로 다시 줄이고 )
- 우클릭 - `스냅샷`
  - 현재 시점을 다시 roll back할 수 있음
- `찍기`
  - 스냅샷 이름 : `os-install`
  - 스냅샷 설명 :
    - guru : 비밀~
    - root : 비밀~
    - ipaddr : 10.100.0.105
    - sshd
    - text login

<br>