---
title: \[CS 기초\] Home Server
categories: [CS]
tags: [CS]
excerpt: IP address, Router, NAT, Port Forwarding, Dynamic/Static IP, DHCP
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Home Server

Contents

1. IP address
2. Router
3. Network Address Translation (NAT)
4. IP address 확인하는 방법 
5. Port Forwarding
6. Dynamic vs Static IP Address
7. DHCP (Dynamic Host Configuration Protocol)

<br>

# 1. IP address

IP Address

- 컴퓨터 고유의 ID ( 주민등록번호 )



IP 주소 규칙(형식)

- IPv4  (xxx.xxx.xxx.xxx)
- IPv6  (xxxx.xxxx.xxxx.xxxx.xxxx.xxxx.xxxx.xxxx)

<br>

# 2. Router

- 컴퓨터는 **IP 주소를 가지고 있어야** 서로 통신을 할 수 있다.

- 통신사랑 계약을 해서 회선을 연결하게 되면, **IP를 부여받게 된다.**

  만약, 집에 **여러 대의 기기**가 있으면?

  ex) 스마트폰, 태블릿, 노트북...

- **공유기(Router)**를 구매할 것! ( =교환원 )

  - WAN (Wide Area Network) : 통신사랑 계약해서 받게되어서 연결하는 케이블

    ( 외부 인터넷에 연결해주는 역할 )

    - **public IP address : 59.6.66.238**

  - LAN (Local Area Network) : 다양한 기기에 연결

    - **private IP address :**
      - **공유기의 IP : 192.168.0.1**
      - ex) 스마트폰 (무선)
        - **ex) 192.168.0.2**
      - ex) 노트북
        - **ex) 192.168.0.3**
      - ex) 태블릿 ...
        - **ex) 192.168.0.4**

<br>

![figure2](/assets/img/cs/img19.png)

<br>

# 3. Network Address Translation (NAT)

NAT 기술 덕분에, private IP address를 쓰는 기기들도 외부 인터넷에 연결 가능해진다!

if, 노트북(**192.168.0.3**)가 wikipedia를 검색하면..

- 1) 공유기(**192.168.0.1**) 가 이 요청을 받고,

  - 내부 network 안에 있는 요청인지 확인 ( ex. 스마투폰 / 노트북 )

  - 아닐 경우, 외부로 요청 (via WAN)

  - 보내기 전에...

    - 1-1) 누가 요청을 보냈는지를 기록해둔다

      ( "노트북(**192.168.0.3**)가 요청했다"는 사실 )

    - 1-2)  NAT기술을 사용하여, (private IP인)**192.168.0.3**가 요청했다는 사실을 (public IP) **59.6.66.238**가 요청했다고 변경

    - 1-3) 그런 뒤, 이 요청을 WAN을 통해 wikipedia로 보낸다

  - 받는 과정

    - 2-1) wikipedia는**59.6.66.238**로 회신(응답)을 할 것
    - 2-2) 이 정보를 받은 공유기는, **192.168.0.3**으로 해당 정보를 전달해줌

<br>

# 4. IP address 확인하는 방법 

확인하고자 하는 IP

- 1) 세 대의 기기 (**192.168.0.2 ~ 192.168.0.4**)
- 2) 공유기의 private IP ( **192.168.0.1** )
- 3) 공유기의 public IP ( **59.6.66.238** )

<br>

제어판 > Network and Internet  > Network and Sharing Center > Connections

- `Ethernet` 혹은 `Wifi`

Details 

- IPv4 Address : 컴퓨터의 IP (**192.168.0.3**)
- IPv4 Default Gateway : 공유기의 private IP ( **192.168.0.1** )

<br>

### Console에서 확인하기

`ipconfig`

- `etho0` > `inet addr` : 컴퓨터의 IP (**192.168.0.3**)

`route`

- destination이 "default"인 부분의 Gateway : 공유기의 private IP ( **192.168.0.1** )

<br>

웹브라우저에 "공유기의 private IP ( **192.168.0.1** )" 를 입력하면, 공유기로 접속 가능

- 로그인해서 관리자 화면 들어가기 ( = Router의 환경설정 )

- WAN 상에서의 IP는 어떻게 되는가?

  ( = 공유기의 public IP는 어떻게 되는가? )

  - 외부 IP주소 ( **59.6.66.238** )

- ( 혹은, 구글에 my IP검색해도 됨 :  ( **59.6.66.238** ) )

<br>

# 5. Port Forwarding

Client 대신 Server로써 사용하기 위해서는?

- 현재로써는 hard!

- 외부에서 **59.6.66.238**으로 접속할 경우....

  세 대의 기기 중 어디로 연결을 해줘야 할 지 모르기 때문! 식별 불가!

- 따라서, 그 셋 중 누구와 상호작용하고 싶은지를 알려줘야 하고,

  그래서 이를 식별하기 위한 **포트 번호**를 알아야 한다!

<br>

### 포트 번호

- 0~65535까지의 포트 번호

- 0~1023 : Well-known port

  - ex) 22-SSH

  - ex) 80-http ( 웹 페이지를 주고 받을 때 )

    - web server는 기본적으로 80번 포트에 "리스닝"하고 있다

      ( 늘 응답받을 준비 중! )

    - 한대를 추가하고자 한다면, 다른 포트 번호를 사용해야!

      ( ex. 8080 )

- 그럼, 80을 통해? or 8080을 통해...? URL을 통해 알 수 있다!
  - 80번 : http://opentutorials.org:80 or http://opentutorials.org
  - 8080번 : http://opentutorials.org:8080

<br>

### Port Forwarding

Setting

- 공유기의 public IP address : 59.6.66.238

- web server의 private IP address : 192.168.0.4

<br>
Goal

- 외부의 사용자가 59.6.66.238로 입력을 하면,

  web server인 192.168.0.4로 연결시켜 주기!

- ex)

  - from : 59.6.66.238:**8080**
  - to : 192.168.0.3:**80**

- ex2)

  - from : 59.6.66.238:**8081**
  - to : 192.168.0.4:**80**

<br>

Port Forwarding하는 방법?

- 위처럼 Router의 환경 설정 들어가기
- "포트 포워드 설정"
  - 규칙 이름 : web server 1
  - 내부 IP주소 : 192-168-0-4
  - 외부 포트 : 8081
  - 내부 포트 : 80
- 해석 : 외부에서 **8081**로 들어오면, **80번 포트**에 연결된 **내부 IP 192.168.0.4**에 연결해준다!

- 웹브라우저에 :  59.6.66.238:**8081**로 접속을 할 경우..
  - **내부 IP 192.168.0.4**에 연결이 된 것을 확인할 수 있다!

<br>

# 6. Dynamic vs Static IP Address

부족한 IP 주소 문제를 해결하기 위해!

<br>

ISP : Internet Service Provider (통신사)와 계약!

- 케이블을 꽂으면, IP address를 부여받게 된다 ( ex. 59.6.66.238 )

<br>

하지만...모든 집에 이러한 IP를 부여하게 되면 부족해진다!

- **안쓰면 거둬가고, 쓰면 다시 부여하고!** 

- 따라서 **IP가 계속 동적(dynamic)하게 바뀌게 된다.**

  ( 고정 IP를 쓰고 싶으면, 추가적인 요금을 내야! )

<br>

# 7. DHCP (Dynamic Host Configuration Protocol)

192.168.0.2~192.168.0.4를 어떻게 부여받게 될까?

수동으로..? NO! 자동으로 설정해주는 **"DHCP"**

( 케이블 꽂으면, Wifi 무선 연결 시, "동적으로 IP address"가 자동으로 setting되도록! )

<br>

DHCP

- DHCP server : 공유기

- DHCP client : 컴퓨터/노트북/태블릿/핸드폰...

  ( 공장에서 기록된 고유의 식별자 : mac address / physical address )

<br>

DHCP client : *DHCP서버님! 저는mac address ~~인 기기입니다.* *저에게 IP 주소를 부여해주세요!*

DHCP server : *부여 완료!* (192.168.0.4 를 부여해줌 )

DHCP client : *네! 사용하겠습니다!*

DHCP server : *기록 완료!*

<br>



