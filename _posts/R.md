---
title: All about R
categories: [R]
tags: [R]
excerpt: Must Learning with R (위키독스) 정리
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

참고 : https://wikidocs.net/book/4315

<br>

# 1. R 기본문법 1단계

```R
getwd() # 현재 경로
setwd('바꿀경로') # 경로 변경
```

<br>

## 1) 벡터

`c()` : 벡터

`rep()` : repeat ( 반복된 데이터 )

`seq()` : sequence ( 순차적 데이터 )

<br>

```R
x1 = c(1:10) 
# 1,2,...,10

x2 = seq(1,10,2)
# 1,3,5,7,9

x3 = rep(1,10)
# 1,1,...,1
```

<br>

## 2) 데이터 프레임

```R
df = data.frame(
 X = c(1:10),
 y = rep(1,10)
)

head(df)
```

<br>

## 3) for loop

```R
A=c(1,2,3,4,5) 

for (i in A){
   print(i)
}
```

<br>

## 4) indexing

### 벡터

```R
A[2] # 2번째 
A[1:2] # 1,2번째 
A[-1] # 1번째 제외
A[c(1,2,5)] #1,2,5번째 
```

<br>

### 데이터프레임

```R
df[1,] # 1행
df[,1] # 1열
df[c(1,2,3),-2] # 1,2,3 행 & 2열 제외
```

<br>

## 5) change datatype

데이터 형태 바꾸기

```R
x1 = as.integer(x)
x2 = as.numeric(x)
x3 = as.factor(x)
x4 = as.character(x)
```

<br>

데이터 형태 확인하기

```R
is.integer(x)
is.numeric(x)
is.factor(x)
is.character(x)
```

<br>

데이터 형태/요약 확인하기

```R
str(x1)
summary(x1)
```

<br>

## 6) sample

복원추출/비복원추출

```R
sample(1:45, 6)  # 비복원추출
sample(1:45, 6, replace = FALSE)  # 비복원추출

sample(1:45, 6, replace = TRUE)  # 복원추출
```

<br>

난수 고정

```R
set.seed(1234)
```

<br>

## 7) if 문

```R
A = c(1,2,3,4,5)

if( 7 %in% A){
 print("TRUE")
} else{ 
   print("FALSE")
}
```

<br>

## 8) 함수

```R
my function = function(x){
    y = 2*x
    return(y)
}
```

<br>

## 9) 패키지

```R
install.packages("패키지명")
library(패키지명)
```

<br>

# 2. R 기본문법 2단계

## 1) 데이터 불러오기

- 데이터 불러오기

```R
data = read.csv('경로')
```

<br>

- 기본적인 EDA

```R
head(data)
str(data)
summary(data)
```

<br>

## 2) conditioning & subset

- conditioning

```R
data$new_column = ifelse(data$old > 0,'POS','NEG')
```

<br>

- subset

```R
df_partial = subset(df,old>0)
df_partial = subset(df,old>0 & height>175)
df_partial = subset(df,old>0 | height>175)
```

<br>

## 3) 집계된 데이터 ( plyr 패키지 )

````R
library(plyr)
````

```R
Summary_df = ddply(df, # (1)
                   
                   c("department","salary"), # (2)
                  summarise,
                  
                   M_SF=mean(satisfaction_level),  # (3)
                   COUNT=length(department),
                   M_WH=round(mean(average_montly_hours),2))
                  )
```

해석

- (1) dataframe
- (2) department & salary 칼럼 별로(groupby) summarize하기
- (3) 다음의 3가지 변수 생성
  - 1) satisfaction_level의 평균
  - 2) 직원 수
  - 3) average_montly_hours의 평균

<br>

결과 :

```R
Summary_df
----------------------------------------------------
    department salary      M_SF COUNT   M_WH
1   accounting   high 0.6140541    74 205.91
2   accounting    low 0.5741620   358 199.90
3   accounting medium 0.5836418   335 201.47
4           hr   high 0.6731111    45 209.07
5           hr    low 0.6086567   335 202.46
6           hr medium 0.5803064   359 193.86
........
```

<br>

# 3. ggplot2

## 1) 큰 틀

```R
ggplot(data,aes(x=var1,y=var2))
```

- aes ( aesthetic ) : 변수는 이 안에 있어야!

<br>

다양한 그래프들

- `geom_bar( )` : bar plot
- `geom_histogram( )`  : histogram
- `geom_boxplot( )` : box plot
- `geom_line( )` : line plot

<br>

다양한 옵션들

- `labs( )`  : legend label
- `ggtitle( )` : title
- `xlabs( )`, `ylabs( )`  : x & y label

<br>

## 2) barplot : `geom_bar`

**기본 그래프**

- x축 : salary
- y축 : 집계된 값 ( count )

```R
ggplot(HR,aes(x=salary)) + geom_bar()
```

<br>

**색상 꾸미기**

```R
# royal blue 색으로
ggplot(HR,aes(x=salary)) +  geom_bar(fill = 'royalblue') 

# var1에 따라 색 다르게
ggplot(HR,aes(x=salary)) +  geom_bar(aes(fill=var1)) 
```

<br>

**범례 / 축 이름**

```R
ggplot(HR,aes(x=salary)) +  
    geom_bar(aes(fill=left)) +
    labs(fill = "Divided by left") + 
    xlab("봉급") + ylab("") 
```

img1

<br>

## 3) histogram : `geom_histogram`

**기본 그래프**

```R
ggplot(HR,aes(x=satisfaction_level))+geom_histogram()
```

<br>

**histogram 꾸미기**

```R
ggplot(HR,aes(x=satisfaction_level)) +
geom_histogram(binwidth = 0.01,col='red',fill='royalblue') 
```

<br>

## 4) density plot : `geom_density`

**기본 그래프**

```R
ggplot(HR,aes(x=satisfaction_level))+geom_density()
```

<br>

**density plot 꾸미기**

```R
ggplot(HR,aes(x=satisfaction_level)) +
geom_density(col='red',fill='royalblue') 
```

<br>

## 5) box plot : `geom_boxplot`

**기본 그래프**

```R
ggplot(HR,aes(x=left,y=satisfaction_level)) + geom_boxplot()
```



**box plot 꾸미기**

```R
ggplot(HR,aes(x=left,y=satisfaction_level)) +
    geom_boxplot(aes(fill = salary),alpha = I(0.4),outlier.colour = 'red') +
    xlab("이직여부") + ylab("만족도") + ggtitle("Boxplot") +
    labs(fill = "임금 수준") 
```

img2

<br>

## 6) scatter plot : `geom_point`

**기본 그래프**

```R
ggplot(HR,aes(x=average_montly_hours,y=satisfaction_level))+
   geom_point()
```





**scatter plot 꾸미기**

```R
ggplot(HR,aes(x=average_montly_hours,y=satisfaction_level))+
   geom_point(aes(col = left)) + 
   labs(col = '이직 여부') + xlab("평균 근무시간") + ylab("만족도")
```

<br>

# 4. 통계값

**데이터 요악값 (summary)**

```R
summary(df$var1)
```

<br>

**quantile**

- ex) 10%, 30%, 60%, 90% 분위수

```R
quantile(df$var1,probs = c(0.1,0.3,0.6,0.9)) 
```

<br>

**colMeans, colSums**

```R
colMeans(df[1:5])
colSums(df[1:5])
```

<br>

# 5. 결측치 & 이상치

결측치 여부 확인

```R
# 특정 변수의 결측치 여부
is.na(df$var1)

# 특정 변수의 결측치 개수
sum(is.na(df$var1))

# 모든 변수의 결측치 개수
colSums(is.na(df))
```

<br>

결측치 제거

- 모든 변수 중, 결측치가 1개라도 있는 행은 모두 제거

```R
df2 = na.omit(df)
```

- 특정 변수에 결측치 있는 행 제거

```R
df3 = df[complete.cases(df$var1),]
```

<br>

결측치 대체

```R
df$var2=df$var1
df$var2[is.na(df$var2)]=999
```

<br>

결측치 생략 후 계산

```R
mean(df$var,na.rm = TRUE)  # return : NA
mean(df$var,na.rm = TRUE)  # return : NA 제외 후 평균
```

<br>

이상치 기준 선정 & 제거

```R
Q1 = quantile(df$var1,probs = c(0.25),na.rm = TRUE) 
Q3 = quantile(df$var1,probs = c(0.75),na.rm = TRUE)

LC = Q1 - 1.5 * (Q3 - Q1) 
UC = Q3 + 1.5 * (Q3 - Q1) 

df2 = subset(df, var1>LC & var2<UC)
```

<br>

# 6. 문자열 처리

- 문자열 대체 : `gsub()`
- 문자열 분리 : `strsplit()`
- 문자열 합치기 : `paste()`
- 문자열 추출 : `substr()`
- 텍스트마이닝 함수: `Corpus()` & `tm_map()` & `tdm()`

<br>

**문자열 추출**

- ex) 1~5번째 글자

  ```R
  substr(df$var1,1,5)
  ```

<br>

**문자열 붙이기**

- ex) "@" & "naver.com" 붙이기

  ```R
  paste(df$id,"@","naver.com") # default : 한 칸씩 띄어서
  ### seunghan96 @ naver.com
  ### seunghan97 @ naver.com
  ### ...
  
  paste(df$id,"@","naver.com", sep="")
  ### seunghan96@naver.com
  ### seunghan97@naver.com
  ### ...
  ```

<br>

**문자열 분리**

- 특정 문자를 기준으로, 분리하기

  ( 주의 : 무조건 character 형태의 변수여야! )

- ex) ","를 기준으로 분리

  ```R
  strsplit(df$charvar1, split=",")
  strsplit(as.character(df$var1), split=",")
  ```

<br>

**문자열 대체**

- ex) ","를 "!"로 대체하기

  ```R
  df$var2 = gsub(",","!",df$var1)
  ```

<br>

# 7. Text Mining

```R
library(tm)
```

<br>

Text Mining 프로세스

1. Corpus 생성

2. TDM 생성

3. 문자 처리

   (특수문자 제거, 조사 제거, 숫자 제거 등..)

4. 문자열 변수 생성

<br>

## step 1) corpus 생성

코퍼스 생성

```R
my_corpus = Corpus(VectorSource(df$var1)) 
```

<br>

전처리 과정

- 1) 공백 제거
- 2) 특수문자 제거
- 3) 숫자 제거
- 4) 소문자 변환
- 5) 불용어 제거

```R
my_corpus = tm_map(my_corpus,stripWhitespace)
my_corpus = tm_map(my_corpus,removePunctuation)
my_corpus = tm_map(my_corpus, removeNumbers)
my_corpus = tm_map(my_corpus, tolower) 
my_corpus = tm_map(my_corpus,removeWords,
                 c(stopwords("english"),"my","custom","words"))
```

<br>

## step 2) TDM 생성

```R
my_tdm = DocumentTermMatrix(my_corpus)
my_tdm_df = as.data.frame(as.matrix(my_tdm))
```

<br>

기존 데이터랑 합치기

```R
df_total = cbind(df,my_tdm_df)
```



## step 3) 문자열 데이터 시각화

- term document matrix 만들기

```R
my_tdm2 = TermDocumentMatrix(my_corpus)
my_tdm2_mat = as.matrix(my_tdm2) 
```

<br>

- 최빈 등장 순으로 sorting & df로 만들기

```R
v = sort(rowSums(my_tdm2_mat),decreasing=TRUE) 
df_for_wc = data.frame(word = names(v),freq=v)  
```

<br>

- **시각화 1 : word cloud**
  - min.freq : 단어 최소 등장 횟수
  - max.words : 최대 단어 개수
  - random.order : 단어 위치 random 여부

```R
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

wordcloud(words = df_for_wc$word, freq = ddf_for_wc$freq,
          min.freq = 5,  max.words=200, random.order=FALSE,
          colors=brewer.pal(8, "Dark2"))
```

<br>

- **시각화 2 : 단어 빈도 그래프**

```R
top_N = 10

barplot(d[1:top_N,]$freq, las = 2, 
        names.arg = d[1:top_N,]$word,
        col ="lightblue", 
        main ="Most frequent words",
        ylab = "Word frequencies")
```

<br>

# 8. dplyr

```R
library(dplyr)
library(reshape)
```

<br>

apply 함수

```R
apply(df, 1, function) # row
apply(df, 2, function) # column

# A = B
apply(df[,1:2],1,mean) # [A]
rowMeans(df[,1:2]) # [B]
```

<br>

`%>%` ( pipeline )

```R
df[,1:2] %>%  
  rowMeans() %>%
  head()
```

```R
df2022 = df %>% filter(Year == "2022")
```

```R
# 샘플링
df_partial = df %>% sample_frac(size = 0.4, replace = FALSE)
df_partial = df %>% sample_n(size = 2, replace = FALSE)
```

```R
# index 통해 추출
df_partial = df %>% slice(1:10)
```

```R
# 특정 기준으로 추출
df_TOP = df %>% top_n(5,var1) 
```

```R
# 정렬하기 
df_sorted = df %>% arrange(var1)  # 오름차순
df_sorted = df %>% arrange(-var1)  # 내림차순
```

```R
# 특정 변수만 추출
df_selected = df %>% select(1:2)
df_selected = df %>% select(var1,var2)
```

```R
# 조건 따라 필터링 ( ex. 데이터 타입 )
df_factor = df %>% select_if(is.factor)
df_integer = df %>% select_if(is.integer)
```

<br>

데이터 집계

```R
df %>% 
   summarise(new_var1 = mean(var1),
             new_var2 = max(var2),
             new_var3 = length(var3))
```

```R
df2 = df %>%
     subset(var1 == 1) %>%
     group_by(var2) %>%
     dplyr::summarise(new_var1 = mean(var1),
                      new_var2 = max(var2),
                      new_var3 = length(var3))
```

<br>

변수 추가

```R
df3 = df2 %>% mutate(new_var4 = new_var1 / new_var2)
```

<br>

# 9. 중복 데이터 제거 & 정렬

## 1) 중복 데이터 제거

- 1차원

```R
unique_vector = unique(vector)
```

<br>

- 2차원

```R
# 2차원
## 1개의 변수라도 중복이면, 전부 제거
df2 = df[-which(duplicated(df)),]

## 특정 변수 1개가 중복이면, 제거
df2 = df[-which(duplicated(df$var1)),]

## 특정 변수 n개가 중복이면, 제거
df2 = df[!duplicated(df[,c(3,4,5)]),]
df2 = df[!duplicated(df[,c('var1','var2')]),]
```

<br>

## 2) 데이터 정렬

- 날짜 최신순으로 정렬!

```R
df$DATE = as.Date(df$DATE,"%Y-%m-%d")
df = df[order(df[,'DATE'],decreasing = TRUE), ]
```

<br>

# 10. Reshape

```R
library(reshape)
```



[형태1]

img3

[형태2]

img4

<br>

형태 1 -> 형태2 : `cast`

형태 2 -> 형태1 : `melt`

```R
cast_data = cast(df,OBS + NAME + ID + DATE ~ TEST)

melt_data = melt(cast_data,id=c("OBS","NAME","ID","DATE"))
melt_data = na.omit(MELT_DATA)
```

<br>

# 11. Merge

특정 key값을 기준으로 합치기

```R
merged_df = merge(df1,df2, by = "ID",all.x = TRUE)
merged_df = merge(df1,df2, by = "ID",all = TRUE)
merged_df = merge(df1,df2, by = "ID",all.y = TRUE)
merged_df = merge(df1,df2, by = "ID",all = FALSE)
```

<br>

img5

<br>

# 12. group_by ( feat `dplyr`)

- grouping

```R
group_df = df %>%
  group_by(var1,var2) %>%
  summarise(new_var1 = round(mean(var3)),
            new_var2 = round(median(var4)),
            new_var3 = round(max(var5)),
            new_var4 = length(var6))
```

<br>

- un-grouping

```R
ungroup_df = group_df %>% ungroup()
```

<br>

# 13. mutate

for 새로운 변수 생성

- ex) **mutate()**, **mutate_if()**, **mutate_at()**

<br>

## 1) mutate

1개의 변수 추가

```R
df2 = df %>%
 mutate(new_var = round(var1/var2,2)) %>%
 select(var3,var4,var5)
```

<br>

## 2) mutate_if

 지정해준 모든 변수에 대해, 조건 만족 시,  적용

```R
# ex) integer 타입 변수를 모두 numeric으로 변경

df2 = df %>% mutate_if(is.integer,as.numeric) 
```

<br>

## 3) mutate_at

 지정해준 모든 변수에 대해 적용

```R
df2 = df %>%
 mutate_at(vars(-var1,-var2,-var3),log) %>%
 select_if(is.numeric)
```

