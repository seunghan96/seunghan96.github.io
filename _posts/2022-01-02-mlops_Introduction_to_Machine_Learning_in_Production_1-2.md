---
title: \[Week 1-1\] Introduction
categories: [MLOPS]
tags: []
excerpt: (coursera) Introduction to ML in production \n - Overview of the ML Lifecycle and Deployment

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# ML engineering production, 



# ML Project Life Cycle

## [1] Welcome

After training ML model... now what?

$\rightarrow$ useless, unless **how to put them into production**!

<br>

Goal 

- not just build a ML model,
- but also **put them into production**

$\rightarrow$ learn **entire life cycle** of ML project

<br>

Example )

- use CV to inspect phone defects

  ( put a bounding box around them )

- with data set of scratched phones, maybe able to train a model

But.... what to do to put this into production deployment? 

<br>

Process

- prediction server is to accept API calls,

  ( prediction server = cloud, edge device ... )

- receive an image,

- make a decision as to whether or 

![figure2](/assets/img/mlops/img3.png)

<br>

Then...*what's the problem??*

- trained model might have worked well on your test set
- BUT.... ***real life  production deployment might gives the model much darker images!*** ( = data drift / concept drift )

![figure2](/assets/img/mlops/img4.png)

<br>

Much more problem in real world !!

$\rightarrow$ will learn **lot of important practical  things for building ML systems that work not just in the lab, "but in a production deployment environment"**

 <br>

Another challenge : ***takes a lot more than ML code***

- ML model codel = just only 5-10%

![figure2](/assets/img/mlops/img5.png)

<br>

## [2] Steps of an ML Project

![figure2](/assets/img/mlops/img6.png)

<br>

## [3] Case Study : speech recognition

### step 1) scoping

![figure2](/assets/img/mlops/img7.png)

- define task (project) = Speech Recongition
  - X = voice
  - y = text
- decide key metrics
  - ex) accuracy, latency, throughput
- estimate resources & timeline

<br>

### step 2) data

![figure2](/assets/img/mlops/img8.png)

- define data
  - Q) is the data labeled "consistently" ?
    - ex) with same voice, the label might be..
      - a) "Um, the weather is"
      - b) "Um... the weather is"
      - c) "The weather is"
  - Q) how much silence before/after the clip?
  - Q) how to perform "volume normalization"?

<br>

### step 3) Modeling

![figure2](/assets/img/mlops/img9.png)

ML model = (a) + (b) + (c)

- (a) code
- (b) hyperparameters
- (c) data

<br>

Research/Academia's task

- **"DATA" is fixed** & change "code" & "hyperparameters"

Product team' task

- **"CODE" is fixed** & change "data" & "hyperparameters"

<br>

### step 4) Deployment

![figure2](/assets/img/mlops/img10.png)

<br>

## [4] Course Outline

review : **ML project life cycle. **

preview : will learn...

- starting from the end goal "deployment", and then move toward "scoping" 
- ( deployment $\rightarrow$ modeling $\rightarrow$ data $\rightarrow$ scoping )