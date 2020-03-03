---
layout: splash
permalink: /

header:
  overlay_filter: "0.2"
  overlay_image: /assets/img/main.jpg
excerpt: "Studying Data Science at Yonsei University"

feature_row2:
  - image_path: /assets/img/ML.jpg
    alt: "placeholder image 2"
    title: "Machine Learning"
    excerpt: 'About Various ML Algorithms..'
    url: "categories/ml/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/DL.jpg
    alt: "placeholder image 2"
    title: "Deep Learning"
    excerpt: 'Natural Language Processing, Computer Vision..'
    url: "categories/dl/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/RL.jpg
    alt: "placeholder image 2"
    title: "Reinforcement Learning"
    excerpt: 'Q-Learning, DQN..'
    url: "categories/rl/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/ST.jpg
    alt: "placeholder image 2"
    title: "Statistics"
    excerpt: 'Bayesian Statistics'
    url: "/categories/st/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/etc.jpg
    alt: "placeholder image 2"
    title: "Others"
    excerpt: 'Projects, Competetion..'
    url: "/categories/etc/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
    
feature_row:
  - image_path: /assets/img/SeunghanLee2.jpg
    alt: "placeholder image 2"
    title: "Seunghan Lee"
    excerpt: 'Yonsei Univerisity <br> (major 1) Business Administration <br> (major 2) Applied Statistics <br> Data Science Lab <br> <br> T. 010-8768-8472 <br> E. seunghan9612@gmail.com'
    url: "https://github.com/seunghan96"
    btn_label: "Read More"
    btn_class: "btn--primary"
    

---

{% include feature_row id="intro" type="center"%}

{% include feature_row id="feature_row2" %}

{% include feature_row id="feature_row" type="left" %}
