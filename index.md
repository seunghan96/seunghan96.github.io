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
    # excerpt: ' '
    url: "categories/ml/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/DL.jpg
    alt: "placeholder image 2"
    title: "Deep Learning"
    # excerpt: ' '
    url: "categories/dl/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
  - image_path: /assets/img/RL.png
    alt: "placeholder image 2"
    title: "Reinforcement Learning"
    # excerpt: ' '
    url: "categories/rl/"
    btn_label: "Read More"
    btn_class: "btn--primary"

  - image_path: /assets/img/ST.jpg
    alt: "placeholder image 2"
    title: "Statistics"
    # excerpt: ' '
    url: "/categories/st/"
    btn_label: "Read More"
    btn_class: "btn--primary"
    
    
feature_row:
  - image_path: /assets/img/pig.jpg
    alt: "placeholder image 2"
    title: "Seunghan Lee"
    excerpt: '연세대학교 경영학과, 응용통계학과 <br> DataScience 학회 DSL 소속 <br> T. 010-8768-8472 <br> E. seunghan9612@gmail.com'
    url: "https://github.com/seunghan96"
    btn_label: "Read More"
    btn_class: "btn--primary"
    

---

{% include feature_row id="intro" type="center"%}

{% include feature_row id="feature_row2" %}

{% include feature_row id="feature_row" type="left" %}
