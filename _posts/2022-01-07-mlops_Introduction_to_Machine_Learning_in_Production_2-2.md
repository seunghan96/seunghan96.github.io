---
title: \[Week 2-2\] Error analysis and performance auditing
categories: [MLOPS]
tags: []
excerpt: (coursera) Introduction to ML in production - 2.Select and Train a Model
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Error analysis and performance auditing



## [1] Error Analysis example

Error Analysis

- tell you what's to do to improve algorithm's performance

<br>

### Ex ) speech recognition

![figure2](/assets/img/mlops/img37.png)

<br>

Error Analysis is an **iterative process**

![figure2](/assets/img/mlops/img38.png)

- during of error analysis, can also add additional tags!

- go back to see if some of the other examples have added tags!

<br>

### Ex ) visual inspection

finding defects in smart phones

example of tags

- specific class labels ( ex. scratch, dent ... )
- image properties ( ex. blurry, dark ... )
- other meta data ( ex. phone model, factory ... )

<br>

### Ex ) product recommendation

example of tags

- user demographics
- product features / category

<br>

### Useful metrics for each tag

1. what % of errors has that tag?
2. of data with that tag, what % is misclassified?
3. what % of all data has that tag?
4. how much room for improvement is there on the data with that tag?

<br>

## [2] Prioritizing what to work on

![figure2](/assets/img/mlops/img39.png)

right most column : contribution to raising average accuracy

<br>

Which category to focus on?

- 1) how much room for improvement?
- 2) how frequently that category appears?
- 3) how easy to improve accuracy?

<br>

After choosing which category to focus on....

- 1) collect more data!
- 2) data augmentation
- 3) improve label accuracy / data quality

<br>

## [3] Skewed datasets

Skip

- accuracy / precision / recall / F1 score ...

<br>

## [4] Performance auditing

even though well on accuracy/F1 score....

**"performance audit"** before pushing it to production!

$$\rightarrow$$ might save you from significant **post deployment problems**

<br>

Double check your system!

( accuracy, fairness/bias, etc ... )

- step 1) brainstorm the ways the system might go wrong
  - performance on "subsets of data"
    - ex) gender, age, ethnicity..
  - how common are certain errors
    - ex) FP, FN
  - performance on rare cases
- step 2) establish metrics to assess the performance of those issues
  - performance on slices of the data ( not on entire dev set )
  - after establishing metrics... MLOps can help automatic evaluation!
    - ex) TFMA ( Tensorflow model analysis )
- step 3) buy-in from the business of the product owner

<br>

Example ) speech recongition

- step 1) brainstorm the ways the system might go wrong
  - ex) accuracy on different genders/ethnicities
  - ex) accuracy on different device
  - ex) prevalence of rude mis-transcripition
    - GAN (generative adversarial network) $$\rightarrow$$ gang? gun?
- step 2) establish metrics to assess the performance of those issues
  - ex) mean accuracy on different genders/ethnicities
  - ex) mean accuracy on different device
  - ex) checking rude words