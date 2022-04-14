---
title: \[Week 2-3\] Feature Selection
categories: [MLOPS]
tags: []
excerpt: (coursera) Machine Learning Data Lifecycle in Production - Feature Engineering, Transformation and Selection
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( reference : [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) ) 

# Feature Selection

## [1] Feature Spaces

Outline

- What is Feature Space
- Introduction to Feature Selection
- Feature Selection Methods
  - (1) filter methods
  - (2) wrapper methods
  - (3) embedded methos

<br>

### Feature Space

- $N$ features $\rightarrow$ $N$ dimension feature space

  ( do not include target label )

<br>

### Feature Space Coverage

***IMPORTANT*** : ensure feature space coverage!!

- **train & evaluation datasets’** feature coverage $\approx$ **serving dataset’s** feature coverage

- same feature coverage  =
  - same numerical ranges
  - same classes
  - similar characteristics of data ( IMAGE )
  - simliar vocab, syntax, semantics ( NLP )

<br>

Challenges

- Data is affected by …. seasonality / trend / drift
- New values in features & labels in serving data

$\rightarrow$ thus, need **CONTINUOUS MONITORING**

<br>

## [2] Feature Selection

- identify features’ correlation
- remove unimportant features
- reduce dimension of feature space

<br>

Why??

 ![figure2](/assets/img/mlops/img106.png)

<br>

### Feature Selection methods

1. Unsupervised
   - remove redundant features
2. Supervised
   - select most contributing features
   - ex) filter methods, wrapper methods, embedded methods

## [3] Filter Methods

 ![figure2](/assets/img/mlops/img107.png)

<br>

Filter Methods

- (1) correlation
- (2) univariate feature selection

<br>

### (1) correlation

- high correlation $\rightarrow$ redundant information
- draw correlation matrix
- ex) Pearson Correlation, Univariate Feature Selection

<br>

### (2) Univariate feature selection

( feat. Sklearn )

Univariate Feature Selection Routines

- `SelectKBest`
- `SelectPercentile`
- `GenericUnivariateSelect`

<br>

Statistical tests

- Regression : `f_regression`, `mutual_info_regression`
- Classification : `chi2`, `f_classif`, `mutual_info_classif`

<br>

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=20)

X_new = selector.fit_transform(X, y)
X_new_feature_idx = selector.get_support()
```

<br>

## [4] Wrapper Methods

 ![figure2](/assets/img/mlops/img108.png)

<br>

Wrapper Methods

- (1) Forward Selection
- (2) Backward Elimination
- (3) Recursive Feature Elimination

<br>

### (1) Forward Selection

Iterative, Greedy method

1. start with 1 feature

2. evaluate model performance,

   when **ADDING each of additional features** ( one at a time )

3. add next feature,

   that gives **the BEST performance**

4. Repeat until no improvement

<br>

### (2) Backward Elimination

1. start with ALL features

2. evaluate model performance,

   when **REMOVING each feature** ( one at a time )

3. Remove next feature,

   that gives **the WORST performance**

4. Repeat until no improvement

<br>

### (3) Recursive Feature Elimination

1. select model ( JUST FOR measuring FEATURE IMPORTANCE )

2. select desired \# of features
3. fit the model
4. **Rank features by importance**
5. discard least important features
6. repeat until desired \# of features remains

<br>

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

K = 20
model = RandomForestClassifier()
rfe = RFE(model, K)
rfe = rfe.fit(X,y)

X_features_idx = rfe.get_support()
X_features = X.columns[X_features_idx]
```

<br>

## [5] Embedded Methods

Embedded Methods

- (1) L1 regularization
- (2) Feature Importance
  - scores for each feature in data
  - discard features with low feature importance

```python
model = RandomForestClassifier()
model = model.fit(X,y)
FI = model.feature_importances_

print(FI)
FI.nlargest(10).plot(kind='barh')
```

