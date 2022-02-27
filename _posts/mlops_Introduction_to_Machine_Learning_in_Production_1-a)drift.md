# Concept & Data Drift

(reference : https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb)

<br>

Problem : corrupted, late, or incomplete data

$\rightarrow$ solved...then is it OK? that's not all !!

<br>

Contents

1. Model decay
2. Data drift
3. Concept drift
4. How to deal with drift?

<br>

# 1. Model decay

***Past performance is no guarantee of future results***

= model drift / model decay / staleness

<br>

Reason :

- 1) data drift
- 2) concept drift

<br>

Retraining might help!

![figure2](/assets/img/mlops/img19.png)

<br>

# 2. Data Drift

( Data drift = feature drift = population/covariate shift )

***input data has changed!***

- old model might not be suitable for new data!

<br>

Example 1) online advertising

- task : want to predict **how likely they will make a purchase**

- feature distribution of "source channel" might change over time!

![figure2](/assets/img/mlops/img20.png)

<br>

Example 2) Demographic change

- people get old over time!

![figure2](/assets/img/mlops/img21.png)

<br>

"Degree of decay" depends on the task!

<br>

## Training-serving skew

Cause of skew is different from data drift!

( actually, there is no "drift", but more like "mismatch" )

<br>

Example )  

- TRAIN on artificially constructed or cleaned dataset
- INFERENCE on real-world dataset

![figure2](/assets/img/mlops/img22.png)

<br>

# 3. Concept Drift

***patterns ( relation of X & Y ) the model learned changes***

## a) gradual concept drift

![figure2](/assets/img/mlops/img23.png)

follows the gradual changes in "external factors"

examples )

- competitor launches new products
- macroeconomic conditions change

individual change might be small, but big as a whole

<br>

### b) sudden concept drift

![figure2](/assets/img/mlops/img24.png)

example) COVID-19

- shopping patterns changed suddenly!

  $\rightarrow$ change in "demand forecast"

<br>

# 4. How to deal with drift?

Need to "RETRAIN" the model!

- method 1) Retrain the model using all available data
- method 2) Retrain the model using all available data + higher weight on new data
- method 3) Retrain the model using NEW data

<br>

other options

- domain adaptations
- building a composition of models taht use BOTH old & new data
- entirely new architecture
- ...

<br>

