완성된 함수 **imbalance_by_inverse_frequency_with_strength** 는 이제 alpha 파라미터를 통해 **불균형 강도(imbalance severity)** 를 조절할 수 있습니다.



------





## **✅ 작동 개요**





1. train_test_split으로 전체 데이터를 5:5로 나눕니다 (stratify=y로 클래스 균형 유지).

2. Train set에서:

   

   - 전체 데이터의 클래스 빈도를 기준으로
   - **(1 / frequency) \** alpha** 를 계산하여
   - 클래스별 oversampling 비율 결정
   - 빈도가 적은 클래스일수록 더 많이 복제됨 (alpha ↑ 일수록 더 심함)

   





------





## **🎛️ 파라미터 요약**



| **파라미터**  | **설명**                                       |
| ------------- | ---------------------------------------------- |
| test_size=0.5 | test 비율 (기본 50%)                           |
| alpha=1.0     | 불균형 강도 alpha=0: 균형 alpha ↑: 불균형 심화 |
| random_state  | 랜덤 시드                                      |



------





## **✅ 사용 예시**



```
X_train_imb, X_test, y_train_imb, y_test = imbalance_by_inverse_frequency_with_strength(X, y, alpha=2.0)
```



- alpha = 0.0 → 모든 클래스 동일하게 샘플링됨 (균형)
- alpha = 1.0 → 빈도가 적은 클래스가 더 많이 포함됨
- alpha = 2.0 이상 → 심각한 불균형 유도됨





------





## **📈 원하시면**





- 클래스 분포 barplot (train vs test)
- alpha에 따른 불균형도 변화 시각화





도 추가로 도와드릴 수 있습니다. 원하시나요? 



함수 imbalance_by_inverse_frequency_with_strength_min10는 다음과 같이 완성되었습니다:



------





## **✅ 기능 요약**





이 함수는 다음의 3단계 절차를 따릅니다:





### **1. 데이터 5:5로 나누기**





- train_test_split(test_size=0.5, stratify=y)로 **클래스 분포를 유지한 상태**로 분할







### **2. Train 데이터에 불균형 유도**





- 전체 데이터의 클래스 빈도를 바탕으로,
- 각 클래스에 대해 inverse frequency ^ alpha 값을 기반으로 oversampling 비율 결정
- **각 클래스 최소 10개 이상 보장**







### **3. Test 데이터에도 클래스당 최소 10개 보장**





- 부족한 클래스는 **복제(중복 허용)** 을 통해 채움





------





## **⚙️ 주요 파라미터**



| **파라미터**   | **설명**                                          |
| -------------- | ------------------------------------------------- |
| alpha=0.0      | 균형 (모든 클래스 동일하게 복제)                  |
| alpha=1.0 이상 | 빈도 적은 클래스일수록 더 많이 복제 → 불균형 심화 |
| min=10         | 각 클래스가 train/test에 최소 10개는 있도록 보장  |



------





## **✅ 사용 예시**



```
X_train_imb, X_test, y_train_imb, y_test = imbalance_by_inverse_frequency_with_strength_min10(
    X, y, alpha=2.0
)
```



------



필요하시면:



- train/test 클래스 분포를 시각화하거나
- alpha 값에 따른 imbalance 변화 실험용 loop





도 추가로 도와드릴 수 있습니다. 



좋습니다! 아래는 imbalance_by_inverse_frequency_with_strength 함수의 **수식 표현**입니다. 여기서는 min 10개 보장 조건은 제외하고, **클래스별 inverse frequency를 기반으로 oversampling 비율을 조절하는 방식**만 수식화하였습니다.



------





## **📘 수식 정의**







### **주어진 정보**





- 전체 데이터셋:

  \mathcal{D} = \{(x_i, y_i)\}_{i=1}^N

- 클래스 집합:

  \mathcal{C} = \{c_1, c_2, \dots, c_K\}

- 클래스 c_k의 빈도 (전체 데이터 기준):

  f_k = \frac{|\{i : y_i = c_k\}|}{N}





------





## **🧠 Oversampling 비율 계산**

### **1. 각 클래스의 가중치 정의 (Inverse frequency with strength α)**

$w_k = \left(\frac{1}{f_k}\right)^\alpha$

- $\alpha$ = 0이면 균등 가중치 → 모든 클래스 동일
- $\alpha > 0$이면 rare class일수록 더 높은 가중치





### **2. 정규화된 가중치**

$\tilde{w}_k = \frac{w_k}{\sum_{j=1}^K w_j}$.





### **3. 클래스별 oversampling 수 (train 총 샘플 수를** N_{\text{train}}**이라 할 때)**

$n_k = \left\lfloor \tilde{w}_k \cdot N_{\text{train}} \right\rfloor$.

- 각 클래스 $c_k$에 대해, $n_k$개 샘플을 복제 추출 (with replacement)
- 전체적으로는 $\sum_{k=1}^K n_k \approx N_{\text{train}}$





## **✅ 결과**

- train set은 $\{(x_i, y_i)\}$ 중 oversampled된 n_k개씩 클래스별 포함
- test set은 train_test_split을 통해 무작위 균등 분할됨







나는 tabular data의 y class label의 distribution shift 상황을 가정하고자, 기존의 데이터셋의 분리 방식에 변화를 주었어.

- (기존) train:test = 50:50으로 나눔.
  - 따라서, estimated 값으로, train과 test의 y label 분포는 같을 것.
- (변화) 아래의 `imbalance_by_inverse_frequency_with_strength_min10` 함수
  - Step 1) train:test = 50:50으로 나눔.
    - (기존)과 동일하게 50:50으로 나누는 이유는, 최종적으로 test 성능을 측정할 때, (기존)의 방식과 차이가 나지 않게 하기 위해서, test는 최대한 변화를 주지 않게 하려고 했음.
  - Step 2) train 데이터셋을 oversampling 하는 방식으로 imbalance를 유도함.



아래의 `imbalance_by_inverse_frequency_with_strength_min10` 를 통해, 내가 어떠한 식으로 변화를 주었는지

- (1) 수식 (Mathematical expression)으로 설명
- (2) 제안한 분리방식의 변화에 대한 영어 설명 (in paragraph)
- (3) 제안한 분리방식의 변화에 대한 영어 설명 (in brief itemized format)
- (4) 제안한 분리방식의 변화에 대한 영어 설명 (in detailed itemized format)

이 네 가지를 차례로 설명해줘.



이제 `imbalance_by_inverse_frequency_with_strength_min10` 를 보여줄게.

```
def imbalance_by_inverse_frequency_with_strength_min10(X, y, test_size=0.5, alpha=1.0, random_state=42):
    """
    Step 1: Split data into train/test using scikit-learn's train_test_split with 50:50 ratio.
    Step 2: Induce class imbalance in the training set by oversampling inversely to class frequency^alpha,
            while ensuring each class has at least 10 samples in both train and test sets.

    Parameters:
    - X, y: input features and labels
    - test_size: proportion for the test set (default 0.5)
    - alpha: imbalance strength (0 = uniform, higher = stronger imbalance)
    - random_state: random seed for reproducibility

    Returns:
    - X_train_imbalanced, X_test_final, y_train_imbalanced, y_test_final
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import numpy as np

    X = np.asarray(X)
    y = np.asarray(y)

    # Step 1: 50-50 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Compute inverse-frequency^alpha weights from full data
    class_labels, counts = np.unique(y, return_counts=True)
    freqs = counts / counts.sum()
    imbalance_weights = (1.0 / freqs) ** alpha
    imbalance_weights = imbalance_weights / imbalance_weights.sum()

    # Determine number of samples to draw from each class in train set
    total_train_samples = len(y_train)
    samples_per_class = {
        cls: max(10, int(imbalance_weights[i] * total_train_samples))
        for i, cls in enumerate(class_labels)
    }

    # Oversample train set
    X_train_imbalanced, y_train_imbalanced = [], []
    rng = np.random.default_rng(random_state)
    for cls in class_labels:
        cls_indices = np.where(y_train == cls)[0]
        n_samples = samples_per_class[cls]

        if len(cls_indices) == 0:
            continue

        resampled_idx = rng.choice(cls_indices, size=n_samples, replace=True)
        X_train_imbalanced.append(X_train[resampled_idx])
        y_train_imbalanced.append(y_train[resampled_idx])

    # Step 3: Filter test set to ensure at least 10 samples per class
    X_test_filtered, y_test_filtered = [], []
    for cls in class_labels:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) >= 10:
            X_test_filtered.append(X_test[cls_indices])
            y_test_filtered.append(y_test[cls_indices])
        else:
            # Fill with duplicated samples if fewer than 10
            resampled_idx = rng.choice(cls_indices, size=10, replace=True)
            X_test_filtered.append(X_test[resampled_idx])
            y_test_filtered.append(y_test[resampled_idx])

    return (
        np.vstack(X_train_imbalanced),
        np.vstack(X_test_filtered),
        np.hstack(y_train_imbalanced),
        np.hstack(y_test_filtered)
    )
```





아래는 주신 함수 imbalance_by_inverse_frequency_with_strength_min10 에 대한 요청한 네 가지 항목입니다:



------





## **✅ (1) 수식 표현 (Mathematical Expression)**

주어진 전체 클래스 분포 $p(y = c_k) = f_k$ (클래스 빈도 비율) 에 대해,

훈련 데이터의 클래스 $c_k$에 대한 **oversampling 수**는 다음과 같이 계산됩니다:



w_k = \left( \frac{1}{f_k} \right)^\alpha, \quad \tilde{w}k = \frac{w_k}{\sum{j} w_j}



n_k^{\text{train}} = \max\left(10, \left\lfloor \tilde{w}k \cdot N{\text{train}} \right\rfloor\right)



- f_k: 전체 데이터에서 클래스 c_k의 비율
- \alpha: imbalance 조절 강도
- N_{\text{train}}: split 이후 train set의 총 샘플 수
- n_k^{\text{train}}: oversampling된 train 데이터에서 클래스 c_k의 샘플 수





→ 이로 인해 실제 train 분포는:



\hat{p}_{\text{train}}(y = c_k) = \frac{n_k^{\text{train}}}{\sum_j n_j^{\text{train}}}



※ test set은 stratified split 후 각 클래스 최소 10개 이상이 되도록 보장 (resample 포함)



------





## **✅ (2) Paragraph 설명 (in English)**





In the original setup, the dataset was divided evenly into training and testing sets (50:50), preserving the label distribution in both subsets. To simulate a label distribution shift while keeping the test distribution consistent, we introduced a new data splitting scheme via the imbalance_by_inverse_frequency_with_strength_min10 function. First, a stratified 50:50 split ensures a fair test set. Then, class imbalance is artificially introduced into the training set by oversampling classes inversely proportional to their original frequencies raised to a tunable power \alpha. As a result, rare classes become overrepresented in the training set while the test set remains close to the original distribution, thereby simulating a label shift scenario between training and testing.



------





## **✅ (3) Brief Itemized Summary (in English)**





- Original split: 50:50 train:test with preserved class distribution

- New split:

  

  - Still 50:50 to keep test evaluation comparable
  - Induce imbalance in train via inverse-frequency-based oversampling

  

- Rare classes are overrepresented in train

- Test remains close to original label distribution





------





## **✅ (4) Detailed Itemized Explanation (in English)**



\begin{itemize}
    \item \textbf{Original splitting:} The dataset is divided into training and test subsets with a 50:50 ratio using stratified sampling, ensuring the label distributions are approximately equal.
    \item \textbf{Motivation for change:} To simulate label distribution shift between training and test without altering the test set too much, while still allowing fair evaluation.
    \item \textbf{Step 1 (Split):} The 50:50 stratified split is retained to ensure the test set reflects the same distribution as the original data.
    \item \textbf{Step 2 (Train imbalance):} The training set is oversampled such that the number of samples per class is proportional to the inverse of their original class frequencies raised to the power of \( \alpha \). This makes rare classes more frequent in the training set.
    \item \textbf{Step 3 (Test guarantee):} The test set is filtered (or resampled if necessary) to guarantee a minimum of 10 samples per class, preserving class diversity.
    \item \textbf{Result:} The label distribution in the training set is skewed (according to \( \alpha \)), while the test set remains balanced, effectively simulating a distribution shift setting.
\end{itemize}



------

# 최종

## Label Shift Simulation via Inverse Frequency Oversampling

To simulate a distribution shift between training and test sets, we modify the standard stratified split by applying inverse-frequency-based oversampling to the training set. The process is as follows:

Given an original dataset with features $X$ and labels $y$, we first split it into training and test sets using a stratified 50:50 split, following the previous works. 

Let $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$ denote the set of class labels, and $f_k$ be the empirical frequency of class $c_k$ in the full dataset:
$f_k = \frac{|\{i : y_i = c_k\}|}{N}$, where $N$ is the total number of samples.

To induce class imbalance in the training set, we define oversampling weights using an inverse frequency scaling:
$w_k = \left( \frac{1}{f_k} \right)^\alpha, \quad \tilde{w}_k = \frac{w_k}{\sum_{j=1}^{K} w_j}$,
where $\alpha \geq 0$ controls the imbalance strength. 

A higher value of $\alpha$ leads to stronger overrepresentation of rare classes.

The number of training samples drawn for each class is then given by:
$n_k^{\text{train}} = \max\left(10, \left\lfloor \tilde{w}_k \cdot N_{\text{train}} \right\rfloor \right)$,
ensuring that each class has at least 10 samples. 

Oversampling is performed with replacement.

Meanwhile, the test set remains close to the original label distribution due to the stratified split. To ensure adequate representation, we resample any class with fewer than 10 test instances to guarantee a minimum of 10 per class.

This results in a training set with an intentionally skewed label distribution, while maintaining a balanced test set, thereby creating a controlled label shift scenario.



# 최최종

### Inducing Label Distribution Shift via Controlled Oversampling

- [개요] To simulate label distribution shift between training and test data,  we construct a procedure that selectively alters the class distribution of the training set while preserving that of the test set. 

- [Step 1] First, the original dataset $(X, y)$ is split into training and test subsets using a stratified 50:50 split, following prior work to ensure that the test set preserves the original label distribution and enables fair comparison across experimental settings. 
- [Step 2] After the split, we induce a distribution shift between the training and test label distributions by oversampling each class in the training set according to the inverse of its empirical frequency.



[Step2 -detail] 

- Specifically, let $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$ denote the set of class labels, and $f_k$ be the relative frequency of class $c_k$ in the original dataset: $f_k = \frac{|\{i : y_i = c_k\}|}{N}$ where $N$ = total number of samples.
- To control the degree of shift, we compute the normalized oversampling weights as: $w_k = \left( \frac{1}{f_k} \right)^\alpha$,  $\tilde{w}_k = \frac{w_k}{\sum_{j=1}^{K} w_j}$ where $\alpha \geq 0$ is a scalar hyperparameter.  We then draw $n_k^{\text{train}} = \left\lfloor \tilde{w}_k \cdot N_{\text{train}} \right\rfloor $ samples for each class $k$ in the training set.
- The result is a training set whose class distribution intentionally deviates from the original, while the test set remains stratified and optionally resampled to ensure a minimum of 10 examples per class.




The parameter $\alpha$ serves as an **imbalance strength factor**, controlling how strongly rare classes are emphasized in the training set. 

- When $\alpha = 0$, each class is sampled uniformly, maintaining a balanced training distribution. 
- As $\alpha$ increases, the sampling becomes more biased toward rare classes, amplifying the disparity between training and test label distributions. 
- Thus, larger values of $\alpha$ correspond to more severe label shift scenarios.



```


```



N_train 인덱스 뽑아서 inference 시에도 적용

어떻게 고정되게?

<img src="/Users/seunghan96/Library/Application Support/typora-user-images/image-20250622161805417.png" alt="image-20250622161805417" style="zoom:50%;" />
