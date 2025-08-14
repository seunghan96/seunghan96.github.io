# Drift에 강한 TabPFN



# [1] LocalPFN의 수식

[Input] Entire training dataset + test dataset

- $\mathcal{D}_{\text{train}} \triangleq \left\{(x^{i}_{\text{train}}, y^{i}_{\text{train}})\right\}_{i=1}^{N}$.
  - Feature-label pairs $x^{i}_{\text{train}} \in \mathbb{R}^D$ and $y^{i}_{\text{train}} \in \{1, \ldots, C\}$ 
  - Query point $x_{\text{qy}}$ (potentially in a batch)

[Output] Distribution over labels $y_{\text{qy}} \in \{1, \ldots, C\}$. 

<br>

Posterior predictive distribution

- $p_\theta(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}}) = \frac{\exp\left(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[y_{\text{qy}}]\right)}{\sum_{c=1}^C \exp\left(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[c]\right)}$.
  - where [·] denotes the vector indexing operation




# [2] LocalPFN 수식 기반으로 adjust

- $p_\theta(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}}) = \frac{\exp\left(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[y_{\text{qy}}]\right)}{\sum_{c=1}^C \exp\left(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[c]\right)} \times \frac{\textbf{Test prior(= (b))}}{\textbf{Train prior(= (a))}}$.



# [3] GPT 수식 기반으로 adjust

## (1) Detail

$
p^{\text{adjusted}}_\theta(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}})
= \underbrace{\frac{\exp(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[y_{\text{qy}}])}
                   {\sum_{c=1}^C \exp(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[c])}}_{\text{Original Posterior}}
\times
\underbrace{\frac{\hat{p}_{\text{test}}(y_{\text{qy}})}
                 {\hat{p}_{\text{train}}(y_{\text{qy}})}}_{\text{Prior Adjustment}}
$

<br>

## (2) Brief

$p^{\text{adj}}_\theta(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}}) = p_\theta^{\text{orig}}(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}}) \times \frac{p_{\text{test}}(y_{\text{qy}})}{p_{\text{train}}(y_{\text{qy}})}$

- $p^{\text{orig}}_{\theta}(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}}) = \frac{\exp(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[y_{\text{qy}}])} {\sum_{c=1}^C \exp(f_\theta(x_{\text{qy}}, \mathcal{D}_{\text{train}})[c])}$.

<br>

### Train Prior ($p_{\text{train}}$)

- **(a-1) 실제 라벨 분포**

  $p_{\text{train}}^{(\text{Gt})}(y=c) = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \mathbb{1}[y^{(i)}_{\text{train}} = c]$.

- **(a-2) 모델 예측 분포 평균**

  $p_{\text{train}}^{(\text{Avg. Pred})}(y=c) = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} p_\theta(y=c \mid x^{(i)}_{\text{train}}, \mathcal{D}_{\text{train}})$.

<br>

### Test Prior ($p_{\text{test}}$)

- **(b-1) 실제 라벨 분포**

  $p_{\text{test}}^{(\text{Gt})}(y=c) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{1}[y^{(i)}_{\text{test}} = c]$.

- **(b-2) 모델 예측 분포 평균**

  $p_{\text{test}}^{(\text{Avg. Pred})}(y=c) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} p_\theta(y=c \mid x^{(i)}_{\text{test}}, \mathcal{D}_{\text{train}})$.

- **(b-3) 모델 예측 분포 (query별 개별 posterior 사용)**

  $p_{\text{test}}^{(\text{Pred})}(y_{\text{qy}}) = p_\theta(y_{\text{qy}} \mid x_{\text{qy}}, \mathcal{D}_{\text{train}})$.





\begin{itemize}
    \item $p_{\text{train}}^{(1)}$: Prior estimated from the empirical label distribution of the training set.
    \item $p_{\text{train}}^{(2)}$: Prior estimated by averaging the model’s posterior predictions over all training inputs.
    \item $p_{\text{test}}^{(1)}$: Prior estimated from the empirical label distribution of the test set (requires access to ground-truth labels; not available in practice).
    \item $p_{\text{test}}^{(2)}$: Prior estimated by averaging the model’s posterior predictions over all test inputs.
    \item $p_{\text{test}}^{(3)}$: Prior estimated by using the model’s posterior prediction for each individual test input (per-query adjustment).

| Category | Types                                   | (1) Prior based on ..          | (2) Detailed Expression                                      | (3) Mathematical Expression |
| -------- | --------------------------------------- | ------------------------------ | ------------------------------------------------------------ | --------------------------- |
| ptrain   | $p_{\text{train}}^{(\text{Gt})}$        | Label ($y$)                    | Estimated from the empirical label distribution of the training set. | 1                           |
| ptrain   | $p_{\text{train}}^{(\text{Avg. Pred})}$ | Prediction ($\hat{y}$)         | Estimated by averaging the model’s posterior predictions over all training inputs. | 2                           |
| ------   | ------                                  | ------                         | ------                                                       | ------                      |
| ptest    | $p_{\text{test}}^{(\text{Gt})}$         | Label ($y$) -> unavailable     | Estimated from the empirical label distribution of the test set (requires access to ground-truth labels; not available in practice) | a                           |
| ptest    | $p_{\text{test}}^{(\text{Avg. Pred})}$  | Average Prediction ($\hat{y}$) | Estimated by averaging the model’s posterior predictions over all test inputs. | b                           |
| ptest    | $p_{\text{test}}^{(\text{Pred})}$       | Prediction ($\hat{y}$)         | Estimated by using the model’s posterior prediction for each individual test input (per-query adjustment). | c                           |
| ptest    | $p_{\text{test}}^{\text{(Uniform)}}$    | None                           | Uniform prior that assumes no prior information is available; each class is assigned equal probability. |                             |

|      | Train             | Test              | Shift X | Shift O | Efficient |
| ---- | ----------------- | ----------------- | ------- | ------- | --------- |
| v1   | Ground truth      | Uniform           | X       |         | O         |
| v4   | Ground truth      | Prediction (개별) | O       | OO      | O         |
| v6   | Ground truth      | Prediction (평균) | O       | O       | O         |
| v2   | Prediction (평균) | Uniform           | X       |         | X         |
| v3   | Prediction (평균) | Prediction (개별) | O       |         | X         |
| v5   | Prediction (평균) | Prediction (평균) | O       |         | X         |
|      |                   |                   |         |         |           |

```
python -m src.evals.eval_tabpfn_label_shift_v3_tau_auto_pq_dist
python -m src.evals.eval_tabpfn_label_wo_shift_v3_tau_auto_pq_dist
python -m src.evals.eval_tabpfn_label_shift_v5_tau_auto_pq_dist
python -m src.evals.eval_tabpfn_label_wo_shift_v5_tau_auto_pq_dist
```



## **🧩 6가지 조합 표 (모든 경우 수식 적용 가능)**



| **조합 번호** | **Train Prior** | **Test Prior**  | **Posterior 조정 수식**                                      |
| ------------- | --------------- | --------------- | ------------------------------------------------------------ |
| 1             | 실제 라벨 (a-1) | 실제 라벨 (b-1) | $\times \frac{p_{\text{test}}^{(1)}}{p_{\text{train}}^{(1)}}$ |
| 2             | 실제 라벨 (a-1) | 예측 평균 (b-2) | $\times \frac{p_{\text{test}}^{(2)}}{p_{\text{train}}^{(1)}}$ |
| 3             | 실제 라벨 (a-1) | 예측 개별 (b-3) | $\times \frac{p_{\text{test}}^{(3)}}{p_{\text{train}}^{(1)}}$ |
| 4             | 예측 평균 (a-2) | 실제 라벨 (b-1) | $\times \frac{p_{\text{test}}^{(1)}}{p_{\text{train}}^{(2)}}$ |
| 5             | 예측 평균 (a-2) | 예측 평균 (b-2) | $\times \frac{p_{\text{test}}^{(2)}}{p_{\text{train}}^{(2)}}$ |
| 6             | 예측 평균 (a-2) | 예측 개별 (b-3) | $\times \frac{p_{\text{test}}^{(3)}}{p_{\text{train}}^{(2)}}$ |



------





## **📘 주석 처리 (선택 사항)**



```
% p_train^{(1)} : train label 기반 prior
% p_train^{(2)} : train posterior 평균 기반 prior
% p_test^{(1)}  : test label 기반 prior (ground truth, 비현실적)
% p_test^{(2)}  : test posterior 평균 기반 prior
% p_test^{(3)}  : test query 개별 posterior 기반 prior
```

\begin{table}[ht]
\centering
\caption{Summary of six prior adjustment combinations based on different train/test prior definitions.}
\label{tab:prior_adjustment}
\begin{tabular}{clll}
\toprule
\textbf{Case} & \textbf{Train Prior} & \textbf{Test Prior} & \textbf{Adjustment Factor} \\
\midrule
1 & Empirical label distribution from train set
  & Empirical label distribution from test set
  & $\displaystyle \frac{p_{\text{test}}^{(1)}(y)}{p_{\text{train}}^{(1)}(y)}$ \\[6pt]
2 & Empirical label distribution from train set
  & Mean posterior from test set
  & $\displaystyle \frac{p_{\text{test}}^{(2)}(y)}{p_{\text{train}}^{(1)}(y)}$ \\[6pt]
3 & Empirical label distribution from train set
  & Per-query posterior from test set
  & $\displaystyle \frac{p_{\text{test}}^{(3)}(y_{\text{qy}})}{p_{\text{train}}^{(1)}(y_{\text{qy}})}$ \\[6pt]
4 & Mean posterior from train set
  & Empirical label distribution from test set
  & $\displaystyle \frac{p_{\text{test}}^{(1)}(y)}{p_{\text{train}}^{(2)}(y)}$ \\[6pt]
5 & Mean posterior from train set
  & Mean posterior from test set
  & $\displaystyle \frac{p_{\text{test}}^{(2)}(y)}{p_{\text{train}}^{(2)}(y)}$ \\[6pt]
6 & Mean posterior from train set
  & Per-query posterior from test set
  & $\displaystyle \frac{p_{\text{test}}^{(3)}(y_{\text{qy}})}{p_{\text{train}}^{(2)}(y_{\text{qy}})}$ \\
\bottomrule
\end{tabular}
\end{table}





아래는 주어진 주석들을 **영어로 서술형 문장**, **itemize**, **enumerate** 형식으로 각각 정리한 내용입니다. 논문이나 문서에서 바로 사용할 수 있도록 다듬었습니다.



------





## **📘 1. 줄글 설명 (Descriptive Text)**





We consider different ways to define class prior distributions from the train and test datasets for the purpose of posterior adjustment.

For the **training prior**, two variants are used:

- (1) the empirical label distribution derived directly from the training labels, and
- (2) the average posterior distribution obtained by applying the model to the training inputs.



For the **test prior**, three alternatives are considered:

- (1) the empirical label distribution computed from the test labels, which is typically unavailable in real-world deployment scenarios (thus considered unrealistic),
- (2) the average posterior distribution computed across all test inputs, and
- (3) the posterior distribution computed individually for each test input, used on a per-query basis.









## **📝 2. Itemize Version (Bullet List)**



\begin{itemize}
    \item $p_{\text{train}}^{(1)}$: Prior estimated from the empirical label distribution of the training set.
    \item $p_{\text{train}}^{(2)}$: Prior estimated by averaging the model’s posterior predictions over all training inputs.
    \item $p_{\text{test}}^{(1)}$: Prior estimated from the empirical label distribution of the test set (requires access to ground-truth labels; not available in practice).
    \item $p_{\text{test}}^{(2)}$: Prior estimated by averaging the model’s posterior predictions over all test inputs.
    \item $p_{\text{test}}^{(3)}$: Prior estimated by using the model’s posterior prediction for each individual test input (per-query adjustment).
\end{itemize}



------





## **🔢 3. Enumerated Version (Numbered List)**



\begin{enumerate}
    \item $p_{\text{train}}^{(1)}$: The class prior is computed from the label frequency distribution in the training set.
    \item $p_{\text{train}}^{(2)}$: The class prior is computed as the average of model-predicted posteriors over the training set.
    \item $p_{\text{test}}^{(1)}$: The class prior is computed from the label frequency distribution in the test set (ground-truth; unrealistic in practice).
    \item $p_{\text{test}}^{(2)}$: The class prior is computed as the average of model-predicted posteriors over the test set.
    \item $p_{\text{test}}^{(3)}$: The class prior is computed per query by using the model’s posterior for each test input individually.
  \ item A uniform prior that assumes no prior information is available; each class is assigned equal probability.

\end{enumerate}



------



이 표현들은 논문 본문, 부록, 또는 테이블 주석 등에 자연스럽게 포함될 수 있도록 다듬어져 있습니다. 필요시 더욱 포멀하게 또는 간결하게도 조정해 드릴 수 있어요.
