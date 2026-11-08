# Evaluation metrics

> **One-line summary**: For classification, look at the confusion matrix and threshold-dependent P/R/F1; for ranking, look at ROC-AUC and PR-AP; for regression, look at RMSE/MAE/R². Metrics must match the business objective and the data distribution, and offline metrics are not the same as online revenue.
>
> **Keywords**: `ROC-AUC` PR curve F1 NDCG MAPE calibration A/B testing

## 1. Core concepts

- **Offline metrics**: computed on labeled data for model selection and regression testing. **Online metrics**: CTR, CVR, dwell time, GMV, etc., measured in A/B tests.
- **Threshold-dependent metrics**: Precision, Recall, F1, and Accuracy depend on the classification threshold. **Threshold-independent metrics**: ROC-AUC and PR-AUC evaluate ranking quality.
- **Probabilistic model vs decision**: the model outputs probabilities, while the business needs decisions; the threshold defines the mapping between them and the P/R trade-off.

Binary confusion matrix:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| Actual positive | TP | FN |
| Actual negative | FP | TN |

## 2. Principles and derivations

### 2.1 Classification metrics

$$\text{Precision}=\frac{TP}{TP+FP}, \qquad \text{Recall}=\frac{TP}{TP+FN}, \qquad F_1=\frac{2PR}{P+R}$$

More generally, with a weight:

$$F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}$$

$\beta>1$ favors recall, while $\beta<1$ favors precision. Also:

$$\text{Specificity}=\frac{TN}{TN+FP}, \qquad \text{BalancedAcc}=\frac{\text{Recall}+\text{Specificity}}{2}$$

**Why accuracy can lie**: when only 1% of samples are positive, predicting all negative gives 99% accuracy but zero recall. Under imbalance, look at PR-AUC, F1, Recall at a fixed Precision, or cost-weighted metrics.

### 2.2 ROC and AUC

The ROC curve plots TPR (i.e., Recall) against FPR while sweeping all thresholds:

$$\text{TPR}=\frac{TP}{TP+FN}, \qquad \text{FPR}=\frac{FP}{FP+TN}$$

**Probabilistic interpretation of AUC**: the probability that a randomly drawn positive sample scores higher than a randomly drawn negative sample:

$$\text{AUC}=P(s^+>s^-)+\frac12P(s^+=s^-) =\frac{1}{mn}\sum_{i:y_i=1}\sum_{j:y_j=0}\left[\mathbb{1}(s_i>s_j)+\frac12\mathbb{1}(s_i=s_j)\right]$$

Using ranks: sort all samples by score and let $R_+$ be the sum of positive-sample ranks; then

$$\text{AUC}=\frac{R_+ - m(m+1)/2}{mn}$$

Sorting costs $O(N\log N)$ and a linear scan follows, which is more efficient than computing TPR/FPR point by point at every threshold.

**Properties**: AUC depends only on the ranking of scores and is invariant to monotone transforms. Under random resampling that changes the positive/negative ratio, the expected AUC stays roughly the same (it measures separability of the conditional distributions), but it **overestimates** the practical business value under class imbalance.

### 2.3 PR curve and AP

The PR curve plots Precision against Recall. Average precision (AP) approximates the area under the PR curve:

$$\text{AP}=\sum_{n}(R_n-R_{n-1})P_n$$

($P_n$ is usually the precision in that recall interval; sklearn's average_precision uses a step-wise integral, while the interpolated version uses the maximum precision to the right.)

**ROC-AUC vs PR-AUC**: when positives are rare, the denominator of FPR (the number of negatives) is huge, so a few FPs barely change FPR and the ROC curve still looks good; for Precision, the denominator is the number of predicted positives, so the impact of FPs is amplified. Therefore prefer PR-AUC **under imbalance or when positive hits matter**, and ROC-AUC when classes are roughly balanced and overall ranking is the concern.

### 2.4 Regression metrics

$$\text{MSE}=\frac1m\sum_i(y_i-\hat y_i)^2, \qquad \text{RMSE}=\sqrt{\text{MSE}}, \qquad \text{MAE}=\frac1m\sum_i|y_i-\hat y_i|$$

$$R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}$$

- **RMSE** penalizes large errors more heavily and has the same unit as $y$; **MAE** is robust to outliers, but its optimum is the least-absolute-deviation fit (the median).
- **MAPE** $=\frac{100\%}{m}\sum_i|y_i-\hat y_i|/|y_i|$: dimensionless and comparable across scales, but it explodes as $y_i\to0$ and is asymmetric between under- and over-prediction.
- **R²**: the improvement relative to "always predict the mean"; it is negative when the model is worse than the mean, which is entirely possible on a test set.
- **Huber / quantile loss**: Huber behaves like MSE for small residuals and like MAE for large ones; quantile loss $\rho_\tau(e)=e(\tau-\mathbb{1}(e<0))$ is used for prediction intervals or the median.

### 2.5 Ranking metrics

Let $rel_i$ be the relevance at position $i$:

$$\text{DCG@K}=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}, \qquad \text{NDCG@K}=\frac{\text{DCG@K}}{\text{IDCG@K}}$$

- **NDCG**: position discounting (later positions get less weight) plus graded relevance; IDCG is the DCG of the ideal ranking, so the value lies in [0,1]. Suitable for search and recommendation.
- **MAP**: computes AP per user and averages; it only distinguishes relevant from irrelevant and ignores position discounting.
- **MRR**: the mean reciprocal rank of the first relevant result; suitable for QA/retrieval with a single correct answer.
- **HR@K**: whether a relevant item appears in the top K; simple and intuitive.
- **GAUC**: computes AUC per user and averages weighted by impressions; common in recommendation and avoids scores being incomparable across users.

### 2.6 Online metrics and A/B testing

- **CTR** = clicks/impressions; **CVR** = conversions/clicks; **CTCVR** = conversions/impressions = CTR × CVR.
- A/B test: random assignment → observe metric differences → statistical significance. The confidence interval for the difference of two proportions is

$$\hat\Delta=\hat p_T-\hat p_C, \qquad SE=\sqrt{\frac{\hat p_T(1-\hat p_T)}{n_T}+\frac{\hat p_C(1-\hat p_C)}{n_C}}$$

$\hat\Delta\pm1.96\,SE$ is the 95% confidence interval. The sample size is determined by the baseline rate, the minimum detectable effect (MDE), the significance level, and statistical power; compute power before the experiment and do not peek repeatedly afterward.

**Common pitfalls**: novelty effects, seasonality, non-independent assignment (the same user in both groups), sample-ratio mismatch (SRM), multiple comparisons, and Simpson's paradox. Besides the core metric, set **guardrail metrics** (latency, complaint rate, revenue).

### 2.7 Metric selection quick reference

| Scenario | Primary metric | Notes |
| --- | --- | --- |
| Balanced binary classification | Accuracy, ROC-AUC | Simple and intuitive |
| Imbalanced binary classification | PR-AUC, F1, Recall@P | Focus on positive hits |
| Probability prediction | LogLoss, Brier, calibration curve | Probabilities must be calibrated |
| Multi-class | Macro-F1, Micro-F1 | Look at macro when classes are imbalanced |
| Regression | RMSE/MAE, R² | Use MAE when outliers are common |
| Search/recommendation | NDCG@K, MAP, MRR, HR | Focus on ranking positions |
| Online | CTR/CVR, A/B confidence interval | Pair with guardrail metrics |

## 3. Implementation notes

```python
import numpy as np
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, f1_score,
                             mean_absolute_error, r2_score, ndcg_score)

print(classification_report(y_test, y_pred, digits=3))
print("AUC:", roc_auc_score(y_test, y_score))
print("AP:", average_precision_score(y_test, y_score))

# Compute AUC by ranks: O(N log N)
def auc_by_rank(y, score):
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Ties: use average ranks (omitted here for illustration)
    m = y.sum(); n = len(y) - m
    return (ranks[y == 1].sum() - m * (m + 1) / 2) / (m * n)

# NDCG: relevance matrix of shape (n_samples, n_docs)
print("NDCG@10:", ndcg_score(y_true_rel, y_score, k=10))
print("MAE:", mean_absolute_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))
```

Key points: AUC/AP take **probabilities or scores**, so do not hard-threshold them to 0/1 first; specify the average parameter for multi-class metrics; and online experiment reports should include confidence intervals, not just point estimates.

## 4. Interview Q&A

### Q1. What is the probabilistic interpretation of AUC, and what is its computational complexity?
**Answer**: AUC is the probability that a randomly drawn positive sample scores higher than a randomly drawn negative sample (ties count as 0.5). It therefore depends only on ranking, not on a threshold. The rank formula is $\text{AUC}=(R_+-m(m+1)/2)/(mn)$: sorting costs $O(N\log N)$ and a linear scan follows; brute-force pairwise comparison is $O(mn)$.
**Follow-up**: Is AUC invariant to monotone transforms of the score? Yes, because it uses only ranks.

### Q2. Is AUC immune to class imbalance?
**Answer**: Two layers. When the positive/negative ratio is changed randomly (e.g., random undersampling), the expected AUC stays roughly the same, because it measures separability of the two score distributions. But when positives are rare in the business, a high AUC can correspond to very low Precision and limited practical value. So under imbalance, do not look only at AUC; look at PR-AUC, Recall at a fixed Precision, or cost-sensitive metrics.
**Follow-up**: Why can test-set AUC still change after undersampling? With finite samples the variance grows, and if sampling is not fully random (by difficulty/time), the distribution changes.

### Q3. Why is the PR curve preferred under class imbalance?
**Answer**: The denominator of FPR is the number of negatives; when negatives vastly outnumber positives, a few FPs barely move FPR and the ROC curve still looks good. The denominator of Precision is the number of predicted positives, so the impact of FPs is amplified and exposes the problem of "sacrificing a lot of precision for recall". If the business cares only about the positive class (fraud, disease), PR is closer to the decision.
**Follow-up**: What is the baseline for PR-AUC? The positive-class prevalence (the AP of a random model is roughly the prevalence), not 0.5.

### Q4. How should the classification threshold be chosen?
**Answer**: The threshold is not part of the model but part of the decision. Methods: choose on the validation set under a business constraint (e.g., maximize Precision subject to Recall ≥ 0.9), minimize cost ($\text{cost}=C_{FP}FP+C_{FN}FN$), or pick top-K by target hit rate. Choose the threshold on an independent validation set, never tune it on the test set; under class imbalance, prior adjustment or cost-sensitive learning can directly change the optimal threshold.
**Follow-up**: Threshold moving versus resampling, which is better? Threshold moving does not change model ranking, is interpretable and cheap; resampling changes the training distribution, may improve ranking, but distorts probabilities.

### Q5. Why does F1 use the harmonic mean rather than the arithmetic mean?
**Answer**: The harmonic mean is more sensitive to the smaller value: if either Precision or Recall is near 0, F1 is near 0, which harshly penalizes a lopsided model; the arithmetic mean hides this imbalance (e.g., with P=1 and R=0.01, the arithmetic mean is 0.505 while F1 is about 0.02). F1 is also the balance point between P and R, which makes it convenient for single-metric comparisons.
**Follow-up**: When should Fβ be used? When missing a positive is worse (e.g., disease screening), use $\beta>1$ to favor recall; when false alarms are worse (e.g., account bans), use $\beta<1$ to favor precision.

### Q6. What is the difference among macro, micro, and weighted averaging for multi-class metrics?
**Answer**: Macro computes the metric per class and then averages, weighting every class equally, so it exposes small-class performance; micro aggregates TP/FP/FN across classes and then computes, which equals overall accuracy for single-label multi-class (dominated by large classes); weighted averages by class sample count, sitting between the two and closer to the overall metric but hiding small classes. Under class imbalance with concern for small classes, use macro-F1.
**Follow-up**: What about multi-label tasks? Micro still uses global statistics, macro averages over labels, and sample-average averages over samples.

### Q7. What does R² mean, and why can it be negative?
**Answer**: $R^2=1-SS_{res}/SS_{tot}$ measures how much better the model is than "always predict the training mean"; ideally it is 1. If the model's squared error on the test set is larger than that of predicting the mean, $SS_{res}>SS_{tot}$ and R² is negative. A negative R² does not mean the code is wrong; it means the model is worse than the baseline on that data, which is common under distribution shift or severe overfitting.
**Follow-up**: Is R² the same as the squared correlation coefficient? For simple linear regression with an intercept they are equal; for multivariate or nonlinear models they are not necessarily equal.

### Q8. What is wrong with MAPE, and when should it not be used?
**Answer**: When $y_i$ is near 0 the denominator blows up and the metric becomes unstable; it is asymmetric between under- and over-prediction (underestimating by 50% and overestimating by 100% give different absolute percentages); it is sensitive to outliers; and it ignores scale/business cost. For demand forecasting, use sMAPE, WAPE ($\sum|e|/\sum|y|$), or quantile loss instead.
**Follow-up**: What is the advantage of WAPE? It is a total-volume relative error with total demand as the denominator, so it does not explode on a single small value and suits sparse sales.

### Q9. Why are NDCG's gain and discount designed this way, and how do you choose between NDCG and MAP?
**Answer**: The gain $2^{rel}-1$ makes high relevance grow exponentially, while the discount $1/\log_2(i+1)$ gives earlier positions more weight, matching the decay of user attention; IDCG normalization makes different queries comparable. MAP only distinguishes relevant from irrelevant and does not model position discounting, so it suits binary relevance; for graded relevance (e.g., 0/1/2/3) and position-sensitive settings, use NDCG.
**Follow-up**: What are NDCG's drawbacks? It is sensitive to annotation quality, and values at different K are not directly comparable.

### Q10. Offline AUC improved but online CTR did not. Why?
**Answer**: (1) Offline evaluation is misaligned with the online objective (AUC ranking vs CTR calibration); (2) training/serving feature inconsistency or data leakage inflates offline results; (3) position bias/impression bias means offline data are not randomly distributed; (4) feedback loops (the model only recommends what it has seen); (5) online latency or candidate-set changes; (6) experiment assignment or metric noise. Diagnose with feature consistency checks, online calibration, position-stratified evaluation, and counterfactual evaluation.
**Follow-up**: How do AUC and calibration relate? AUC only looks at ranking and does not guarantee calibrated probabilities; CTR bidding needs calibrated probabilities, so poor calibration can give high AUC but low revenue.

### Q11. How do you determine sample size and significance for an A/B test?
**Answer**: First set the baseline metric $p$, the minimum detectable relative lift, the significance level $\alpha$ (often 0.05), and power $1-\beta$ (often 0.8), then estimate the required sample per group with the two-proportion sample-size formula; the experiment duration should cover a full cycle (at least one week) and avoid novelty effects. Report confidence intervals and p-values, and account for multiple-comparison correction and SRM checks.
**Follow-up**: Why can't you repeatedly check significance mid-experiment? It inflates the type-I error (peeking problem); use sequential testing or alpha spending.

### Q12. If two models have the same AUC, are they equally good?
**Answer**: Not necessarily. AUC is a global summary of ranking: one model may be better at the head (high-precision region) and another at the tail, yet the integral is the same; when the business only cares about top-K, the difference can be large. AUC also does not reflect calibration, so the probability mass can be completely different. Combine PR curves, Recall@K, calibration curves, and stratified metrics.
**Follow-up**: How do you compare models for top-K scenarios? Look at Precision@K, Recall@K, NDCG@K, or business-revenue curves.

## 5. Pitfalls and follow-ups

- **Thresholding before computing AUC/AP**: this discards ranking information and degrades AUC into an accuracy variant.
- **Choosing the threshold or tuning on the test set**: thresholds and hyperparameters should be chosen on validation data; report the test set only once.
- **Mixing up PR-AUC and ROC-AUC**: under imbalance the two can lead to opposite conclusions; state the metric definition when reporting.
- **Omitting the averaging mode for multi-class metrics**: macro/micro can differ substantially.
- **Thinking a negative R² means a bug**: it is negative whenever the model is worse than the mean baseline.
- **Using MAPE on targets near 0**: the denominator explodes; switch to WAPE/sMAPE.
- **Watching only the core online metric**: ignoring latency, complaints, and revenue guardrails can lead to disaster.

## 6. Cheat sheet

| Metric | Formula / conclusion |
| --- | --- |
| Precision | $TP/(TP+FP)$ |
| Recall/TPR | $TP/(TP+FN)$ |
| FPR | $FP/(FP+TN)$ |
| F1 | $2PR/(P+R)$ |
| AUC | $P(s^+>s^-)$, sorting $O(N\log N)$ |
| AP | $\sum(R_n-R_{n-1})P_n$ |
| RMSE/MAE | Root mean squared error / mean absolute error |
| R² | $1-SS_{res}/SS_{tot}$, can be negative |
| NDCG | $DCG/IDCG$, position discount plus graded relevance |
| CTR/CVR | Clicks/impressions, conversions/clicks |

**Mnemonic**: *balanced data, look at AUC; sparse positives, look at PR; ranking, look at NDCG; probabilities, look at calibration; regression starts with MAE, and avoid MAPE when outliers abound.*

## 7. References

- [scikit-learn · Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [MLE Interview · Evaluation metrics](https://longxingtan.gitbook.io/mle-interview)
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
- [NDCG ranking evaluation metrics](https://zhuanlan.zhihu.com/p/448686098)
