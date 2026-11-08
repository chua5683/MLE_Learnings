# Sampling and class imbalance

> **One-line summary**: Class imbalance is fundamentally about asymmetric misclassification costs plus a prior shift; resampling only changes the training distribution, while cost sensitivity, threshold moving, and probability calibration directly align the model with business objectives.
>
> **Keywords**: `SMOTE` oversampling undersampling class weight Focal Loss threshold moving probability calibration

## 1. Core concepts

- **Class imbalance**: the positive-to-negative ratio is extreme (e.g., fraud at 1:1000). To minimize overall loss, the model leans toward the majority class, so minority-class recall is very low.
- **Three levels of intervention**:
  1. **Data level**: oversampling, undersampling, SMOTE, data augmentation;
  2. **Algorithm level**: class weights, cost-sensitive losses, Focal Loss;
  3. **Decision level**: threshold moving, probability calibration, business rules.
- **Prior shift**: resampling changes $p(y)$, which changes the meaning of the probabilities the model learns; this is the root of many "good offline, bad online" failures.

| Technique | Effect | Main side effect |
| --- | --- | --- |
| Random oversampling | Increases minority-class weight | Duplicate samples → overfitting, inflated probabilities |
| Random undersampling | Reduces the majority class | Loses information, higher variance |
| SMOTE | Synthesizes minority samples | May generate noisy or out-of-region samples |
| class weight | Weighted loss | Distorts probabilities; needs calibration |
| Threshold moving | Changes the decision boundary | Does not improve ranking, only the operating point |
| Calibration | Corrects probabilities | Needs an independent calibration set |

## 2. Principles and derivations

### 2.1 Resampling

**Random oversampling**: duplicate minority samples. This is equivalent to giving the minority class more weight in the loss, but duplicates let the model memorize them, increasing variance and inflating probabilities.

**Random undersampling**: discard majority samples. Training is fast but information is lost; if sampling is random by class, the expected AUC stays roughly the same, but probabilities and PR metrics change.

**SMOTE (Synthetic Minority Over-sampling Technique)**: for a minority sample $x_i$, randomly choose $x_{zi}$ among its $k$ minority neighbors and interpolate on the segment between them:

$$
x_{new}=x_i+\lambda(x_{zi}-x_i), \qquad \lambda\sim U(0,1)
$$

Variants:
- **Borderline-SMOTE**: synthesize only near the boundary to avoid piling up points in safe regions.
- **ADASYN**: adapt the number of synthetic samples per minority sample according to the proportion of majority neighbors.
- **SMOTE-NC**: handles data with categorical features.
- **SMOTE + Tomek/ENN**: synthesize first, then clean up noisy/overlapping samples.

Note: SMOTE interpolates in feature space, so be careful with high-dimensional sparse data, categorical features, and features on very different scales; interpolation can produce unrealistic samples (age 25 and 45 interpolating to 35 is fine, but interpolating category codes is meaningless).

### 2.2 Cost-sensitive learning

**Class weights**: multiply the minority-class loss by a larger weight. The balanced weights in sklearn are

$$
w_c=\frac{N}{K\cdot n_c}
$$

where $N$ is the total number of samples, $K$ is the number of classes, and $n_c$ is the number of samples in class $c$. Weighted logistic regression is equivalent to weighting the likelihood and changes the optimal decision boundary.

**Cost matrix and optimal threshold**: let the expected cost of predicting positive be $(1-p)C_{FP}$ and that of predicting negative be $pC_{FN}$, and make the former smaller:

$$
(1-p)C_{FP}<pC_{FN}
\quad\Longleftrightarrow\quad
p>\frac{C_{FP}}{C_{FP}+C_{FN}}
$$

So the optimal threshold is not 0.5; it is determined by the ratio of false-positive to false-negative costs. The larger $C_{FN}$ is, the lower the threshold and the higher the recall.

**Focal Loss**:

$$
\text{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t
$$

$p_t$ is the predicted probability of the true class; $(1-p_t)^\gamma$ down-weights easy samples, $\gamma$ is often 2, and $\alpha_t$ balances classes. It originated in detection, but the idea is general: **make the model focus on hard samples**. Focal Loss is not resampling and does not change the data distribution, but it does change probability calibration.

### 2.3 Threshold moving

Once the model outputs probabilities, the default 0.5 is just a habit. The threshold $t$ determines the operating point:

- If the business fears misses (disease screening, fraud recall) → lower the threshold: higher recall at the cost of precision.
- If the business fears false alarms (account bans, intrusive notifications) → raise the threshold: higher precision at the cost of recall.

Choose the threshold on the validation set according to the objective (Recall ≥ 0.9, maximum F1, minimum expected cost, top-K hits), and report on the test set only once.

### 2.4 Probability calibration

**Why calibration is needed**: resampling, class weights, and Focal Loss all push output probabilities away from true frequencies. Calibration maps scores to probabilities whose frequency interpretation is correct.

- **Reliability diagram**: bucket predicted probabilities and compare the average predicted probability in each bucket with the actual positive rate; ideally this lies on the diagonal.
- **ECE (expected calibration error)**: the weighted average of $|\text{acc}-\text{conf}|$ over buckets.
- **Brier score**: $\frac1m\sum(\hat p_i-y_i)^2$, penalizing both calibration and discrimination.
- **Platt scaling**: fit $p=\sigma(a\cdot s+b)$ on an independent calibration set; few parameters, works with little data.
- **Isotonic regression**: fit a monotone non-decreasing piecewise function; flexible but needs more calibration data and overfits on small data.

**Recovering probabilities after sampling**: if the positive class is randomly oversampled by a factor $r$, the learned probability $p'$ and the true probability $p$ satisfy

$$
p'=\frac{rp}{rp+(1-p)}
\quad\Longrightarrow\quad
p=\frac{p'}{r+(1-r)p'}
$$

Random undersampling can be corrected similarly using the prior ratio. A safer approach is to **recalibrate on the un-sampled validation set**.

### 2.5 Effect of sampling on AUC / PR

- **AUC**: depends only on the ranking of positive versus negative scores. Random oversampling/undersampling does not change the conditional distribution in expectation, so AUC stays roughly the same (with finite-sample fluctuations). A large AUC jump after sampling is usually not due to sampling itself but to changes in the model, threshold, or data split.
- **PR-AUC**: Precision depends on the positive prior; sampling changes the prior, shifting the whole PR curve and the AP baseline (the positive rate), so results are not directly comparable to the un-sampled case.
- **Conclusion**: under imbalance, report **PR-AUC, F1, Recall at a fixed Precision, and calibration curves**, not just AUC.

### 2.6 Recommended workflow

1. Split train/validation/test first, and resample **only inside the training folds**.
2. Try class weights or cost-sensitive losses first (simple and interpretable).
3. Choose the threshold from the PR curve and set the operating point by business cost.
4. Calibrate probabilities on an un-sampled validation set.
5. Monitor minority-class recall, calibration error, and data drift online.

## 3. Implementation notes

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import precision_recall_curve, average_precision_score

# Oversample only inside training folds: imblearn's Pipeline resamples the training data at fit time
pipe = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=42)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
])
pipe.fit(X_train, y_train)

# Choose the threshold from the PR curve: maximize F1 on the validation set
proba = pipe.predict_proba(X_valid)[:, 1]
prec, rec, thr = precision_recall_curve(y_valid, proba)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_t = thr[f1[:-1].argmax()]
print("AP:", average_precision_score(y_valid, proba), "best threshold:", best_t)

# Probability calibration: fit on an independent validation set, then apply to test
cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
cal.fit(X_valid, y_valid)
p_test = cal.predict_proba(X_test)[:, 1]
```

Note: FrozenEstimator requires sklearn 1.6+; older versions can fall back to cv="prefit". Resampling applies only to training, while validation/test keep the original distribution.

## 4. Interview Q&A

### Q1. Why is accuracy unreliable under class imbalance?
**Answer**: Accuracy treats positive and negative samples equally. When positives are 1%, predicting all negative gives 99% accuracy but zero recall, which is useless for the business. Use PR-AUC, F1, Recall at a fixed Precision, cost-weighted metrics, or Balanced Accuracy, and incorporate business costs.
**Follow-up**: What is Balanced Accuracy? The arithmetic mean of positive-class recall and negative-class recall (specificity); it is insensitive to class proportions.

### Q2. Why does random oversampling inflate predicted probabilities, and how do you recover?
**Answer**: Oversampling changes the positive prior $p(y=1)$, so the posterior the model learns corresponds to the sampled distribution. If the positive class is duplicated $r$ times, $p'=rp/(rp+1-p)$; invert to get $p=p'/(r+(1-r)p')$. More robustly, recalibrate with Platt/Isotonic on an un-sampled validation set.
**Follow-up**: Does undersampling bias probabilities too? Yes, in the opposite direction (the positive proportion is raised), and it also needs calibration.

### Q3. What is the principle behind SMOTE, and what are its drawbacks?
**Answer**: For each minority sample, find its $k$ minority neighbors and randomly interpolate on the connecting segments, $x_i+\lambda(x_{zi}-x_i)$, to generate new samples and enlarge the minority region. Drawbacks: interpolation may be unrealistic in high-dimensional/sparse data; categorical features need special handling; if the minority class contains noise, more noise is synthesized nearby; synthetic points may cross the decision boundary and cause overlap; and it helps only marginally under extreme imbalance.
**Follow-up**: What is the essential difference between SMOTE and random oversampling? Random oversampling copies existing samples (no new information), while SMOTE performs local interpolation in feature space (more diversity, but a risk of noise).

### Q4. Why can't validation and test sets be oversampled?
**Answer**: Validation/test sets must represent the real deployment distribution; oversampling artificially changes the positive/negative ratio and makes metrics (especially Precision, PR-AUC, and calibration) deviate from true online behavior. The correct approach is to sample only inside training folds, keep validation at the original distribution, and use it for threshold selection and calibration.
**Follow-up**: How do you guarantee this during cross-validation? Put the sampler inside an imblearn Pipeline, fit it on each training fold, and leave validation folds un-sampled.

### Q5. Does AUC change after sampling?
**Answer**: In theory, random oversampling/undersampling does not change the conditional score distribution, so the expected AUC stays roughly the same, with finite-sample fluctuations. If AUC changes noticeably, check whether the validation set was also sampled, whether sampling was non-random (by difficulty/time), and whether the model's ranking changed because of sampling. An unchanged AUC does not mean the model is better; business metrics (PR/F1/calibration) may change substantially.
**Follow-up**: When can sampling genuinely change AUC? Under non-random sampling (e.g., keeping only hard negatives), when sampling changes model capacity/regularization relative to the data, or when sampling introduces label leakage.

### Q6. Are class weights and oversampling equivalent?
**Answer**: For linear models such as logistic regression, weighting the minority class and weighted oversampling are equivalent in expected loss (both amplify the minority loss); but the implementations differ: class weights do not duplicate data, are cheaper, and do not increase overfitting risk, while oversampling actually increases the sample count and affects regularization and tree split statistics. For tree models they are not fully equivalent: weights affect split gains and leaf values, while sampling changes the sample distribution of candidate splits.
**Follow-up**: Which is better? Prefer class weights; combining with SMOTE may help when data are extremely scarce, but validate it.

### Q7. Why can Focal Loss handle imbalance?
**Answer**: $\text{FL}=-\alpha_t(1-p_t)^\gamma\log p_t$. When a sample is easy ($p_t\to1$), $(1-p_t)^\gamma\to0$ and its loss is greatly reduced, so the model focuses on hard samples; $\alpha_t$ then balances classes. It does not change the data distribution, but it changes gradients and probability calibration, so the threshold/calibration usually needs to be redone.
**Follow-up**: How does Focal Loss differ from class weights? Class weights are static per-class weights, while Focal Loss weights dynamically by sample difficulty.

### Q8. How should the threshold be chosen, and how does it relate to cost sensitivity?
**Answer**: Choose on the validation set according to the business objective: maximize F1, maximize Precision subject to a recall floor, or minimize expected cost $C_{FP}FP+C_{FN}FN$. Theoretically, if the positive probability is $p$, the optimal threshold is $C_{FP}/(C_{FP}+C_{FN})$. So threshold selection essentially makes the cost matrix explicit, while class weights change the cost implicitly during training.
**Follow-up**: Can threshold moving improve AUC? No, AUC is independent of the threshold; it only changes the operating point (the P/R combination).

### Q9. How do you choose between Platt scaling and isotonic regression?
**Answer**: Platt fits a sigmoid with two parameters; it is stable with little data and preserves monotonicity, but can only correct S-shaped bias. Isotonic fits a monotone piecewise function; it is more flexible and can correct any monotone bias, but needs more calibration data and overfits on small samples. In practice, use Platt with little data and isotonic with enough data, and validate with reliability diagrams/ECE.
**Follow-up**: Can the calibration set be shared with the validation set? No; the calibration set must be independent of training and tuning, otherwise the estimate is optimistic.

### Q10. What is your complete plan for extreme imbalance?
**Answer**: (1) Clarify the business cost and metric (recall-first or precision-first); (2) split the data and use class weights/cost-sensitive loss inside training folds, with SMOTE if necessary; (3) choose the threshold from the PR curve and set the operating point by cost; (4) calibrate probabilities on an un-sampled validation set; (5) report PR-AUC, F1, Recall@P, and calibration error; (6) monitor minority recall and drift online, and mine hard examples when needed.
**Follow-up**: Why try class weights before SMOTE? The change is small, interpretable, and does not introduce synthetic-sample risk; SMOTE is a supplement, not the default.

## 5. Pitfalls and follow-ups

- **Sampling before splitting**: validation gets synthetic/duplicated samples and metrics are inflated.
- **Comparing PR-AUC directly after sampling**: the prior and baseline both change; compare against a baseline from the same distribution.
- **Not calibrating after oversampling**: probabilities are systematically too high, distorting CTR/risk bidding.
- **Using SMOTE on high-dimensional sparse one-hot features**: interpolation creates meaningless features; prefer class weights.
- **Tuning only the threshold without checking ranking**: threshold moving cannot rescue a poorly ranked model.
- **Reporting imbalanced results with accuracy**: interviewers will immediately mark it down.

## 6. Cheat sheet

| Topic | Key conclusion |
| --- | --- |
| Oversampling | Duplicates minority samples; probabilities are inflated and need calibration |
| Undersampling | Loses majority information; expected AUC unchanged |
| SMOTE | $x_i+\lambda(x_{zi}-x_i)$; risk of noise |
| class weight | $w_c=N/(K n_c)$; weighting during training |
| Optimal threshold | $C_{FP}/(C_{FP}+C_{FN})$ |
| Focal Loss | $-\alpha_t(1-p_t)^\gamma\log p_t$; focuses on hard examples |
| Calibration | Platt (little data) / isotonic (more data) |
| Metrics | Under imbalance look at PR-AUC/F1/Recall@P, not just AUC |
| Workflow | Split first → sample training folds → choose threshold on validation → calibrate |

**Mnemonic**: *sampling changes the distribution, weights change the loss, thresholds change the decision, calibration changes the probability; never sample the validation set, never trust AUC alone.*

## 7. References

- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [scikit-learn · Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
