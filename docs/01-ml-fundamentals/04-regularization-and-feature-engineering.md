# Regularization and feature engineering

> **One-line summary**: Regularization controls model complexity with priors or penalties (L1 for sparsity, L2 for smoothness), while feature engineering turns raw data into representations that are easy for a model to learn; both must be done without data leakage.
>
> **Keywords**: `L1/L2` ElasticNet Bayesian priors feature scaling target encoding data leakage

## 1. Core concepts

- **Regularization**: adding a penalty on model complexity to the empirical risk, equivalent to placing a prior on parameters and trading a little bias for lower variance.
- **Feature engineering**: scaling, encoding, combining, selecting, and aggregating over time to inject domain knowledge into the model. It pays off most for linear and distance-based models; for tree models it mainly means encoding and missing-value handling.
- **Data leakage**: using information at training time that is unavailable at prediction time (especially labels or future information), which inflates offline metrics and then fails in production.

Regularization and feature engineering share one goal: **make the model stable on unseen data**. The former constrains the hypothesis space; the latter improves the data representation.

## 2. Principles and derivations

### 2.1 L1, L2, and ElasticNet

The regularized empirical risk is

$$
\min_w \frac{1}{m}\sum_{i=1}^{m}L(y_i,f(x_i;w))+\lambda\Omega(w)
$$

- **L2 (Ridge)**: $\Omega(w)=\frac12\|w\|_2^2$; the gradient gains an extra $\lambda w$, so parameters shrink proportionally. Linear regression has the closed-form solution $w^*=(X^\top X+\lambda I)^{-1}X^\top y$.
- **L1 (Lasso)**: $\Omega(w)=\|w\|_1$; it is non-differentiable at 0, so subgradients are used. It is equivalent to applying a soft-threshold to each coordinate, $S_\lambda(z)=\mathrm{sign}(z)\max(|z|-\lambda,0)$, which can push some weights to **exactly 0**.
- **ElasticNet**: $\Omega=\lambda_1\|w\|_1+\lambda_2\|w\|_2^2$, balancing sparsity and grouping of correlated features; it is more stable than pure Lasso when there are many features and few samples.

**Why is L1 sparse?** Three perspectives:
1. **Geometry**: the L1 ball is a diamond, and the loss contour first touches it at a vertex, where some coordinates are exactly 0; the L2 ball is round, so the touch point is generally not on an axis.
2. **Bayesian**: L2 corresponds to a Gaussian prior $w\sim\mathcal{N}(0,\tau^2)$, while L1 corresponds to a Laplace prior $p(w)\propto e^{-|w|/b}$, which has a spike at 0, so the MAP solution favors sparsity.
3. **Optimization**: the subgradient interval at 0 is $[-\lambda,\lambda]$, which contains 0; as long as the feature's correlation with the residual is not large enough, the optimum stays at 0.

| Aspect | L1 / Lasso | L2 / Ridge |
| --- | --- | --- |
| Penalty | $\sum\vert w_j\vert $ | $\sum w_j^2$ |
| Solution | Sparse (some exactly 0) | Dense, globally shrunk |
| Prior | Laplace | Gaussian |
| Correlated features | Picks one at random | Keeps them together, shares weight |
| Optimization | Subgradient/coordinate descent/proximal | Closed form/gradient descent |
| Use case | High-dimensional sparse, feature selection | Correlated features, stable prediction only |

### 2.2 Other regularization techniques (classical ML view)

- **Early stopping**: stop when validation error starts to rise. For quadratic loss, it approximates constraining parameters near their initialization, similar to L2.
- **Data augmentation**: expands the training distribution and reduces variance, but may introduce bias.
- **Label smoothing**: turns a hard label $y$ into $(1-\epsilon)y+\epsilon/K$, preventing overconfidence and improving calibration.
- **Dropout**: randomly zeros neurons during training; it can be viewed as an implicit ensemble of many subnetworks and approximates adaptive L2 (covered in the deep-learning chapter).
- **Weight decay vs L2**: under plain SGD the two are equivalent; under Adam, L2 enters the gradient and gets adaptively rescaled, while AdamW decouples it into true weight decay.

### 2.3 Feature scaling

| Method | Formula | Characteristics |
| --- | --- | --- |
| Standardization | $z=(x-\mu)/\sigma$ | Zero mean, unit variance; most common |
| Min-Max | $(x-x_{min})/(x_{max}-x_{min})$ | Maps to [0,1]; sensitive to outliers |
| Robust | $(x-\text{median})/\text{IQR}$ | Resistant to outliers |
| Log / quantile transform | $\log(1+x)$, rank-gauss | Handles long tails and skew |

**Which models need scaling**: KNN, KMeans, SVM (especially with an RBF kernel), PCA, regularized linear models, and neural networks, because they rely on distances, inner products, or gradient scale. **Tree models do not need it**, because splits only compare order. Scaling parameters must be fit on the training set only and then applied to validation/test data.

### 2.4 Categorical feature encoding

- **One-hot**: simple and effective for few categories; with high cardinality it explodes in dimension, becomes sparse, and is inefficient for tree splits.
- **Ordinal/Label**: only suitable for ordered categories; forcing an order onto unordered categories introduces a spurious magnitude relationship (linear models suffer, trees tolerate it partially).
- **Target/mean encoding**: replaces a category with the mean label for that category. It is expressive but leaks easily. A common smoothed form is

$$
\tilde y_c=\frac{n_c\bar y_c + m\bar y_{global}}{n_c+m}
$$

where $n_c$ is the number of samples in the category and $m$ is the smoothing strength. It must be computed **out-of-fold**, and the test set uses statistics from the full training set.
- **Count/frequency encoding**: uses category frequency; simple and leak-free, but may lose information.
- **Hashing**: hashes categories into a fixed dimension, suitable for very high cardinality or online settings, at the cost of collisions.
- **Embedding**: lets a deep model learn a low-dimensional representation (see Chapter 2).

### 2.5 Feature selection

| Type | Methods | Pros | Cons |
| --- | --- | --- | --- |
| Filter | Variance, chi-square, mutual information, correlation | Fast, model-agnostic | Ignores feature interactions |
| Wrapper | RFE, forward/backward search | Directly optimizes model metric | Expensive, easy to overfit validation |
| Embedded | L1, tree importance, regularization | Done during training | Depends on the specific model |

**Iron rule**: feature selection is a label-dependent step, so it must be done **inside every cross-validation fold**. Selecting features on the full data before CV systematically overestimates performance.

### 2.6 Data leakage: the most common source of accidents

Typical scenarios:
1. **Preprocessing leakage**: standardizing, imputing, or selecting features on the full data before splitting into train and test.
2. **Target-encoding leakage**: encoding a sample with a mean that includes its own label.
3. **Temporal leakage**: randomly splitting time series, or using features that are only available in the future (e.g., "clicks in the next 7 days").
4. **Group leakage**: samples from the same user, device, or patient appear in both training and validation.
5. **Duplicate samples**: the same row appears multiple times, so train and validation "collide" after a random split.
6. **Label-leaking features**: a feature that is actually a post-hoc record of the label (e.g., "already paid" predicting "will file a claim").

**Detection**: absurdly high CV scores, a single feature with abnormally high importance, a huge offline gain with no online change. **Prevention**: put all fitting steps inside a pipeline; split by time or group; validate point-in-time correctness of feature computation; run a time-based holdout before launch.

### 2.7 Time features and missing values

- **Time features**: year/month/day/hour, weekday, holiday flags; encode periodicity with $\sin(2\pi t/T)$ and $\cos(2\pi t/T)$; time deltas, rolling-window statistics (past 7/30 days), and trends. Rolling features must use only data available before the current timestamp.
- **Missing values**: missing-indicator variables (missingness itself may be a signal), mean/median/model-based imputation, or letting trees handle NaN directly. Note that under MNAR (missingness related to the label), imputation introduces bias.

## 3. Implementation notes

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

num_cols = ["age", "income", "click_7d"]
cat_cols = ["city", "device"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           min_frequency=20))]), cat_cols),
])
model = Pipeline([("pre", pre),
                  ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=1000))])
scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
```

Key points: ColumnTransformer ensures scaling/encoding are fit only on training folds; handle_unknown="ignore" prevents errors when new categories appear online; for high-cardinality categories, consider target encoding or hashing, always computed strictly out-of-fold.

## 4. Interview Q&A

### Q1. Why does L1 produce sparse solutions?
**Answer**: Geometrically, the L1 constraint is a diamond, and the loss contour usually touches it at a vertex, which means some coordinates are 0; in the Bayesian view, L1 corresponds to a Laplace prior with a spike at 0, so MAP favors 0; in optimization, the subgradient at 0 is an interval, so coordinate descent pushes the coordinate exactly to 0 as long as its correlation with the residual is not large enough. The round L2 constraint lacks this property and only shrinks weights globally.
**Follow-up**: Is the L1 solution unique? When features are highly correlated, Lasso may pick one arbitrarily; the solution is unstable, and ElasticNet is more stable.

### Q2. Why is L2 regularization equivalent to a Gaussian prior?
**Answer**: The negative log of $p(w)=\prod_j\mathcal{N}(0,\tau^2)$ is proportional to $\sum w_j^2/(2\tau^2)$. The MAP estimate is $\arg\min[-\log p(y|X,w)+\lambda\|w\|^2]$; setting $\lambda=\sigma^2/(2\tau^2)$ gives Ridge. L1 corresponds to a Laplace prior, so L1/L2 correspond to a "sparse prior" and a "smooth prior", respectively.
**Follow-up**: How do $\lambda$ and the prior variance relate? A smaller $\tau^2$ (stronger prior) corresponds to a larger $\lambda$.

### Q3. How do you choose between L1 and L2, and what problem does ElasticNet solve?
**Answer**: Use L1 when you need feature selection, high-dimensional sparsity, or interpretability; use L2 when features are correlated and you only need stable prediction; use ElasticNet when the number of features far exceeds the number of samples and features are correlated in groups, because it keeps L1's sparsity while letting correlated features enter and leave together, yielding a more stable solution. In practice, tune $\lambda_1,\lambda_2$ by cross-validation.
**Follow-up**: Why does Lasso select at most $n$ features when the feature count exceeds the sample count? That is the degrees-of-freedom bound of the Lasso solution; ElasticNet can exceed it.

### Q4. Why should features usually be standardized before regularization?
**Answer**: L1/L2 penalize all coefficients equally, but different feature scales mean that "the same coefficient size" represents different real effects. Without standardization, a large-scale feature needs only a tiny coefficient, so the penalty barely affects it, while a small-scale feature is forced to have a large coefficient and is over-penalized. After standardization, the penalty is comparable across features, and the choice of $\lambda$ becomes meaningful. Tree models are unaffected.
**Follow-up**: Should the intercept be penalized? Usually not, because it does not affect model complexity; sklearn does not penalize it by default either.

### Q5. Is a larger regularization coefficient always better? How do you choose it?
**Answer**: No. As $\lambda$ grows from 0, variance falls and bias rises, and validation error first falls and then rises, forming a U shape. Search over a log grid with cross-validation, or inspect a validation curve. Too small overfits; too large underfits (coefficients are all squeezed toward 0).
**Follow-up**: Why use a log grid instead of a linear grid? The effect of regularization strength is multiplicative and spans orders of magnitude, so a log grid is more efficient.

### Q6. How are early stopping, dropout, and L2 regularization related?
**Answer**: All three limit the model's effective capacity. Early stopping constrains parameters near their initialization, approximating L2 for quadratic loss; dropout randomly drops units during training, equivalent to ensembling many subnetworks and approximating adaptive L2; L2 explicitly penalizes parameter magnitude. They can be combined, but all should be used with a validation set to avoid "early stopping on the test set".
**Follow-up**: Why multiply by keep_prob at inference for dropout (or use inverted dropout)? To keep the expected output consistent between training and inference.

### Q7. How should high-cardinality categorical features be handled?
**Answer**: Prefer target encoding (out-of-fold plus smoothing) or count encoding; when there are very many categories, use hashing to control dimensionality, or let tree models handle categorical features directly (LightGBM/CatBoost); you can also merge rare categories by business logic (min_frequency). One-hot with thousands of categories explodes in dimension and becomes sparse, so it is usually a poor choice.
**Follow-up**: Why does target encoding overfit easily? With few samples per category the mean is noisy, and the encoding uses the label; smoothing and out-of-fold computation are the cure.

### Q8. How does target encoding avoid data leakage?
**Answer**: Use K-fold out-of-fold encoding: split the training set into K folds and compute each fold's encoding using only the labels from the other folds; the test set uses statistics from the full training set. Add smoothing (shrink toward the global mean) and a minimum-sample threshold. For temporal data, compute "past statistics for the future", never random folds.
**Follow-up**: Do you still need nested CV after out-of-fold encoding? Yes, when evaluating the final model you still need an outer CV, so the same validation set is not used both for encoding and evaluation.

### Q9. What are common data-leakage scenarios and how do you detect them?
**Answer**: Preprocessing/feature selection before splitting, target encoding with full labels, random time-series splits, samples from the same group across folds, duplicate samples, and post-hoc label features. Detection: unusually high CV, a single feature with excessive predictive power, a large offline/online gap, and high scores after shuffling labels. Prevention: pipelines, time/group splits, point-in-time validation, and a time-based holdout before launch.
**Follow-up**: Why is the shuffled-label test useful? If CV remains significantly above random after shuffling labels, there is leakage or duplication in the pipeline.

### Q10. Which models need feature scaling, and why do tree models not?
**Answer**: KNN, KMeans, SVM, PCA, regularized linear models, and neural networks need it; trees compare feature values only within a feature, and monotone transforms do not change split points, so they do not. Scaling must be fit on the training set, with validation/test only transformed, otherwise it leaks.
**Follow-up**: Do trees need no scale handling at all? Missing values, extreme values, and categorical encoding still need handling; scale simply does not affect splits.

### Q11. How do you choose a feature-selection method, and why must it be done inside cross-validation?
**Answer**: For many features and speed, use filters (variance/mutual information/chi-square); for model-guided selection where computation allows, use wrappers (RFE); for automatic selection during training, use embedded methods (L1/tree importance). Feature selection uses labels, so if it is done on the full data before CV, label information from the validation fold has already entered the selection process and metrics become optimistically high; the correct approach is to put the selector inside the pipeline and fit it on each training fold.
**Follow-up**: What is the core difference between filters and wrappers? Filters evaluate features independently of the model, while wrappers use model performance as the criterion.

### Q12. How should missing values be handled?
**Answer**: First determine the mechanism (MCAR/MAR/MNAR). Simple approaches: mean/median/mode imputation plus a missing indicator; fill categorical features with a "missing" category; trees can consume NaN directly (XGBoost/LightGBM learn a direction); or use model-based imputation or multiple imputation. Under MNAR, imputation introduces bias, so explicitly model the missingness mechanism or run a sensitivity analysis.
**Follow-up**: Why add a missing indicator? Missingness itself may carry information and lets the model distinguish "true 0" from "missing".

## 5. Pitfalls and follow-ups

- **Fitting preprocessors on the full data**: scaling, imputation, encoding, and feature selection must all be fit inside training folds.
- **Including the intercept in the regularizer**: the intercept does not control complexity and is usually not penalized.
- **Optimizing L1 directly with gradient descent**: it is non-differentiable at 0, so use subgradients/proximal methods/coordinate descent; sklearn's saga solver supports L1.
- **Replacing categories directly with target encoding**: must be out-of-fold plus smoothed, otherwise it leaks.
- **Feeding high-cardinality one-hot features to trees**: sparse and inefficient for splits; consider target encoding or native categorical support.
- **Using future information in time features**: rolling windows and cumulative statistics must be point-in-time correct.
- **Suspecting leakage when offline gains are huge**: especially when AUC suddenly jumps from 0.7 to 0.95.

## 6. Cheat sheet

| Topic | Key conclusion |
| --- | --- |
| L1 | Sparse, Laplace prior, soft threshold, can select features |
| L2 | Smooth shrinkage, Gaussian prior, closed form, handles collinearity |
| ElasticNet | L1+L2, groups correlated features, more stable in high dimensions |
| Before regularization | Standardize (except tree models) |
| Encoding | Low cardinality: one-hot; high cardinality: target encoding/hashing/native categorical |
| Target encoding | Out-of-fold + smoothing + minimum sample count |
| Feature selection | filter/wrapper/embedded, always inside CV |
| Leakage | Pipeline, time/group splits, point-in-time |
| Scaling | Needed by distance/gradient/regularized models, not by trees |

**Mnemonic**: *L1 selects features, L2 stabilizes coefficients; encoding must prevent leakage, selection stays inside folds; scaling depends on the model, trees only look at order.*

## 7. References

- [The Elements of Statistical Learning, Ch. 3](https://hastie.su.domains/ElemStatLearn/)
- [scikit-learn · Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn · Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [MLE Interview · Feature engineering](https://longxingtan.gitbook.io/mle-interview)
