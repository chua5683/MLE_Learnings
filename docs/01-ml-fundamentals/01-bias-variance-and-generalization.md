# Bias, variance, and generalization

> **One-line summary**: Generalization error decomposes into bias² + variance + irreducible noise; model capacity, regularization, and data size jointly determine the balance among the three.
>
> **Keywords**: `bias-variance decomposition` overfitting underfitting cross-validation learning curves VC dimension

## 1. Core concepts

- **Empirical risk**: the average loss on the training set, the objective the optimizer actually minimizes.
- **Expected risk / generalization error**: the average loss over the true data distribution. It is what we care about but cannot compute directly.
- **Model capacity**: how rich the functions in the hypothesis space are. Higher capacity makes it easier to fit the training set, and easier to memorize noise.
- **Overfitting**: training error is low while validation error is clearly higher; the model has learned noise specific to the training set.
- **Underfitting**: both training and validation errors are high; the model fails to capture even the regularities in the training data.
- **IID assumption**: training and test samples are independent and identically distributed. Almost all generalization guarantees rest on this assumption.

An intuitive framework: **learning = balancing model capacity against data size**. If capacity is too large relative to the data, you overfit; if it is too small, you underfit. Regularization is an artificial reduction of capacity.

## 2. Principles and derivations

### 2.1 Bias-variance decomposition

Assume the data-generating process is

$$y = f(x) + \varepsilon, \qquad \mathbb{E}[\varepsilon]=0, \qquad \operatorname{Var}(\varepsilon)=\sigma^2$$

The training set $D$ is sampled independently from the data distribution, and the learner outputs $f_D(x)$. At a fixed test point $x$, take the expectation over the randomness of the training set and the noise:

$$
\begin{aligned}
\mathbb{E}_{D,\varepsilon}\left[(y - f_D(x))^2\right]
&= \mathbb{E}\left[(f(x) + \varepsilon - f_D(x))^2\right] \\
&= \mathbb{E}_D\left[(f(x) - f_D(x))^2\right] + \sigma^2
\end{aligned}
$$

The cross term $2\mathbb{E}[\varepsilon(f-f_D)]$ vanishes because $\varepsilon$ is independent of $D$ and has zero mean. Now expand $f_D$ around its mean $\bar f(x)=\mathbb{E}_D[f_D(x)]$:

$$
\mathbb{E}_D[(f - f_D)^2]
= (f - \bar f)^2 + \mathbb{E}_D[(f_D - \bar f)^2]
$$

Combining the two gives the classic decomposition:

$$
\underbrace{\mathbb{E}[(y-f_D(x))^2]}_{\text{expected test error}}
= \underbrace{(\bar f(x)-f(x))^2}_{\text{bias}^2}
+ \underbrace{\mathbb{E}_D[(f_D(x)-\bar f(x))^2]}_{\text{variance}}
+ \underbrace{\sigma^2}_{\text{irreducible noise}}
$$

- **Bias**: the gap between the average prediction of the model family and the true function; it measures whether the hypothesis space can represent the true function.
- **Variance**: the fluctuation of predictions across different training sets; it measures how sensitive the model is to sample perturbation.
- **Noise**: randomness inherent in the data. No model can remove it; it is the lower bound on error.

Note that this is a **pointwise** decomposition; in practice we take another expectation over the distribution of $x$. The true function $f$ is unknown, so the decomposition is mainly a tool for explanation and diagnosis, not a quantity we can compute directly.

### 2.2 Why high variance shows up as overfitting

High variance means that swapping in a different training set changes the model output substantially: the decision boundary wiggles around individual samples and fits the noise as well. A typical parameter-estimation example is the sample mean: with $m$ samples, its variance is $\sigma^2/m$; **more data directly reduces variance** without systematically increasing bias.

As model complexity grows, the hypothesis space expands: bias usually falls and variance usually rises. Total error is U-shaped, with the optimum between underfitting and overfitting. This is the **bias-variance trade-off**.

### 2.3 Cross-validation

$k$-fold cross-validation splits the data into $k$ parts, using one part for validation and the other $k-1$ parts for training in turn:

$$
\widehat{R}_{CV} = \frac{1}{k}\sum_{i=1}^{k} \frac{1}{|D_i|}\sum_{(x,y)\in D_i} L(y, f_{-i}(x))
$$

where $f_{-i}$ is the model trained without fold $i$.

- $k=5$ or $10$ is an empirical compromise: larger $k$ brings the training set closer to the full data and reduces bias, but costs about $k$ times more and makes the fold models more similar, so the variance estimate becomes unstable.
- **Leave-one-out (LOOCV, $k=m$)** is nearly unbiased but expensive, and the folds are highly correlated, so its estimate variance is not necessarily small; classification tasks may also fail to preserve class proportions.
- **Stratified $k$-fold**: each fold matches the overall class proportions. Use it for classification and imbalanced data.
- **Grouped $k$-fold (GroupKFold)**: samples from the same user or patient must stay in the same fold, otherwise information leaks.
- **Time-series splitting**: train only on the past and validate on the future; never shuffle.
- **Nested cross-validation**: the inner loop selects hyperparameters and the outer loop evaluates final performance, avoiding the optimistic bias of using the same validation set for both tuning and reporting.

### 2.4 Learning curves and sample size

A learning curve plots training and validation error against training-set size:

- Both curves are high and close → high bias (underfitting); more data helps little, so add features or capacity, or reduce regularization.
- Training error is low, validation error is much higher, and the gap is large → high variance (overfitting); more data, more regularization, or a simpler model usually helps.
- Validation error is still falling as data grows → more data will continue to pay off.

The theoretical intuition comes from uniform convergence. For a finite hypothesis space $\mathcal{H}$, Hoeffding's inequality plus a union bound gives, with probability at least $1-\delta$, for all $h\in\mathcal{H}$:

$$
R(h) \le \widehat{R}(h) + \sqrt{\frac{\log|\mathcal{H}| + \log(2/\delta)}{2m}}
$$

For an infinite hypothesis space, replace $\log|\mathcal{H}|$ with the VC dimension $d$; the generalization gap scales as

$$
R(h) \le \widehat{R}(h) + O\left(\sqrt{\frac{d\log(m/d) + \log(1/\delta)}{m}}\right)
$$

**Takeaway**: the generalization gap scales roughly as $\sqrt{d/m}$, so the sample size must be much larger than the effective capacity for the model to be reliable. PAC learning formalizes this statement: output, with high probability (probably), a hypothesis whose error is small enough (approximately correct).

### 2.5 Regularization, ensembles, and data size

Regularized empirical-risk minimization is

$$
\min_{h\in\mathcal{H}} \widehat{R}(h) + \lambda \Omega(h)
$$

The regularizer shrinks the effective hypothesis space and usually **increases bias while reducing variance**; choosing $\lambda$ means finding the optimum between the two. Data augmentation, dropout, early stopping, and bagging mainly work on the variance-reduction side (dropout can also be read as implicit model ensembling).

## 3. Implementation notes

```python
import numpy as np
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     learning_curve, validation_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1) Cross-validation: keep preprocessing inside the pipeline to avoid out-of-fold leakage
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
cv = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error")
print("CV RMSE:", np.sqrt(-scores).mean(), "+/-", np.sqrt(-scores).std())

# 2) Use a learning curve to decide between more data and more capacity
sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="neg_mean_squared_error", shuffle=True, random_state=42)
train_err = -train_scores.mean(axis=1)
val_err = -val_scores.mean(axis=1)

# 3) Use a validation curve to inspect regularization strength:
#    too large an alpha underfits, too small an alpha overfits
alphas = np.logspace(-3, 3, 13)
train_s, val_s = validation_curve(pipe, X, y, param_name="ridge__alpha",
                                  param_range=alphas, cv=cv,
                                  scoring="neg_mean_squared_error")
```

Rule of thumb: **training error reflects bias, while the train-validation gap reflects variance**. Plotting both curves separately is far more informative than staring at a single score.

## 4. Interview Q&A

### Q1. Derive the bias-variance decomposition.
**Answer**: Let $y=f(x)+\varepsilon$ with $\mathbb{E}\varepsilon=0$ and $\operatorname{Var}(\varepsilon)=\sigma^2$. Write the test error as $\mathbb{E}[(f+\varepsilon-f_D)^2]$; after expansion the noise cross term is zero, leaving $\mathbb{E}_D[(f-f_D)^2]+\sigma^2$. Add and subtract the mean $\bar f$ of $f_D$: $f-f_D=(f-\bar f)-(f_D-\bar f)$; the cross-term expectation is zero, so we get $(f-\bar f)^2+\mathbb{E}_D[(f_D-\bar f)^2]+\sigma^2$, i.e., bias² + variance + noise.
**Follow-up**: Why decompose squared loss specifically? Other losses (e.g., 0-1 loss or absolute loss) generally do not admit such a clean additive decomposition.

### Q2. How do you tell whether a model has high bias or high variance, and how do you fix each?
**Answer**: Compare training and validation error. If both are high and close → high bias/underfitting: add features, increase model capacity, reduce regularization, or train longer. If training error is low but validation error is much higher → high variance/overfitting: add data, add regularization, simplify the model, use early stopping or ensembling. If validation error keeps falling as the training set grows, more data is worthwhile.
**Follow-up**: Do not reuse the validation set to tune the model repeatedly, or you will overfit the validation set. Report final numbers on an independent test set or via nested CV.

### Q3. Why do people usually use 5-fold or 10-fold cross-validation?
**Answer**: Larger $k$ brings the training set closer to the full data and reduces bias, but costs about $k$ times more; the fold models are highly correlated, so beyond a point larger $k$ barely improves the variance of the estimate and makes the validation fold too small and noisy. Five folds are cheap; ten folds have lower bias. Both are practical sweet spots. Consider LOOCV only when data is very scarce.
**Follow-up**: With large datasets you can fall back to a single hold-out split with a large validation set, because the validation estimate itself is then stable enough.

### Q4. What are common pitfalls in cross-validation?
**Answer**: (1) Doing preprocessing or feature selection outside the folds, which leaks data; (2) not stratifying, so some fold has an unbalanced class ratio; (3) letting the same entity (user, patient, time) cross folds, which leaks information; (4) shuffling time series; (5) tuning repeatedly on CV scores and then reporting them as a generalization estimate; (6) ignoring group correlation. The fix is to put every data-dependent step inside a pipeline and use GroupKFold, TimeSeriesSplit, or nested CV.
**Follow-up**: How much does feature selection on the full data before CV overestimate performance? There is no fixed number, but with small samples and high dimensions the overestimate can be dramatic, sometimes selecting pure noise as "useful features".

### Q5. Does adding training data reduce both bias and variance?
**Answer**: Usually it significantly reduces variance only. Bias is determined by the hypothesis space and does not automatically fall as data grows. If the model cannot represent the true regularity (high bias), more data is nearly useless; change the model or add features. Data augmentation is different: it changes the training distribution and may introduce additional bias, but it can effectively reduce variance.
**Follow-up**: For nonparametric methods such as $k$-nearest neighbors, both bias and variance change as data grows (with fixed $k$, bias depends on data density), so parametric intuition does not carry over mechanically.

### Q6. Why does bagging mainly reduce variance while boosting mainly reduces bias?
**Answer**: Bagging averages many **approximately independent** models. If each model has variance $\sigma^2$ and pairwise correlation $\rho$, the average has variance $\rho\sigma^2 + (1-\rho)\sigma^2/B$, which mainly shrinks the variance term while leaving bias roughly unchanged. Boosting fits residuals or negative gradients sequentially and keeps correcting the current model's systematic errors, so it mainly reduces bias; if trees are too deep or there are too many rounds, it can overfit (variance rises).
**Follow-up**: Random forests further lower the correlation $\rho$ between trees via feature subsampling, which makes them more stable than plain bagging.

### Q7. How does regularization change bias and variance?
**Answer**: A regularizer constrains parameter magnitude or the hypothesis space, making the model more rigid and less sensitive to training-set perturbation → lower variance. At the same time it may suppress genuine signal → higher bias. As $\lambda$ grows from zero, the model moves from overfitting toward underfitting; the optimal $\lambda$ minimizes validation error. L2 shrinks parameters smoothly, while L1 produces sparse solutions; both push bias and variance in the same directions but have different geometry.
**Follow-up**: Why is early stopping also a form of regularization? It restricts the effective search region to parameters near initialization, similar to L2, without an explicit penalty term.

### Q8. What is the VC dimension, and how does it relate to overfitting?
**Answer**: The VC dimension is the largest number of samples the hypothesis space can shatter; it measures expressive power. A finite VC dimension is a sufficient condition for learnability; the generalization gap grows roughly as $\sqrt{d/m}$, so a larger $d$ and fewer samples make overfitting more likely. Neural networks have huge parameter counts, but VC-dimension analysis is often overly pessimistic; in practice we use finer capacity measures such as margin and norm.
**Follow-up**: Is the VC dimension the same as the number of parameters? No. For example, linear classifiers in $\mathbb{R}^2$ have VC dimension 3, while they also have 3 parameters (including the bias); that coincidence is exactly a coincidence.

### Q9. How do you use learning curves to decide between "more data" and "more model"?
**Answer**: Plot error against training-set size. If validation error is still clearly falling at the high-data end → more data pays off. If both training and validation errors are high and close → high bias; add capacity or features. If training error is low and validation error is high with a gap that does not shrink as data grows → high variance; add regularization, simplify the model, or ensemble. You can also plot validation error against model complexity to find the bottom of the U.
**Follow-up**: Learning curves are noisy, especially at small data sizes; repeat the split several times and average.

### Q10. Does the bias-variance decomposition still hold when training and test distributions differ?
**Answer**: The decomposition itself only assumes a fixed $x$ and expectations over $D$ and the noise, so it still holds. But the conclusion that "validation error ≈ generalization error" relies on IID; distribution shift (covariate shift, label shift, concept drift) introduces an additional shift error that ordinary CV cannot estimate. In that case, split validation by time or domain, use importance weighting, or apply domain adaptation.
**Follow-up**: Under covariate shift, importance weighting $w(x)=p_{test}(x)/p_{train}(x)$ corrects the expected risk, but the weights can have very high variance.

## 5. Pitfalls and follow-ups

- **Treating low training error as good generalization**: training error can go to zero; generalization is the only thing that matters.
- **Confusing variance with parameter variance**: it is the variance of model predictions across training sets, not the estimation variance of a single parameter.
- **Treating the $k$-fold CV score as a confidence interval**: fold models are correlated, so the standard deviation underestimates uncertainty; report the distribution over repeated CV runs.
- **Assuming LOOCV is always better**: it has high variance, cannot stratify, and is expensive; small-sample classification often uses stratified 5×2 CV.
- **Doing any label-dependent step outside the folds**: scaling, imputation, feature selection, target encoding, and hyperparameter search.
- **Splitting time-series or recommendation data randomly**: random splits leak future information into training.
- **Believing data size is a silver bullet**: high-bias models gain little from more data; diagnose first.

## 6. Cheat sheet

| Symptom | Training error | Validation error | Bias | Variance | First response |
| --- | --- | --- | --- | --- | --- |
| Underfitting | High | High | High | Low | Add features/capacity, reduce regularization, train longer |
| Overfitting | Low | High | Low | High | Add data, add regularization, early stopping, ensemble, simplify |
| Insufficient data | Low | High | Low | High | Data augmentation, transfer learning, strong regularization |
| Distribution shift | Normal | High | -- | -- | Split by time/domain, importance weighting |

**Core formulas**

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \sigma^2
\qquad\qquad
\text{Bagging variance} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

**Mnemonic**: *training error shows bias, the gap shows variance; capacity trades bias, data trades variance; regularization suppresses variance, ensembles stabilize outputs.*

## 7. References

- [The Elements of Statistical Learning, Ch. 7](https://hastie.su.domains/ElemStatLearn/)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [MLE Interview · ML theory](https://longxingtan.gitbook.io/mle-interview)
- [Understanding the Bias-Variance Decomposition](http://scott.fortmann-roe.com/docs/BiasVariance.html)
- [scikit-learn · Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
