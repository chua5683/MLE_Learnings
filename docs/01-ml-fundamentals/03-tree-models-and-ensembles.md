# Tree models and ensembling

> **One-line summary**: Decision trees fit data by recursive partitioning, random forests reduce variance through bagging, and GBDT/XGBoost/LightGBM reduce bias through boosting; the key to XGBoost is the second-order expansion of the objective, and the key to LightGBM is histograms plus leaf-wise growth.
>
> **Keywords**: `information gain` Gini random forest GBDT XGBoost second-order expansion LightGBM histogram

## 1. Core concepts

- **Decision tree**: recursively partitions the feature space with axis-aligned splits; each leaf outputs a constant (a class/probability for classification, a mean for regression).
- **Ensemble learning**: combines multiple weak learners. **Bagging** trains in parallel and averages results, mainly reducing variance; **Boosting** trains sequentially, with each round correcting the previous round's errors, mainly reducing bias.
- **Bias-variance view**: a single tree overfits easily (high variance); averaging a forest reduces variance. GBDT keeps increasing fitting capacity, so training error falls monotonically, but too many rounds let variance creep back up.

| Method | Training | Base-learner relationship | Main effect | Representative |
| --- | --- | --- | --- | --- |
| Bagging | Parallel | Independent | Reduce variance | Random forest |
| Boosting | Sequential | Depends on previous round | Reduce bias | GBDT/XGBoost/LightGBM |
| Stacking | Two levels | Meta-learner | Combine heterogeneous models | Common in competitions |

## 2. Principles and derivations

### 2.1 Split criteria for decision trees

Let the current node contain sample set $D$ with class proportions $p_k$.

**Entropy and information gain (ID3)**:

$$H(D)=-\sum_{k=1}^{K}p_k\log_2 p_k, \qquad g(D,A)=H(D)-\sum_{v}\frac{|D^v|}{|D|}H(D^v)$$

Information gain favors features with many distinct values; in the extreme, using an ID column can drive entropy to 0.

**Information gain ratio (C4.5)** penalizes by the feature's own entropy:

$$g_R(D,A)=\frac{g(D,A)}{H_A(D)}, \qquad H_A(D)=-\sum_{v}\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$$

**Gini index (CART)**:

$$\text{Gini}(D)=1-\sum_{k}p_k^2, \qquad \text{Gini}(D,A)=\sum_{v}\frac{|D^v|}{|D|}\text{Gini}(D^v)$$

For binary classification, $\text{Gini}=2p(1-p)$, which has a similar shape to entropy but avoids logarithms, so it is faster in engineering.

**Regression trees** choose splits by squared error or variance reduction:

$$\text{Var}(D)=\frac{1}{|D|}\sum_{i\in D}(y_i-\bar y_D)^2, \qquad \Delta=\text{Var}(D)-\sum_v\frac{|D^v|}{|D|}\text{Var}(D^v)$$

**Continuous features**: sort the feature, then consider midpoints between adjacent values as candidate thresholds; a single feature costs $O(m\log m)$, and the greedy search for the best split costs about $O(md\log m)$ per level.

**Pruning**: pre-pruning (max_depth, min_samples_split, min_impurity_decrease) trains fast but may underfit; post-pruning grows the full tree and then prunes bottom-up, which works better but costs more. CART uses cost-complexity pruning:

$$R_\alpha(T)=R(T)+\alpha|T|$$

Larger $\alpha$ means a simpler tree; choose $\alpha$ on a validation set.

### 2.2 Random forests

Bagging draws $B$ bootstrap samples with replacement, trains one tree on each, and votes or averages at prediction time. Random forests add another layer of randomness: **each split considers only a random subset of $m_{try}$ features** (often $\sqrt d$ for classification and $d/3$ for regression).

**Why it works**: if a single tree has variance $\sigma^2$ and the pairwise correlation between trees is $\rho$, averaging $B$ trees gives variance

$$\text{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

Bagging shrinks the second term, and feature randomness lowers $\rho$; the two act together.

**OOB (out-of-bag) estimate**: each sample has probability roughly $(1-1/m)^m\to e^{-1}\approx36.8\%$ of not being sampled by a given tree. These out-of-bag samples provide an almost free validation estimate of generalization error.

**Feature importance**:
- **MDI (mean decrease impurity)**: accumulates the impurity decrease attributed to a feature; computed on the training set, fast, but biased toward high-cardinality and continuous features.
- **Permutation importance**: shuffles a feature and measures the drop in validation performance; model-agnostic, but correlated features share importance.
- **SHAP**: additive attribution from cooperative game theory; more consistent, but more expensive.

### 2.3 GBDT: gradient-boosted decision trees

GBDT is an additive model trained with forward stagewise fitting:

$$F_m(x)=F_{m-1}(x)+\nu f_m(x)$$

where $f_m$ is a regression tree and $\nu\in(0,1]$ is the learning rate (shrinkage). Round $m$ fits the **negative gradient (pseudo-residual)**:

$$r_{im}=-\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

- Squared loss: $r_{im}=y_i-F_{m-1}(x_i)$, exactly the residual.
- Log loss ($p=\sigma(F)$): $r_{im}=y_i-p_i$.

Leaf values can be found by a line search that minimizes the loss; for log loss, the Newton step is $\gamma_j=\sum_{i\in I_j}(y_i-p_i)/\sum_{i\in I_j}p_i(1-p_i)$. This explains why GBDT is essentially **gradient descent in function space**: each tree is a negative-gradient direction and the learning rate is the step size.

### 2.4 XGBoost: second-order expansion and regularization

XGBoost adds an explicit tree-complexity regularizer to the objective:

$$\mathcal{L}^{(t)}=\sum_{i=1}^{m}l(y_i,\hat y_i^{(t-1)}+f_t(x_i))+\Omega(f_t), \qquad \Omega(f)=\gamma T+\frac{1}{2}\lambda\|w\|^2$$

where $T$ is the number of leaves and $w$ is the leaf-weight vector. Take a second-order Taylor expansion of the loss ($g_i=\partial_{\hat y}l$, $h_i=\partial^2_{\hat y}l$):

$$\mathcal{L}^{(t)}\approx\sum_{i=1}^{m}\left[g_i f_t(x_i)+\frac{1}{2}h_i f_t^2(x_i)\right]+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2+\text{const}$$

Group samples by leaf, $I_j=\{i\mid q(x_i)=j\}$; the objective becomes an independent quadratic in each leaf:

$$\mathcal{L}^{(t)}=\sum_{j=1}^{T}\left[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2\right]+\gamma T$$

Differentiating with respect to $w_j$ gives the optimal leaf weight:

$$w_j^*=-\frac{G_j}{H_j+\lambda}, \qquad G_j=\sum_{i\in I_j}g_i, \ H_j=\sum_{i\in I_j}h_i$$

Substituting back gives the optimal objective value $-\frac{1}{2}\sum_j\frac{G_j^2}{H_j+\lambda}+\gamma T$. This defines the **split gain**:

$$\text{Gain}=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$$

A split is made only if the gain is $>0$; $\gamma$ is the threshold for splitting, and $\lambda$ shrinks leaf weights. For logistic loss, $g_i=p_i-y_i$ and $h_i=p_i(1-p_i)$.

**Missing values**: at each split, try sending missing samples to the left and to the right, and use the direction with the larger gain as the default, recording it for inference. This uses missingness without imputation.

**Parallelism and engineering**:
- Trees must be built sequentially, but **within a single tree, split finding is parallelized across features**.
- Before training, each feature is pre-sorted and compressed into a block (column storage), reused across iterations to reduce sorting overhead; supports cache-aware access and out-of-core computation.
- For large data, use **approximate quantile sketch** candidates instead of exact enumeration.
- Regularization and anti-overfitting knobs: eta, max_depth, min_child_weight, subsample, colsample_bytree, lambda, alpha, gamma.

### 2.5 LightGBM: histograms, leaf-wise growth, GOSS, EFB

**Histogram algorithm**: discretize each continuous feature into at most 255 bins; training then accumulates gradient statistics per bin. Split finding changes from "scan samples" to "scan bins", and a parent's histogram can be obtained by subtracting the children's histograms (histogram subtraction), so often only the smaller child needs a histogram. Bins can be stored as uint8, saving memory and speeding things up substantially.

**Leaf-wise (best-first) growth**: at each step, split the leaf with the largest gain, instead of expanding level by level as in level-wise growth. For the same number of leaves, leaf-wise training error is lower, but trees become deeper and overfit more easily, so num_leaves, max_depth, and min_data_in_leaf are needed.

**GOSS (Gradient-based One-Side Sampling)**: samples with large gradients contribute more to the gain, so keep the top $a\%$ by absolute gradient; randomly sample $b\%$ from the rest and multiply their weights by $(1-a)/b$ to keep the distribution unbiased. This estimates gains from fewer samples.

**EFB (Exclusive Feature Bundling)**: in sparse data many features are mutually exclusive (not nonzero at the same time); bundle such features into one, reducing dimensionality. The conflict rate is controlled by max_conflict_rate.

**Categorical features**: LightGBM handles categorical features directly by sorting categories according to gradient statistics and finding the optimal grouping, rather than one-hot encoding.

### 2.6 CatBoost in brief

Two keywords for CatBoost: **ordered boosting** (train on "past" samples to compute residuals for the current sample, mitigating target leakage) and **ordered target statistics** (target encoding for categorical features computed incrementally in sample order, avoiding future labels). Its trees are **oblivious trees** (the same split at each level), which predict fast and resist overfitting; the default parameters are usually strong.

### 2.7 Bias-variance summary

Random forests average many weakly correlated trees, mainly reducing variance while keeping bias close to a single tree. GBDT sequentially corrects residuals, mainly reducing bias, but too many rounds, too large a learning rate, or too many leaves let variance creep back. XGBoost/LightGBM regularizers and early stopping re-suppress variance on top of boosting's bias advantage.

## 3. Implementation notes

```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost: early_stopping must be used with eval_set
clf = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    eval_metric="auc", early_stopping_rounds=100, tree_method="hist")
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# LightGBM: num_leaves is the main capacity parameter for leaf-wise growth
gbm = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, max_bin=255, boosting_type="gbdt")
gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
        eval_metric="auc", callbacks=[lgb.early_stopping(100)])
```

Suggested tuning order: first fix the learning rate and number of trees (with early stopping), then tune max_depth/num_leaves/min_child_weight, then subsample/colsample, and finally the regularizers lambda/alpha/gamma.

## 4. Interview Q&A

### Q1. What are the differences among ID3, C4.5, and CART?
**Answer**: ID3 uses information gain, handles only discrete features, and favors features with many values; C4.5 uses the gain ratio to correct that bias, supports continuous features and missing values, and can prune; CART is a binary tree that uses Gini for classification and squared error for regression, supports continuous features, and underlies sklearn and GBDT. All three greedily pick the currently best split and do not guarantee a global optimum.
**Follow-up**: Why does CART use Gini instead of entropy? Gini avoids logarithms and is faster to compute, while the split behavior of the two is very similar.

### Q2. How are decision trees pruned, and how do you choose between pre-pruning and post-pruning?
**Answer**: Pre-pruning limits depth, minimum samples, or minimum impurity decrease during growth; it trains fast but can underfit, because it may miss structures that only pay off after a split. Post-pruning grows the full tree and then, bottom-up, compares validation error before and after removing a subtree, keeping the version with better generalization; it usually works better but costs more. CART cost-complexity pruning uses $R_\alpha(T)=R(T)+\alpha|T|$ to generate a sequence of subtrees and selects $\alpha$ on validation data.
**Follow-up**: Why does a single tree overfit? Leaves with few samples output noisy means and have high variance; pruning reduces variance.

### Q3. Why is a random forest better than a single tree, and what is OOB?
**Answer**: Bagging averages many approximately independent, low-bias trees, reducing variance from $\sigma^2$ to $\rho\sigma^2+(1-\rho)\sigma^2/B$, mainly a variance reduction; feature randomness further lowers the correlation $\rho$. OOB uses the roughly 36.8% of unsampled samples for each tree to make out-of-bag predictions, whose aggregate approximates validation error without an extra held-out set.
**Follow-up**: Can OOB replace cross-validation? Not for time-series or grouped data, because OOB still assumes samples are exchangeable.

### Q4. How is feature importance computed, and what are the pitfalls?
**Answer**: MDI accumulates impurity decrease and is computed on the training set; it is fast but biased toward high-cardinality and continuous features. Permutation importance shuffles a feature and measures the drop in validation performance; it is model-agnostic, but correlated features share importance. SHAP gives more consistent additive attributions. Pitfalls: correlated features mask each other's importance, accuracy-based evaluation distorts results under class imbalance, and computing permutation importance on the training set underestimates importance.
**Follow-up**: How do you validate importance? Run permutation on an independent validation set and check whether removing the feature actually hurts performance.

### Q5. Why does GBDT fit the negative gradient?
**Answer**: Treat the model as a point in function space. The functional gradient of the loss $L(F)$ is $\partial L/\partial F$, and moving along the negative gradient updates the function as $F_m=F_{m-1}-\nu\cdot\partial L/\partial F$. Fitting a regression tree to each sample's negative gradient realizes gradient descent in function space; under squared loss, the negative gradient is exactly the residual. Leaf values are then determined by a line search or Newton step.
**Follow-up**: How does this differ from gradient descent in deep learning? Deep learning updates parameters; GBDT updates the function itself, adding one tree per round with the learning rate as a shrinkage step.

### Q6. Hand-derive the XGBoost objective, leaf weights, and split gain.
**Answer**: The objective is $\sum_i l(y_i,\hat y_i^{(t-1)}+f_t(x_i))+\gamma T+\frac12\lambda\sum_j w_j^2$. The second-order expansion is $\sum_i[g_i f_t(x_i)+\frac12 h_i f_t^2(x_i)]+\Omega(f_t)$. Grouping by leaf, each leaf is a quadratic; differentiating gives $w_j^*=-G_j/(H_j+\lambda)$ and an optimal value of $-\frac12\sum_j G_j^2/(H_j+\lambda)+\gamma T$. The split gain is $\frac12[G_L^2/(H_L+\lambda)+G_R^2/(H_R+\lambda)-(G_L+G_R)^2/(H_L+H_R+\lambda)]-\gamma$.
**Follow-up**: What is the role of $h_i$? It measures loss curvature (the second derivative); $H+\lambda$ acts as confidence, and leaf weights are shrunk when samples are few or curvature is small.

### Q7. What is the difference between XGBoost and GBDT?
**Answer**: GBDT is the framework, using first-order gradients plus a line search; XGBoost adds: (1) a second-order expansion of the objective, giving faster convergence and closed-form leaf weights; (2) explicit tree-complexity regularization ($\gamma T+\frac12\lambda\|w\|^2$); (3) support for custom losses given first- and second-order derivatives; (4) learned default directions for missing values; (5) engineering optimizations such as column-block pre-sorting, approximate quantiles, and parallel/distributed training; (6) built-in cross-validation and early stopping.
**Follow-up**: Why is second-order information useful? It is equivalent to Newton's method, models loss curvature, and gives a more accurate step, typically requiring fewer iterations than first-order gradient descent.

### Q8. How does XGBoost parallelize? Can trees be built in parallel?
**Answer**: Trees cannot be built in parallel, because tree $t$ depends on the predictions of the first $t-1$ trees. Parallelism happens **inside a single tree**: features are pre-sorted into column blocks, and gain computations for different features/candidate splits run on multiple threads; in distributed training, data or features are sharded and all-reduce aggregates gradient histograms. The approximate algorithm reduces the number of candidate points via quantiles.
**Follow-up**: Why is pre-sorting important? Exact greedy splitting would repeatedly sort and scan features; the block structure sorts once and reuses the result across iterations.

### Q9. Why is LightGBM's histogram fast, and what is the risk of leaf-wise growth?
**Answer**: Histograms discretize continuous features into at most 255 bins, so split finding accumulates gradients on bins, with cost proportional to the number of bins rather than samples. A parent histogram can be obtained by subtracting the children's, and only the smaller child needs to be built, saving time and memory. Leaf-wise growth splits the leaf with the largest gain, giving lower error for the same number of leaves, but the tree becomes deep and unbalanced and overfits easily on small data; constrain it with num_leaves, min_data_in_leaf, and max_depth.
**Follow-up**: Does the histogram lose accuracy? Yes, fewer bins means coarser split points; max_bin trades accuracy against speed.

### Q10. What problems do GOSS and EFB solve?
**Answer**: GOSS addresses **many samples**: it keeps samples with large gradients, randomly samples samples with small gradients, and reweights them, reducing the number of samples that participate in statistics each round while trying not to change the gain estimate. EFB addresses **sparse or mutually exclusive features**: it bundles features that are rarely nonzero at the same time into one feature, reducing the feature count and hence histogram construction cost. Both are key to LightGBM's speed on large data.
**Follow-up**: Why is the GOSS weight $(1-a)/b$? Because only a $b$ fraction of small-gradient samples is retained; scaling them up makes their distribution match the full data in expectation.

### Q11. How do you choose between LightGBM and XGBoost?
**Answer**: Prefer LightGBM for large data, many features, and training speed (histograms plus leaf-wise plus GOSS/EFB); XGBoost is very stable when data is medium-sized and you want reliable convergence and a mature ecosystem; with many categorical features, prefer LightGBM/CatBoost; for small data sensitive to overfitting, be careful with leaf-wise and reduce num_leaves or switch to gbdt. In the end, choose by validation metrics and inference cost.
**Follow-up**: Both support early stopping and missing values; engineering concerns also include deployment ecosystem, model size, and cross-platform support.

### Q12. Do tree models need feature normalization? Can they extrapolate?
**Answer**: Trees compare only the ordering of feature values, and monotone transforms do not change splits, so **normalization/standardization is unnecessary**; this is an advantage of trees on features with mixed scales. But trees are piecewise constant and **cannot extrapolate**: feature values larger than anything seen in training fall into the last leaf's constant output, especially for regression; in that case, transform the target or switch to a linear/deep model.
**Follow-up**: Then why process features at all? Missing values, extreme values, and categorical encoding still need handling; only scaling is unnecessary.

## 5. Pitfalls and follow-ups

- **Information gain favors high-cardinality features**: continuous features and ID-like features can game it; use the gain ratio or limit candidate splits.
- **OOB is not universal**: when samples are not independent (users/time), OOB leaks.
- **MDI feature importance is biased**: high-cardinality features naturally get more split opportunities; do not use it directly for business explanations.
- **Tune XGBoost's eta and n_estimators together**: a small learning rate needs more trees, and early stopping determines the actual number.
- **min_child_weight is the sum of $h_i$ in a leaf**, not the sample count; for classification it approximates the weight of positive samples.
- **LightGBM's num_leaves matters more than max_depth**: leaf-wise trees can become very deep, and max_depth is only a safety net.
- **GBDT is sensitive to outliers** under squared loss; use Huber/quantile loss or clean the data first.

## 6. Cheat sheet

| Topic | Key formula / conclusion |
| --- | --- |
| Information gain | $H(D)-\sum_v\frac{\vert D^v\vert }{\vert D\vert }H(D^v)$ |
| Gain ratio | Information gain / intrinsic feature entropy |
| Gini | $1-\sum_k p_k^2$, CART classification |
| Regression tree | Variance reduction / MSE minimization |
| RF variance | $\rho\sigma^2+(1-\rho)\sigma^2/B$ |
| OOB | About $1/e\approx36.8\%$ of samples out of bag |
| GBDT | Fit the negative gradient, $F_m=F_{m-1}+\nu f_m$ |
| XGB leaf | $w_j^*=-G_j/(H_j+\lambda)$ |
| XGB gain | $\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$ |
| LightGBM | Histogram 255 bins, leaf-wise, GOSS, EFB |

**Mnemonic**: *trees look at splits, forests look at correlation; GBDT fits negative gradients, XGB adds second-order terms and regularization; LightGBM uses histograms, and leaf-wise growth needs a depth guard.*

## 7. References

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [MLE Interview · Tree models](https://longxingtan.gitbook.io/mle-interview)
