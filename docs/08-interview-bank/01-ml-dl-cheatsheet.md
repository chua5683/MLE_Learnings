# ML / DL cheat sheet

> **How to use**: skim this the day before an interview. It contains only conclusions, formulas, and comparisons; derivations live in the chapter notes. Entries with an arrow "→" point to the corresponding note.
>
> **Scope**: this page focuses on classical ML; deep learning lists only the most common conclusions. See [Chapter 2](../02-dl-fundamentals/README.md) for derivations and details.

## 1. Generalization and regularization

| Topic | Conclusion / formula |
| --- | --- |
| Bias-variance | $\text{Error}=\text{Bias}^2+\text{Variance}+\sigma^2$ |
| High bias | Both train/validation errors high and close → add features/capacity, reduce regularization |
| High variance | Low train error, high validation error → add data, add regularization, early stopping, ensemble |
| k-fold CV | k=5/10; stratified keeps class proportions; split time series by time |
| Nested CV | Inner loop tunes, outer loop evaluates, avoiding tuning bias |
| VC intuition | Generalization gap $O(\sqrt{d/m})$, with $m\gg d$ |
| L1 | Sparse, Laplace prior, soft threshold $\mathrm{sign}(z)\max(\vert z\vert -\lambda,0)$ |
| L2 | Smooth, Gaussian prior, $w^*=(X^\top X+\lambda I)^{-1}X^\top y$ |
| Prerequisites for regularization | Standardize features; usually do not penalize the bias; choose $\lambda$ by CV |

→ [Bias, variance, and generalization](../01-ml-fundamentals/01-bias-variance-and-generalization.md) · [Regularization and feature engineering](../01-ml-fundamentals/04-regularization-and-feature-engineering.md)

## 2. Linear models and SVM

| Topic | Conclusion / formula |
| --- | --- |
| Least squares | $\nabla=X^\top(Xw-y)=0\Rightarrow w^*=(X^\top X)^{-1}X^\top y$ |
| Gaussian noise | MLE is equivalent to least squares |
| Logistic regression | $p=\sigma(w^\top x)$, $J=-[y\log p+(1-y)\log(1-p)]$ |
| LR gradient | $\nabla_w J=X^\top(p-y)/m$, Hessian positive semidefinite → convex |
| Softmax | $p_k=e^{z_k}/\sum_j e^{z_j}$, gradient $(P-Y)/m$ |
| SVM primal | $\min\frac12\|w\|^2$ s.t. $y_i(w^\top x_i+b)\ge1$ |
| SVM dual | $\max\sum\alpha_i-\frac12\sum\alpha_i\alpha_j y_i y_j K_{ij}$, $0\le\alpha_i\le C$ |
| KKT | $\alpha_i[y_i f(x_i)-1]=0$ → only support vectors have $\alpha_i>0$ |
| Kernel trick | Replace inner products with $K(x_i,x_j)$, an implicit high-dimensional map |
| LR vs SVM | LR outputs probabilities and uses log loss; SVM uses margin/hinge, has a sparse solution, and supports kernels |
| Hyperparameters | Larger C means weaker regularization; larger gamma means more local (RBF) |

→ [Linear models and SVM](../01-ml-fundamentals/02-linear-models-and-svm.md)

## 3. Tree models and ensembling

| Topic | Conclusion / formula |
| --- | --- |
| Information gain | $H(D)-\sum_v\frac{\vert D^v\vert }{\vert D\vert }H(D^v)$, ID3 |
| Gain ratio | Information gain / intrinsic feature entropy, C4.5 |
| Gini | $1-\sum_k p_k^2$, CART classification |
| Regression tree | Variance reduction / MSE minimization, leaf outputs the mean |
| Pruning | Pre-pruning is fast but underfits easily; post-pruning works better but costs more; $R_\alpha(T)=R(T)+\alpha\vert T\vert $ |
| RF | Bagging + feature randomness; variance $\rho\sigma^2+(1-\rho)\sigma^2/B$ |
| OOB | About $1/e\approx36.8\%$ of samples are out of bag |
| GBDT | Additive model fitting the negative gradient $r_{im}=-\partial L/\partial F$ |
| XGB objective | $\sum_i[g_i f_t+\frac12 h_i f_t^2]+\gamma T+\frac12\lambda\|w\|^2$ |
| XGB leaf | $w_j^*=-G_j/(H_j+\lambda)$ |
| XGB gain | $\frac12[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma$ |
| Missing values | Try sending missing samples left and right; use the higher-gain direction as the default |
| LightGBM | Histogram 255 bins, leaf-wise, GOSS, EFB |
| leaf-wise | Lower error for the same number of leaves, but deeper and easier to overfit |
| GOSS | Keep large gradients and sample small gradients with weight $(1-a)/b$ |
| CatBoost | ordered boosting, ordered target statistics, oblivious trees |

→ [Tree models and ensembling](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

## 4. Evaluation metrics

| Topic | Conclusion / formula |
| --- | --- |
| Precision / Recall | $TP/(TP+FP)$ / $TP/(TP+FN)$ |
| F1 | $2PR/(P+R)$; $F_\beta$ adjusts the P/R weight |
| ROC-AUC | $P(s^+>s^-)$; rank computation $O(N\log N)$ |
| AUC limits | Ranking only, not calibration; high AUC can coexist with low Precision under imbalance |
| PR / AP | $\text{AP}=\sum(R_n-R_{n-1})P_n$; prefer under imbalance |
| RMSE / MAE | Sensitive to large errors / robust; MAPE explodes as $y\to0$ |
| R² | $1-SS_{res}/SS_{tot}$, can be negative |
| NDCG | $DCG/IDCG$, $DCG=\sum(2^{rel_i}-1)/\log_2(i+1)$ |
| MAP / MRR / HR | Binary relevance / first-hit rank / whether the top K contains a hit |
| GAUC | Compute AUC per user, then weight by impressions |
| Online | CTR=clicks/impressions, CVR=conversions/clicks; A/B looks at confidence intervals |
| Multi-class | macro (equal weight per class) / micro (global aggregate) / weighted |

→ [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)

## 5. Imbalance, sampling, and calibration

| Topic | Conclusion / formula |
| --- | --- |
| Oversampling | Duplicate the minority class; probabilities inflate and need calibration |
| Undersampling | Loses majority information; expected AUC unchanged |
| SMOTE | $x_i+\lambda(x_{zi}-x_i)$, $\lambda\sim U(0,1)$ |
| class weight | $w_c=N/(K n_c)$; equivalent to weighted loss |
| Optimal threshold | $C_{FP}/(C_{FP}+C_{FN})$ |
| Focal Loss | $-\alpha_t(1-p_t)^\gamma\log p_t$ |
| Calibration | Platt (little data) / isotonic (more data); reliability diagram, ECE, Brier |
| Recovery after sampling | $p=p'/(r+(1-r)p')$ |
| Iron rule | Sample only inside training folds; keep validation/test at the original distribution |

→ [Sampling and class imbalance](../01-ml-fundamentals/06-sampling-and-class-imbalance.md)

## 6. Probabilistic models and EM

| Topic | Conclusion / formula |
| --- | --- |
| Naive Bayes | $p(y\vert x)\propto p(y)\prod_j p(x_j\vert y)$ |
| Smoothing | $(N_{jvc}+\alpha)/(N_c+\alpha\vert V_j\vert )$ |
| MLE / MAP | Likelihood / likelihood + prior (= regularization) |
| EM decomposition | $\log p(X\vert \theta)=\text{ELBO}+KL$ |
| E / M | $q=p(Z\vert X,\theta_{old})$ / maximize $\sum_Z q\log p(X,Z\vert \theta)$ |
| Monotonicity | Likelihood never decreases per round; converges to a local optimum |
| GMM | $\gamma_{ik}\propto\pi_k\mathcal{N}(x_i\vert \mu_k,\Sigma_k)$; $\pi_k=N_k/m$ |
| GMM vs KMeans | Soft assignment + covariance vs hard assignment + isotropic |
| HMM forward | $\alpha_t(j)=B_j(x_t)\sum_i\alpha_{t-1}(i)A_{ij}$ |
| Viterbi | Replace the forward sum with max + backtracking |
| CRF | $p(y\vert x)=\exp(\sum\lambda_k f_k)/Z(x)$, discriminative |

→ [Probabilistic models and EM](../01-ml-fundamentals/07-probabilistic-models-and-em.md)

## 7. Deep-learning bridge (conclusions only)

| Topic | Conclusion |
| --- | --- |
| Backpropagation | Chain rule; $\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^\top$ |
| Vanishing gradients | sigmoid/tanh saturation; mitigate with ReLU + residuals + normalization |
| Initialization | Xavier preserves variance; He is tailored to ReLU |
| BatchNorm | Uses batch statistics in training and running mean/variance at inference |
| Adam | First/second moments + bias correction; AdamW decouples weight decay |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$, complexity $O(n^2d)$ |

→ [Deep Learning Fundamentals](../02-dl-fundamentals/README.md) · [Transformer basics](../02-dl-fundamentals/05-transformer-basics.md)

## 8. One-page mnemonic

- **Generalization**: training error shows bias, the gap shows variance; capacity trades bias, data trades variance.
- **Linear**: regression uses least squares, classification uses cross-entropy; SVM looks at margins, the dual yields kernels.
- **Trees**: trees look at splits, forests look at correlation; GBDT fits negative gradients, XGB adds second-order terms and regularization; LightGBM uses histograms, leaf-wise needs a depth guard.
- **Metrics**: balanced data, look at AUC; sparse positives, look at PR; ranking, look at NDCG; probabilities, look at calibration.
- **Imbalance**: sampling changes the distribution, weights change the loss, thresholds change the decision, calibration changes the probability.
- **EM**: E-step finds posteriors, M-step finds parameters; local optima only, initialization matters.

## 9. Related entry points

- [Chapter 1 · Machine Learning Fundamentals](../01-ml-fundamentals/README.md)
- [Chapter 2 · Deep Learning Fundamentals](../02-dl-fundamentals/README.md)
- [Chapter 8 · Interview Question Bank](../08-interview-bank/README.md)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
