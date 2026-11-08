# Linear models and SVM

> **One-line summary**: Linear regression has a closed-form least-squares solution, logistic regression discriminates with cross-entropy, and SVM extends linear models to nonlinear boundaries through margin maximization plus the kernel trick.
>
> **Keywords**: `least squares` logistic regression cross-entropy margin maximization dual problem kernel trick Hinge loss

## 1. Core concepts

A linear model is written uniformly as

$$f(x) = w^\top x + b$$

Models differ only in **how the output is interpreted** and **how the loss is defined**:

| Model | Output | Loss | Optimization |
| --- | --- | --- | --- |
| Linear regression | Real value $\hat y\in\mathbb{R}$ | Squared loss | Closed form / gradient descent |
| Logistic regression | Probability $p\in(0,1)$ | Cross-entropy (negative log-likelihood) | Convex, no closed form |
| Softmax regression | $K$-class probabilities | Multi-class cross-entropy | Convex |
| SVM | Decision value / class | Hinge loss + L2 | Convex quadratic program (dual/SMO) |

Linear models are **discriminative**: they model $p(y|x)$ or the decision boundary directly, not $p(x)$. They are interpretable (coefficients give direction and weight), fast to train, and their extrapolation behavior is analyzable, which makes them the main arena for whiteboard derivations in interviews.

## 2. Principles and derivations

### 2.1 Linear regression and least squares

Fold the bias into the weight vector and let $X\in\mathbb{R}^{m×(d+1)}$ and $y\in\mathbb{R}^m$. The least-squares objective is

$$J(w) = \frac{1}{2}\|Xw - y\|_2^2$$

Taking the gradient with respect to $w$ and setting it to zero gives

$$\nabla_w J = X^\top(Xw - y) = 0 \quad\Longrightarrow\quad X^\top X w = X^\top y$$

If $X^\top X$ has full rank ($m\ge d+1$ and the features are not collinear), we obtain the **normal equation**:

$$w^* = (X^\top X)^{-1}X^\top y$$

The cost is about $O(md^2 + d^3)$: usable when there are many samples and the feature dimension is moderate; when $d$ is large, gradient descent or the dual form is more common. If collinearity makes $X^\top X$ singular, the solution is not unique; use the pseudoinverse or add L2 regularization (Ridge):

$$J(w) = \|Xw-y\|_2^2 + \lambda\|w\|_2^2 \quad\Longrightarrow\quad w^* = (X^\top X + \lambda I)^{-1}X^\top y$$

Because $X^\top X$ is positive semidefinite, adding $\lambda I$ with $\lambda>0$ makes every eigenvalue at least $\lambda$, so the matrix is invertible.

**Probabilistic interpretation**: if $y = w^\top x + \varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2)$, the negative log-likelihood is

$$-\log p(y|X,w) = \frac{1}{2\sigma^2}\|Xw-y\|^2 + \text{const}$$

So **MLE under Gaussian noise is equivalent to least squares**. This also explains why squared loss is sensitive to outliers: the Gaussian tail is light, and large residuals get squared.

### 2.2 Logistic regression

Sigmoid maps the linear output to a probability:

$$p(y=1|x) = \sigma(w^\top x + b), \qquad \sigma(z)=\frac{1}{1+e^{-z}}$$

Maximum likelihood is equivalent to minimizing the negative log-likelihood (cross-entropy):

$$J(w) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]$$

Differentiate with respect to $z_i = w^\top x_i + b$ and use $\sigma'(z)=\sigma(z)(1-\sigma(z))$:

$$\frac{\partial J}{\partial z_i} = p_i - y_i \quad\Longrightarrow\quad \nabla_w J = \frac{1}{m}X^\top(p-y)$$

In vector form: $\frac{\partial J}{\partial w} = \frac{1}{m}\sum_i (p_i-y_i)x_i$. The second derivative (Hessian) is

$$H = \frac{1}{m}X^\top \operatorname{diag}(p_i(1-p_i)) X \succeq 0$$

Therefore the cross-entropy loss is convex in $w$ and has a unique global optimum (unless the data are degenerate). There is no closed form, so we use gradient descent, L-BFGS, or Newton's method/IRLS.

**Why not squared loss?** Placing sigmoid inside MSE makes the loss non-convex in $w$; when the prediction is badly wrong and $\sigma$ saturates, the gradient approaches zero and learning stalls. Cross-entropy has a larger gradient the more wrong the prediction is, which gives better optimization behavior.

**Softmax regression** is the natural extension to multi-class classification:

$$p(y=k|x) = \frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}, \qquad J = -\frac{1}{m}\sum_i \sum_k y_{ik}\log p_{ik}$$

The gradient is again $\frac{1}{m}X^\top(P-Y)$. Note that softmax is invariant to adding a constant to all $z$; the parameters are redundant, and L2 regularization is usually used to pin down the scale.

### 2.3 SVM: margin maximization

Start with the hard-margin, linearly separable case. The hyperplane is $w^\top x + b = 0$, and the geometric distance from a sample to it is

$$\frac{|w^\top x_i + b|}{\|w\|}$$

Because rescaling $w,b$ together does not change the hyperplane, we can require the closest samples to satisfy $|w^\top x_i+b|=1$; then the **geometric margin** is $1/\|w\|$. Maximizing the margin is equivalent to minimizing $\|w\|^2$:

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1, \quad i=1,\dots,m$$

**Why maximize the margin?** The larger the margin, the less sensitive the decision boundary is to sample perturbation or measurement error, and the better the generalization bound (which depends on the margin). Geometrically, we are finding the widest separating strip between the two class convex hulls.

**Dual derivation**. The Lagrangian is

$$L(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i\left[y_i(w^\top x_i+b)-1\right], \quad \alpha_i\ge 0$$

Setting $\partial L/\partial w=0$ and $\partial L/\partial b=0$ gives:

$$w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0$$

Substituting back gives the dual problem:

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j x_i^\top x_j \quad \text{s.t.}\quad \sum_i \alpha_i y_i = 0, \ \alpha_i\ge 0$$

The KKT complementary slackness condition is

$$\alpha_i\left[y_i(w^\top x_i+b)-1\right] = 0$$

So only **support vectors** (with $y_i f(x_i)=1$) have $\alpha_i>0$; all other $\alpha_i=0$. Prediction depends only on the support vectors:

$$f(x) = \sum_i \alpha_i y_i x_i^\top x + b$$

**Soft margin** allows a few violations by introducing slack variables $\xi_i\ge 0$:

$$\min \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.t.}\quad y_i(w^\top x_i+b) \ge 1-\xi_i, \ \xi_i\ge 0$$

The dual constraint becomes $0\le\alpha_i\le C$. Larger $C$ tolerates fewer misclassifications (higher variance, easier to overfit); smaller $C$ gives a wider margin (higher bias). It is equivalent to the Hinge-loss form:

$$\min_{w} \frac{1}{2}\|w\|^2 + C\sum_i \max\left(0, 1-y_i f(x_i)\right)$$

### 2.4 Kernel trick

In the dual, samples appear only through inner products $x_i^\top x_j$. If we map $x$ to a higher-dimensional space via $\phi(x)$, it suffices to compute a kernel function

$$K(x_i,x_j) = \phi(x_i)^\top\phi(x_j)$$

without explicitly constructing $\phi$. This is the **kernel trick**. Common kernels:

- Linear kernel: $K=x_i^\top x_j$, common for high-dimensional sparse text.
- Polynomial kernel: $K=(\gamma x_i^\top x_j + r)^d$.
- RBF/Gaussian kernel: $K=\exp(-\gamma\|x_i-x_j\|^2)$, which corresponds to infinite-dimensional features; $\gamma$ controls locality.
- Sigmoid kernel: resembles a two-layer network but does not always satisfy Mercer's condition.

Mercer's theorem guarantees that if the kernel matrix is positive semidefinite, there exists a feature map $\phi$ for which $K$ is an inner product. Thus SVM is linear in the original space and nonlinear in feature space, and its computation depends only on the kernel matrix ($O(m^2)$ storage).

## 3. Implementation notes

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Logistic regression: C is the inverse regularization strength; penalty supports l1/l2/elasticnet
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(C=1.0, penalty="l2", max_iter=1000))
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

# SVM: RBF kernel; gamma and C are the main hyperparameters
svm = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
svm.fit(X_train, y_train)
print("support vectors:", svm[-1].n_support_)

# Hand-written logistic-regression gradient: p = sigmoid(Xw), grad = X^T(p-y)/m
def logreg_grad(X, y, w):
    p = 1.0 / (1.0 + np.exp(-X @ w))
    return X.T @ (p - y) / len(y)
```

Practical notes: **linear models and SVM are both sensitive to feature scale**, so always standardize; search the regularization coefficient **C** and kernel parameter **gamma** on a log grid with cross-validation; for class imbalance, set class_weight="balanced" or tune the threshold.

## 4. Interview Q&A

### Q1. Derive the logistic-regression gradient and explain why cross-entropy is used.
**Answer**: The model is $p=\sigma(w^\top x)$ and the negative log-likelihood is $J=-[y\log p+(1-y)\log(1-p)]$. Let $z=w^\top x$; then $\partial J/\partial p = -y/p+(1-y)/(1-p)$ and $\partial p/\partial z=p(1-p)$, so multiplying gives $\partial J/\partial z=p-y$; multiplying by $\partial z/\partial w=x$ gives $\nabla_w J=(p-y)x$. Cross-entropy is the negative log-likelihood of a Bernoulli distribution: its gradient is clean, the loss is convex, and the gradient grows with how wrong the prediction is.
**Follow-up**: Adding L2 regularization adds a term $\lambda w$ to the gradient, equivalent to MAP with a Gaussian prior.

### Q2. Why doesn't logistic regression use squared loss?
**Answer**: Squared loss with a sigmoid is non-convex in $w$ and has local minima; moreover, when $p$ is near 0 or 1 and the label is the opposite, the saturated sigmoid makes the gradient close to zero and training stalls. Cross-entropy is convex, and the more severe the misclassification, the larger the gradient. Classification needs a likelihood over conditional probabilities, not a regression residual.
**Follow-up**: If you insist on MSE, remove the sigmoid for a linear regression and threshold the output, but then it is no longer a probabilistic model.

### Q3. What is the closed-form solution of linear regression, and when does it fail?
**Answer**: $w^*=(X^\top X)^{-1}X^\top y$, derived from $X^\top(Xw-y)=0$. It fails when $X^\top X$ is not invertible (collinear features, $d>m$); the solution is then not unique, so use the pseudoinverse, Ridge (adding $\lambda I$), or gradient descent. When $d$ is large, inverting an $O(d^3)$ matrix is also impractical.
**Follow-up**: Why does adding $\lambda I$ guarantee invertibility? $X^\top X$ is positive semidefinite with smallest eigenvalue $\ge 0$; adding $\lambda>0$ makes the smallest eigenvalue $\ge\lambda>0$.

### Q4. Why does SVM maximize the margin?
**Answer**: The margin is the distance from the decision boundary to the closest sample, $1/\|w\|$. A larger margin makes the boundary more robust to noise and finite-sample perturbation, and the theoretical generalization bound depends on the margin; geometrically, it finds the widest separating strip between the two class convex hulls. Maximizing the margin is equivalent to minimizing $\|w\|^2$, a convex quadratic program with a unique global optimum.
**Follow-up**: The functional margin can be made arbitrarily large by rescaling $w,b$, so it must be fixed at 1 before discussing the geometric margin.

### Q5. Derive the SVM dual problem.
**Answer**: Write the Lagrangian $L=\frac12\|w\|^2-\sum\alpha_i[y_i(w^\top x_i+b)-1]$. Differentiating with respect to $w$ gives $w=\sum\alpha_i y_i x_i$; differentiating with respect to $b$ gives $\sum\alpha_i y_i=0$. Substituting back into $L$ to eliminate $w,b$ yields the dual objective in $\alpha$ only: $\sum\alpha_i-\frac12\sum\alpha_i\alpha_j y_i y_j x_i^\top x_j$, subject to $\alpha_i\ge0,\sum\alpha_i y_i=0$. KKT complementary slackness implies either $\alpha_i=0$ or the sample lies exactly on the margin boundary.
**Follow-up**: The dual has simpler constraints, involves only inner products (hence kernels), and yields a sparse solution; the price is an $O(m^2)$ kernel matrix.

### Q6. Why does the kernel trick work, and do we need to know the map $\phi$ explicitly?
**Answer**: The dual objective uses only $x_i^\top x_j$. If there exists $K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)$, replacing the inner product with $K$ is equivalent to training a linear SVM in a high-dimensional space, while all computation stays in the original space and $\phi$ is never constructed. Mercer's theorem guarantees that a positive-semidefinite kernel corresponds to some inner-product space. The RBF kernel even corresponds to an infinite-dimensional feature space.
**Follow-up**: The kernel matrix is $m×m$; with millions of samples, memory and compute become prohibitive, which is one reason deep models are more common on large data.

### Q7. Why is the SVM solution sparse and determined only by support vectors?
**Answer**: KKT complementary slackness requires $\alpha_i[y_i f(x_i)-1]=0$. For correctly classified samples with margin greater than 1, $y_i f(x_i)>1$, so $\alpha_i=0$; only samples on the margin boundary or violating it have $\alpha_i>0$, i.e., the support vectors. Prediction $f(x)=\sum\alpha_i y_i K(x_i,x)+b$ also sums only over those samples.
**Follow-up**: Sparsity helps storage and inference but makes the model sensitive to noise and outliers, because an outlier can become a support vector.

### Q8. How do you choose between logistic regression and SVM?
**Answer**: Choose LR when you need probability outputs, online learning, multi-class support, or interpretable coefficients; choose SVM for a pure classification boundary, small-to-medium samples, or a nonlinear kernel boundary. LR uses log loss and has gradients everywhere; SVM uses hinge loss, has zero gradient outside the margin, and yields a sparse solution. Both are convex linear/kernel methods, and feature scaling matters for both. In industry, large-scale sparse features (CTR) favor LR/FM/deep models; SVM does not scale as well on big data because of kernel-matrix cost.
**Follow-up**: SVM can also output probabilities, but requires Platt scaling (fitting a sigmoid on a validation set), which adds a calibration layer.

### Q9. How does softmax regression differ from "multiple binary logistic regressions"?
**Answer**: Multiple binary classifiers are one-vs-rest: each class has an independent sigmoid, the probabilities need not sum to 1, and the classes do not compete; softmax normalizes a joint distribution, the probabilities sum to 1, and the classes are coupled through the denominator, which suits mutually exclusive multi-class problems. If labels are multi-select (multi-label), use independent sigmoids; for mutually exclusive single labels, use softmax.
**Follow-up**: Softmax has parameter redundancy (adding a constant to all logits changes nothing), while binary sigmoid does not; that is why softmax plus L2 uniquely determines the solution.

### Q10. How can a linear model fit nonlinear relationships?
**Answer**: Three approaches: (1) manual feature transforms (polynomials, interactions, binning, splines); (2) kernel methods (SVM kernels, kernel ridge); (3) tree models or neural networks that learn nonlinearity automatically. A linear model by itself can only express a linear decision boundary, but with feature engineering it can still be strong; the downside is that high-order interactions must be designed by hand, and extrapolation behavior is determined by the basis functions.
**Follow-up**: Polynomial features explode in dimension ($O(d^p)$), so they need regularization and feature selection.

### Q11. What should you watch for with LR/SVM under class imbalance?
**Answer**: For LR, set class_weight (equivalent to weighting the likelihood per class) or tune the threshold; for SVM, use a different $C$ per class or adjust class weights. Note that weighting changes the calibration of predicted probabilities, so recalibrate on a validation set. Do not evaluate with accuracy; use PR-AUC/F1/recall at a precision target. Resampling must be applied only inside training folds, while validation keeps the original distribution.
**Follow-up**: Why does random oversampling make LR probabilities too high? It changes the positive-class prior, so the learned $p(y=1|x)$ corresponds to the sampled distribution.

## 5. Pitfalls and follow-ups

- **Regularization usually does not penalize the bias $b$**: the intercept does not affect model complexity, and sklearn does not penalize the intercept by default.
- **C is the inverse of regularization strength**: larger C means weaker regularization and easier overfitting; do not confuse it with $\lambda$.
- **Feature scaling is mandatory**: SVM kernel distances, the LR regularizer, and gradient-descent convergence all depend on scale.
- **When SVM gamma is large**: each sample has a very local influence and the boundary overfits; a very small gamma is overly smooth and underfits.
- **A hard margin may not exist when data are linearly separable in principle**: real data almost always have noise, so use a soft margin; C controls tolerance.
- **Multi-class strategy**: LR natively uses softmax (multinomial) or OvR; SVM natively uses OvO (libsvm), though OvR is also possible.
- **Calibrate probability outputs**: LR softmax probabilities are relatively reliable, while SVM's decision_function is not a probability.

## 6. Cheat sheet

| Topic | Key formula / conclusion |
| --- | --- |
| Least squares | $w^*=(X^\top X)^{-1}X^\top y$ |
| Ridge | $w^*=(X^\top X+\lambda I)^{-1}X^\top y$ |
| Logistic regression | $p=\sigma(w^\top x)$, $\nabla J=X^\top(p-y)/m$ |
| Softmax | $p_k=e^{z_k}/\sum_j e^{z_j}$, gradient $(P-Y)$ |
| Hard-margin SVM | $\min\frac12\|w\|^2$ s.t. $y_i(w^\top x_i+b)\ge1$ |
| Dual | $\max\sum\alpha_i-\frac12\sum\alpha_i\alpha_j y_i y_j K_{ij}$ |
| Support vectors | samples with $\alpha_i>0$, satisfying $y_i f(x_i)=1$ |
| Kernel trick | replace inner products with $K(x_i,x_j)$, an implicit high-dimensional map |
| Soft margin | $0\le\alpha_i\le C$, Hinge loss + L2 |

**Mnemonic**: *regression uses least squares, classification uses cross-entropy; SVM looks at margins and the dual yields kernels; large C avoids misclassification, large gamma memorizes too finely.*

## 7. References

- [The Elements of Statistical Learning, Ch. 3–4, 12](https://hastie.su.domains/ElemStatLearn/)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [MLE Interview · Linear models and SVM](https://longxingtan.gitbook.io/mle-interview)
- [scikit-learn · Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn · SVM](https://scikit-learn.org/stable/modules/svm.html)
