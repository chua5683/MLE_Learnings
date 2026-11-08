# Probabilistic models and EM

> **One-line summary**: Probabilistic models describe uncertainty through joint distributions or posteriors; when latent variables make direct likelihood maximization hard, EM alternates between an E-step that computes posteriors and an M-step that maximizes the expected complete-data likelihood, improving the ELBO until convergence.
>
> **Keywords**: `naive Bayes` MLE/MAP Bayesian inference EM ELBO GMM HMM CRF

## 1. Core concepts

- **Generative vs discriminative models**: generative models learn $p(x,y)$ (or $p(x|y)p(y)$) and apply Bayes' rule to get $p(y|x)$; discriminative models learn $p(y|x)$ or the decision boundary directly. Naive Bayes, GMM, and HMM are generative; LR, SVM, and CRF are discriminative.
- **Latent variable**: a variable that is not observed but influences the data, such as cluster membership in GMM or the hidden state in HMM. With latent variables, the likelihood sums or integrates over them, making direct optimization hard, which is where EM comes in.
- **MLE / MAP / Bayesian inference**: two point-estimate flavors (frequentist and Bayesian) plus the full-posterior Bayesian approach.

| View | Objective | Characteristics |
| --- | --- | --- |
| MLE | $\arg\max_\theta\log p(D\vert \theta)$ | Asymptotically efficient, high variance on small samples |
| MAP | $\arg\max_\theta[\log p(D\vert \theta)+\log p(\theta)]$ | Prior = regularization, point estimate |
| Bayesian | Posterior $p(\theta\vert D)$ | Preserves uncertainty, expensive |
| EM | MLE/MAP with latent variables | Iterative, monotone, local optimum |

## 2. Principles and derivations

### 2.1 Naive Bayes

By Bayes' rule:

$$p(y|x)=\frac{p(y)p(x|y)}{p(x)} \quad\propto\quad p(y)\prod_{j=1}^{d}p(x_j|y)$$

"Naive" means **assuming features are conditionally independent given the class**. For classification we only compare numerators; the denominator $p(x)$ does not depend on the class.

- **Gaussian naive Bayes**: $p(x_j|y)=\mathcal{N}(x_j;\mu_{jy},\sigma_{jy}^2)$, suitable for continuous features.
- **Multinomial naive Bayes**: $p(x|y)\propto\prod_j p_{jy}^{n_j}$, suitable for word counts/counts.
- **Bernoulli naive Bayes**: features are 0/1, and "absence" is modeled explicitly.

**Laplace smoothing**: if a feature value never appears in a class, its MLE probability is 0, and after multiplying everything the posterior becomes 0. After smoothing,

$$p(x_j=v|y=c)=\frac{N_{jvc}+\alpha}{N_c+\alpha|V_j|}$$

$\alpha=1$ is Laplace smoothing, and $|V_j|$ is the number of possible values of feature $j$.

**Why does NB still work when conditional independence clearly fails?** Classification only requires $\arg\max_y p(y|x)$ to be correct, not the probabilities to be accurate; correlated features "double-count votes", which can make posteriors overconfident but often preserves the ordering. Its probabilities are usually **poorly calibrated**, but training is extremely fast and stable on small samples, making it a strong baseline.

### 2.2 MLE, MAP, and Bayesian inference

- **MLE**: $\hat\theta=\arg\max_\theta\log p(D|\theta)$, using data only.
- **MAP**: $\hat\theta=\arg\max_\theta[\log p(D|\theta)+\log p(\theta)]$; the prior acts as regularization. For example, a Gaussian prior with a Gaussian likelihood gives Ridge, and a Laplace prior gives Lasso.
- **Fully Bayesian**: keep the posterior $p(\theta|D)\propto p(D|\theta)p(\theta)$ and integrate at prediction time:

$$p(x_{new}|D)=\int p(x_{new}|\theta)p(\theta|D)\,d\theta$$

**Conjugate priors** keep the posterior in the same family as the prior, so updates are analytic: Beta-Bernoulli, Dirichlet-Multinomial, Normal-Normal (known variance), Gamma-Poisson. The benefit is simple computation and online updates; the cost is a restricted prior family.

### 2.3 EM derivation

Let $X$ be observed data, $Z$ latent variables, and $\theta$ parameters. The goal is the log marginal likelihood:

$$\log p(X|\theta)=\log\sum_Z p(X,Z|\theta)$$

Direct optimization requires summing (or integrating) over $Z$, which is usually hard. Introduce an arbitrary distribution $q(Z)$ and use Jensen's inequality or a direct decomposition:

$$\log p(X|\theta) =\underbrace{\sum_Z q(Z)\log\frac{p(X,Z|\theta)}{q(Z)}}_{\text{ELBO }\mathcal{L}(q,\theta)} +\underbrace{\sum_Z q(Z)\log\frac{q(Z)}{p(Z|X,\theta)}}_{KL(q\|p(Z|X,\theta))}$$

The KL divergence is non-negative, so the ELBO is a lower bound on the log-likelihood.

- **E-step**: fix $\theta=\theta_{old}$ and set $q(Z)=p(Z|X,\theta_{old})$. Then KL is 0 and the ELBO equals the current log-likelihood.
- **M-step**: fix $q$ and maximize the ELBO:

$$\theta_{new}=\arg\max_\theta\sum_Z q(Z)\log p(X,Z|\theta)$$

This maximizes the **expected complete-data log-likelihood**.

**Monotonicity**:

$$\log p(X|\theta_{new})\ge\mathcal{L}(q,\theta_{new})\ge\mathcal{L}(q,\theta_{old})=\log p(X|\theta_{old})$$

The first inequality follows from KL being non-negative, and the second from the M-step maximization. Hence EM never decreases the likelihood and converges to a local optimum or saddle point; initialization matters (GMM often uses KMeans initialization or multiple restarts).

### 2.4 Gaussian mixture model (GMM)

$$p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x|\mu_k,\Sigma_k), \qquad \sum_k\pi_k=1$$

**E-step**: compute the posterior (responsibility) that each sample belongs to each cluster:

$$\gamma_{ik}=\frac{\pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_{j}\pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)}$$

**M-step**:

$$N_k=\sum_i\gamma_{ik}, \quad \pi_k=\frac{N_k}{m}, \quad \mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i$$

$$\Sigma_k=\frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$$

**GMM vs KMeans**: KMeans makes hard assignments ($\gamma_{ik}\in\{0,1\}$), assumes isotropic clusters with equal variance, and updates only centers; GMM makes soft assignments and learns covariances and mixture weights, so it can represent elliptical clusters. KMeans can be seen as the limit of GMM when variances are fixed and driven to 0. GMM is sensitive to initialization, and covariances can become singular (a cluster left with a single point), so regularize with $\Sigma_k+\epsilon I$.

### 2.5 Hidden Markov model (HMM)

HMM describes a hidden-state sequence $z_{1:T}$ and an observation sequence $x_{1:T}$ through an initial distribution $\pi$, a transition matrix $A_{ij}=p(z_t=j|z_{t-1}=i)$, and emission probabilities $B_j(x_t)=p(x_t|z_t=j)$.

**Forward algorithm** computes the likelihood:

$$\alpha_t(j)=B_j(x_t)\sum_i\alpha_{t-1}(i)A_{ij}, \qquad p(x_{1:T})=\sum_j\alpha_T(j)$$

**Backward algorithm** multiplies backward from the future in a similar way; combining the two gives the state posterior $\gamma_t(i)=\alpha_t(i)\beta_t(i)/p(x)$ and the transition posterior $\xi_t(i,j)$.

**Viterbi decoding** (the most likely state sequence):

$$\delta_t(j)=\max_i[\delta_{t-1}(i)A_{ij}]\cdot B_j(x_t)$$

Backtracking through the stored argmax yields the optimal path; complexity is $O(TK^2)$.

**Baum-Welch = EM**: the E-step uses forward-backward to compute $\gamma,\xi$; the M-step updates $\pi,A,B$:

$$\hat A_{ij}=\frac{\sum_t\xi_t(i,j)}{\sum_t\gamma_t(i)}, \qquad \hat B_j(v)=\frac{\sum_{t:x_t=v}\gamma_t(j)}{\sum_t\gamma_t(j)}$$

**HMM vs CRF**: HMM is generative and assumes observations are conditionally independent given the state and depend only on the current state; CRF is discriminative and directly models $p(y|x)$, allowing arbitrary global features:

$$p(y|x)=\frac{1}{Z(x)}\exp\left(\sum_t\sum_k\lambda_k f_k(y_t,y_{t-1},x,t)\right)$$

CRF does not need the observation-independence assumption and is usually stronger for sequence labeling, but training is more expensive.

### 2.6 Relationship to deep models

- **VAE**: parameterizes $p(x|z)$ and $q(z|x)$ with neural networks and optimizes the ELBO; it is essentially "amortized" variational EM, with the encoder approximating the E-step and the decoder approximating the M-step.
- **Diffusion models**: multi-step latent-variable models whose training objective is also a variant of the ELBO.
- **Sequence labeling**: BiLSTM/Transformer + CRF remains a classic NER architecture; HMM is used in lightweight settings or as an interpretable baseline.
- **NB/LR**: NB is a generative baseline and LR is discriminative; under the naive-Bayes assumption the two can form a generative-discriminative hybrid.

## 3. Implementation notes

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture

# Gaussian naive Bayes: almost no tuning, a good small-sample baseline
nb = GaussianNB(var_smoothing=1e-9)
nb.fit(X_train, y_train)
print("NB log-loss:", -nb.predict_log_proba(X_test)[range(len(y_test)), y_test].mean())

# GMM: multiple restarts plus regularized covariance to avoid singularity
gmm = GaussianMixture(n_components=3, covariance_type="full",
                      reg_covar=1e-6, n_init=5, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)          # hard assignment
resp = gmm.predict_proba(X)      # E-step responsibilities
print("BIC:", gmm.bic(X))

# Hand-written GMM E-step core: log-sum-exp to prevent underflow
def e_step(X, pis, mus, Sigmas):
    logp = []
    for pi, mu, Sigma in zip(pis, mus, Sigmas):
        diff = X - mu
        inv = np.linalg.inv(Sigma)
        log_det = np.linalg.slogdet(Sigma)[1]
        logp.append(np.log(pi) - 0.5 * (diff @ inv * diff).sum(1)
                    - 0.5 * log_det - 0.5 * len(mu) * np.log(2 * np.pi))
    logp = np.vstack(logp).T
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)
```

HMMs are available in hmmlearn (GaussianHMM / MultinomialHMM); in production, watch out for numerical underflow and run the forward algorithm in log space with log-sum-exp.

## 4. Interview Q&A

### Q1. What is "naive" about naive Bayes, and why does it still work when the assumption fails?
**Answer**: Naive means features are conditionally independent given the class, $p(x|y)=\prod_j p(x_j|y)$. This assumption almost never holds in real data, but classification only requires $\arg\max_y p(y|x)$ to be correct, not the probabilities to be exact; correlated features double-count votes and make the posterior overconfident, yet often do not change the ordering. Training only needs counts or means and variances, is stable on small samples, and inference is fast, making it a strong baseline; the downsides are poor calibration and degraded performance when features are strongly correlated.
**Follow-up**: Why are NB posteriors extreme? The independence assumption double-counts correlations, amplifying the evidence and pushing probabilities toward 0/1.

### Q2. Why is Laplace smoothing necessary, and how should $\alpha$ be chosen?
**Answer**: If a feature value never occurs in a class, its MLE probability is 0; at test time, as soon as that feature appears, the product posterior becomes 0 and the model vetoes the class. Smoothing adds a pseudo-count $\alpha$ to every value, avoiding zero probabilities. $\alpha=1$ is Laplace smoothing, and $\alpha<1$ is Lidstone smoothing; choose it by cross-validation, with $\alpha\in[0.1,1]$ common in text classification. Too much smoothing weakens the discriminative signal.
**Follow-up**: Does smoothing affect rare classes more? Yes: the denominator $N_c+\alpha|V|$ has a larger relative effect when $N_c$ is small.

### Q3. What is the difference between generative and discriminative models?
**Answer**: Generative models learn $p(x,y)$ and use Bayes' rule to get $p(y|x)$; they can generate samples, handle missing data, and are theoretically more sample-efficient, but they must model the distribution of $x$ and rely on strong assumptions. Discriminative models learn $p(y|x)$ or the decision boundary directly; they usually achieve higher classification accuracy and optimize the target more directly, but cannot generate and may be overconfident out of distribution. NB/HMM/GMM are generative; LR/SVM/CRF are discriminative.
**Follow-up**: When are generative models better? With small samples, strong priors, a need for generation or missing-data completion, or semi-supervised settings.

### Q4. What is the difference among MLE, MAP, and Bayesian inference?
**Answer**: MLE maximizes the likelihood only and overfits on small samples; MAP adds a prior and produces a point estimate, equivalent to regularization (Gaussian prior → L2, Laplace → L1); fully Bayesian keeps the posterior and integrates over parameters at prediction time, representing uncertainty but at higher computational cost. As data grows, MAP and MLE become asymptotically identical; prior strength determines the difference on small samples.
**Follow-up**: What are conjugate priors good for? The posterior stays in the same family as the prior, updates are analytic, and they suit online learning and fast inference.

### Q5. Hand-derive EM and explain why it converges monotonically.
**Answer**: $\log p(X|\theta)=\log\sum_Z p(X,Z|\theta)$. Introduce $q(Z)$ and decompose into the ELBO $\sum_Z q\log[p(X,Z|\theta)/q]$ plus $KL(q\|p(Z|X,\theta))$. In the E-step, set $q=p(Z|X,\theta_{old})$; KL becomes 0 and the ELBO equals the current likelihood. In the M-step, maximize the ELBO to get $\theta_{new}$. Then $\log p(X|\theta_{new})\ge\mathcal{L}(q,\theta_{new})\ge\mathcal{L}(q,\theta_{old})=\log p(X|\theta_{old})$, so the likelihood never decreases.
**Follow-up**: Does EM always converge to the global optimum? No; it only guarantees a local optimum or saddle point, and initialization is crucial.

### Q6. EM only finds a local optimum. How can this be mitigated?
**Answer**: Multiple restarts (keep the highest likelihood), KMeans/PCA initialization, simulated annealing or random restarts, adding priors and running MAP-EM, or using better optimizers (gradient methods, variational inference). For GMM, regularize the covariance and restrict covariance_type to reduce degrees of freedom. In theory, check whether different restarts reach the same final ELBO.
**Follow-up**: How fast does EM converge? Linear convergence, possibly very slow near the optimum; use parameter change or ELBO improvement as stopping criteria.

### Q7. Write the GMM E-step and M-step. How does GMM relate to KMeans?
**Answer**: The E-step computes responsibilities $\gamma_{ik}=\pi_k\mathcal{N}(x_i|\mu_k,\Sigma_k)/\sum_j\pi_j\mathcal{N}(x_i|\mu_j,\Sigma_j)$; the M-step updates $N_k=\sum_i\gamma_{ik}$, $\pi_k=N_k/m$, $\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i$, and $\Sigma_k=\frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$. KMeans is the special case with hard assignments and isotropic, equal-variance clusters; GMM makes soft assignments and learns covariances, so it can represent elliptical clusters and provide probabilities.
**Follow-up**: Why can GMM covariances become singular? When a cluster receives very few samples its variance tends to 0; use reg_covar regularization or reduce the number of clusters.

### Q8. What is the difference between the HMM forward algorithm and Viterbi?
**Answer**: The forward algorithm sums over all possible paths, $\alpha_t(j)=B_j(x_t)\sum_i\alpha_{t-1}(i)A_{ij}$, producing the likelihood of the observation sequence $p(x_{1:T})$; Viterbi replaces the sum with a max, $\delta_t(j)=\max_i[\delta_{t-1}(i)A_{ij}]B_j(x_t)$, and backtracks to obtain the most likely state sequence. The former is used for evaluation/EM, the latter for decoding. Both have complexity $O(TK^2)$.
**Follow-up**: Why can't Viterbi just take the per-step maximum from forward results? Per-step maxima do not guarantee a globally optimal path; dynamic-programming backtracking is required.

### Q9. How does Baum-Welch relate to EM?
**Answer**: Baum-Welch is EM for HMM parameter estimation. The E-step uses forward-backward to compute state posteriors $\gamma_t(i)$ and transition posteriors $\xi_t(i,j)$; the M-step updates $\pi,A,B$ from these expected counts. It also monotonically increases the observed likelihood and converges to a local optimum; initialization (e.g., uniform or KMeans) affects the result.
**Follow-up**: How do you choose the number of hidden states in an HMM? Use validation likelihood/BIC/downstream metrics, or set it from business priors.

### Q10. What is the difference between CRF and HMM, and why is CRF common for NER?
**Answer**: HMM is generative, models $p(x,z)=p(z)p(x|z)$, and assumes observations are conditionally independent given the state; CRF is discriminative, directly models $p(y|x)=\exp(\sum_t\sum_k\lambda_k f_k(y_t,y_{t-1},x,t))/Z(x)$, and can use arbitrary overlapping features (word shape, prefixes/suffixes, context windows) without the observation-independence assumption. NER features are highly overlapping and context matters, so CRF is usually stronger; but CRF training/inference is more expensive, while HMM is faster and more interpretable.
**Follow-up**: What inference is used for linear-chain CRF? Forward-backward for marginals and Viterbi for decoding, with the same order of complexity as HMM.

### Q11. Give some examples of conjugate priors.
**Answer**: Beta-Bernoulli (coin-flip success rate), Dirichlet-Multinomial (multi-class counts/topic models), Normal-Normal (estimating a mean with known variance), Gamma-Poisson (count rates), and Wishart-Normal (covariance). Conjugacy makes the posterior analytic and incrementally updatable; pseudo-counts can be interpreted as prior sample size.
**Follow-up**: Are conjugate priors always good? Not necessarily; the restricted family may fail to express true beliefs, in which case use variational inference/MCMC.

### Q12. What are the pros and cons of EM versus gradient descent?
**Answer**: EM has a closed-form update per step, monotonically increases the likelihood, and needs no learning rate, making it suitable for models with latent variables and tractable M-steps (GMM, HMM, mixture models); but it only finds local optima, can converge slowly, and the M-step may not have a closed form. Gradient descent is more general and supports stochastic/large-scale training, but needs a learning rate and may not be monotone. In practice, use EM for initialization and gradient methods for fine-tuning, or use variational inference directly.
**Follow-up**: Can the EM E-step be approximated? Yes; variational EM uses an approximate posterior $q$, trading exactness for scalability (the idea behind VAE).

## 5. Pitfalls and follow-ups

- **NB probabilities are not calibrated**: they are often overconfident, so do not plug them directly into expected-revenue calculations.
- **The EM E-step is not "completing the data"**: it computes the latent posterior or expected sufficient statistics.
- **EM is monotone in likelihood, not in validation metrics**: during overfitting, training likelihood rises while validation may fall.
- **Choosing the number of GMM clusters**: use BIC/AIC, validation likelihood, or downstream tasks; do not look only at likelihood (it increases with more clusters).
- **The HMM observation-independence assumption**: observations are independent given the state, so serial correlation can only be carried through hidden states.
- **The CRF partition function**: training requires log Z and inference uses forward-backward; do not confuse it with the HMM likelihood.

## 6. Cheat sheet

| Topic | Key formula / conclusion |
| --- | --- |
| Naive Bayes | $p(y\vert x)\propto p(y)\prod_j p(x_j\vert y)$ |
| Laplace smoothing | $(N_{jvc}+\alpha)/(N_c+\alpha\vert V_j\vert )$ |
| MLE/MAP | Likelihood / likelihood + prior (= regularization) |
| EM decomposition | $\log p(X\vert \theta)=ELBO+KL$ |
| E-step | $q(Z)=p(Z\vert X,\theta_{old})$ |
| M-step | Maximize $\sum_Z q\log p(X,Z\vert \theta)$ |
| GMM E-step | $\gamma_{ik}\propto\pi_k\mathcal{N}(x_i\vert \mu_k,\Sigma_k)$ |
| GMM M-step | $\pi_k=N_k/m$, weighted updates for $\mu_k$ and $\Sigma_k$ |
| HMM forward | $\alpha_t(j)=B_j(x_t)\sum_i\alpha_{t-1}(i)A_{ij}$ |
| Viterbi | Replace the forward sum with max + backtracking |
| CRF | $p(y\vert x)=\exp(\sum\lambda_k f_k)/Z(x)$ |

**Mnemonic**: *NB relies on independence, EM relies on a lower bound; E-step finds posteriors, M-step finds parameters; GMM assigns softly, HMM solves with forward passes; CRF is discriminative, so stack any features you like.*

## 7. References

- [Pattern Recognition and Machine Learning, Ch. 9](https://link.springer.com/book/10.1007/978-0-387-45528-0)
- [The EM Algorithm and Extensions](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470191613)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
- [scikit-learn · Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)
- [hmmlearn documentation](https://hmmlearn.readthedocs.io/)
