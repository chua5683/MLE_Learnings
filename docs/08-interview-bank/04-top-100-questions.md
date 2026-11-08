# Top 100 Interview Questions

> **How to use**: cover the answer and self-quiz; follow the link for any question you miss. Answers are 2–4 lines of key points: conclusion + reason + likely follow-up direction.
>
> **Note**: questions are condensed from the Q&A across the book, grouped by topic and numbered continuously for random sampling.

## A. Machine Learning Fundamentals (Q1–Q22)

### Q1. Write down the bias-variance decomposition and explain each term.
**Answer**: $E[(y-\hat f)^2]=\mathrm{Bias}^2+\mathrm{Var}+\sigma^2$. Bias is the systematic gap between the model's assumptions and the true function; variance is sensitivity to training-set noise; $\sigma^2$ is the irreducible noise floor. More complex models lower bias and raise variance. Regularization and more data mainly reduce variance; better features or larger models reduce bias.
→ See [Bias variance and generalization](../01-ml-fundamentals/01-bias-variance-and-generalization.md)

### Q2. How do you detect overfitting, and how do you mitigate it?
**Answer**: Training loss is low while validation loss is high and the gap grows; the validation curve turns upward. Mitigations: more data, augmentation, L1/L2 regularization, dropout, early stopping, lower model capacity, cross-validated hyperparameters, ensembling.
→ See [Bias variance and generalization](../01-ml-fundamentals/01-bias-variance-and-generalization.md)

### Q3. What is the difference between L1 and L2 regularization, and why does L1 produce sparse solutions?
**Answer**: L1 penalizes the sum of absolute weights; L2 penalizes squared weights. Geometrically the L1 ball is a diamond, so the optimum often lands on an axis and some weights become exactly zero. Bayesian view: L1 is a Laplace prior (peaked), L2 is a Gaussian prior. L1 does feature selection; L2 is more stable and has a closed form.
→ See [Regularization and feature engineering](../01-ml-fundamentals/04-regularization-and-feature-engineering.md)

### Q4. Where does the logistic regression loss come from?
**Answer**: From the Bernoulli likelihood, take the negative log: $-[y\log\hat y+(1-y)\log(1-\hat y)]$ with $\hat y=\sigma(w^Tx)$. It is convex in $w$, and the gradient is $(\hat y-y)x$.
→ See [Linear models and SVM](../01-ml-fundamentals/02-linear-models-and-svm.md)

### Q5. What is the difference between logistic regression and SVM?
**Answer**: LR outputs probabilities, optimizes cross-entropy, and uses all samples; SVM is determined only by support vectors, optimizes hinge loss plus margin maximization, is naturally sparse and more robust to outliers, but needs calibration for probabilities. When linearly separable their decision boundaries are close; the losses differ.
→ See [Linear models and SVM](../01-ml-fundamentals/02-linear-models-and-svm.md)

### Q6. What splitting criteria do decision trees use, and what does each prefer?
**Answer**: Information gain (ID3; prefers many-valued features), gain ratio (C4.5; penalizes many-valued features), Gini impurity (CART; fast to compute). Regression trees use MSE/MAE reduction. Bias toward many-valued features can be mitigated with limits or gain ratio.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q7. Why is a random forest better than a single tree?
**Answer**: Bagging over samples and features reduces variance; averaging $T$ trees gives roughly $\rho\sigma^2+(1-\rho)\sigma^2/T$, so lower tree correlation $\rho$ helps, which is why random feature subsets matter. OOB samples give a nearly unbiased validation estimate.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q8. How is GBDT trained?
**Answer**: It is a sequential additive model: each round fits the current residual (negative gradient), $F_m=F_{m-1}+\nu h_m$, where $\nu$ is the shrinkage learning rate. Classification uses the negative gradient of log loss, regression uses residuals; each tree is a weak learner and the ensemble reduces bias.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q9. How does XGBoost improve on GBDT?
**Answer**: It takes a second-order Taylor expansion of the objective and adds explicit regularization on the number of leaves and leaf weights; splits are chosen by a gain formula; it supports column subsampling, a default direction for missing values, approximate quantile splits, feature-level parallelism, and cache optimizations. Users can supply custom losses.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q10. Why does XGBoost use second-order information?
**Answer**: The second-order term provides curvature, giving a closed-form optimal leaf weight $w^*=-G/(H+\lambda)$ and faster, scale-robust convergence; regularization shrinks leaf weights. First-order information gives only a direction whose step size must be tuned separately.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q11. What are the risks of LightGBM's leaf-wise growth, and how do you control them?
**Answer**: It splits the leaf with the largest gain, which converges quickly but can produce deep, unbalanced trees and overfit. Controls: num_leaves, max_depth, min_data_in_leaf, min_gain_to_split, bagging and feature sampling.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q12. How do you handle severe class imbalance?
**Answer**: Data level: undersampling, oversampling, SMOTE. Algorithm level: class weights / cost-sensitive learning, Focal Loss. Decision level: threshold moving and calibration. Evaluate with PR-AUC, recall at a fixed FPR, or business cost. Do oversampling inside each CV fold to avoid leakage.
→ See [Sampling and class imbalance](../01-ml-fundamentals/06-sampling-and-class-imbalance.md)

### Q13. What is the probabilistic meaning of AUC?
**Answer**: The probability that a randomly chosen positive sample receives a higher score than a randomly chosen negative sample. It equals the Wilcoxon rank-sum statistic, depends only on ranking, and is insensitive to class prevalence.
→ See [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)

### Q14. AUC is high but business results are poor. Why?
**Answer**: AUC measures global ranking, not top-k precision or a fixed-FPR operating point. Business may care about the head (use PR-AUC or recall@k), probabilities may be uncalibrated, the online distribution may have drifted, or sample weights may not match the business objective.
→ See [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)

### Q15. When do you use PR curves versus ROC curves?
**Answer**: With severe imbalance and a focus on the positive class, use PR-AUC: it is sensitive to the number of negatives and reflects head precision. With balanced classes or when overall ranking matters, ROC is more stable.
→ See [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)

### Q16. What is probability calibration, and why is it needed?
**Answer**: Calibration means predicted probabilities match observed frequencies, $P(y=1|\hat p=p)=p$. LR is naturally fairly calibrated; tree models and deep nets are often over- or under-confident. Use Platt scaling, isotonic regression, or temperature scaling. Ranking does not need calibration, but bidding and risk thresholds do.
→ See [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)

### Q17. What forms of cross-validation exist, and how do you do it for time series?
**Answer**: K-fold, stratified K-fold, leave-one-out, repeated K-fold; grouped CV prevents leakage from the same entity. For time series use rolling or expanding windows (walk-forward), always training on the past and validating on the future.
→ See [Bias variance and generalization](../01-ml-fundamentals/01-bias-variance-and-generalization.md)

### Q18. Give a few examples of data leakage.
**Answer**: Fitting scaling or oversampling before the split; using future features to predict the past; ID-like features that leak the label; target encoding computed outside the fold; the same user or session appearing in both train and test; selecting features on the full data before CV.
→ See [Regularization and feature engineering](../01-ml-fundamentals/04-regularization-and-feature-engineering.md)

### Q19. How do you compute feature importance, and what are the pitfalls?
**Answer**: Tree split gain or coverage, permutation importance, and SHAP. Split gain favors high-cardinality features; permutation importance underestimates correlated features; SHAP is consistent but expensive.
→ See [Tree models and ensembles](../01-ml-fundamentals/03-tree-models-and-ensembles.md)

### Q20. What do the E-step and M-step of EM do, and why does it converge?
**Answer**: The E-step computes the posterior over latent variables under current parameters and constructs the ELBO; the M-step maximizes that ELBO over parameters. The ELBO is non-decreasing and lower-bounds the log-likelihood, so EM converges to a local optimum (initialization matters).
→ See [Probabilistic models and EM](../01-ml-fundamentals/07-probabilistic-models-and-em.md)

### Q21. What is the difference between generative and discriminative models?
**Answer**: Generative models model $P(x,y)$ (naive Bayes, GMM): they can sample and handle missing data but make strong assumptions. Discriminative models model $P(y|x)$ directly (LR, SVM, trees): usually more accurate and data-efficient.
→ See [Probabilistic models and EM](../01-ml-fundamentals/07-probabilistic-models-and-em.md)

### Q22. What is backpropagation, fundamentally?
**Answer**: The chain rule plus dynamic programming on the computation graph: cache intermediates in the forward pass, then propagate gradients in reverse topological order; each node only needs its local Jacobian. Cost is the same order as the forward pass (about 2–3×).
→ See [Neural networks and backprop](../02-dl-fundamentals/01-neural-networks-and-backprop.md)

## B. Deep Learning (Q23–Q40)

### Q23. What causes vanishing/exploding gradients, and how do you fix them?
**Answer**: Deep products of Jacobians: spectral radius below 1 decays exponentially, above 1 explodes. Fixes: ReLU-family activations, proper initialization (Xavier/He), BatchNorm/LayerNorm, residual connections, gradient clipping, gated units (LSTM).
→ See [Neural networks and backprop](../02-dl-fundamentals/01-neural-networks-and-backprop.md)

### Q24. Why do residual connections make very deep networks trainable?
**Answer**: They provide an identity shortcut so gradients flow directly backward ($\partial x_{l+1}/\partial x_l=I+F'$), which alleviates vanishing gradients; they also reparameterize the network to learn a residual, which is easier to optimize, and ensemble many effective depths.
→ See [Neural networks and backprop](../02-dl-fundamentals/01-neural-networks-and-backprop.md)

### Q25. How does BatchNorm differ between training and inference?
**Answer**: Training normalizes with the current mini-batch statistics and updates running averages; inference uses the stored running mean/variance, so outputs do not depend on the batch. With very small batches the statistics are noisy, so GroupNorm/LayerNorm are better.
→ See [Normalization techniques](../02-dl-fundamentals/03-normalization-techniques.md)

### Q26. Why does BatchNorm help?
**Answer**: It reduces internal covariate shift (debated), smooths the loss landscape so gradients are more predictable, and allows larger learning rates; batch noise also adds a regularization effect. The practical key is stabilizing each layer's input scale.
→ See [Normalization techniques](../02-dl-fundamentals/03-normalization-techniques.md)

### Q27. What is the difference between BatchNorm and LayerNorm, and when do you use each?
**Answer**: BN normalizes across the batch per channel, depends on batch size, needs running statistics at inference, and suits vision with large batches. LN normalizes across features within each sample, is batch-independent, and suits variable-length sequences, NLP, and Transformers.
→ See [Normalization techniques](../02-dl-fundamentals/03-normalization-techniques.md)

### Q28. What is the difference between Pre-LN and Post-LN, and why do large models use Pre-LN?
**Answer**: Post-LN normalizes after the sublayer (original paper) and needs warmup to remain stable at depth; Pre-LN normalizes before the sublayer with a clean residual path, giving more stable gradients and often no warmup, at the cost of slightly weaker expressiveness and a final norm.
→ See [Normalization techniques](../02-dl-fundamentals/03-normalization-techniques.md)

### Q29. How are Xavier and He initialization derived?
**Answer**: The goal is to keep forward activation variance and backward gradient variance stable. Xavier assumes a linear activation and sets $\mathrm{Var}(w)=2/(n_{in}+n_{out})$; He accounts for ReLU killing half the units and sets $\mathrm{Var}(w)=2/n_{in}$.
→ See [Activations and initialization](../02-dl-fundamentals/02-activations-and-initialization.md)

### Q30. Why do dying ReLUs occur, and how do you fix them?
**Answer**: If a unit's input stays negative, its gradient is exactly zero and its weights stop updating. Fixes: LeakyReLU/ELU/GELU, a smaller learning rate, proper initialization, BatchNorm.
→ See [Activations and initialization](../02-dl-fundamentals/02-activations-and-initialization.md)

### Q31. What is the difference between Adam and SGD, and when is SGD better?
**Answer**: Adam adapts step sizes from first/second moments, converges fast, and is less sensitive to the learning rate; SGD with momentum sometimes generalizes better, uses less memory, and is common for large-batch vision training. LLM pretraining predominantly uses AdamW.
→ See [Optimizers and training tips](../02-dl-fundamentals/06-optimizers-and-training-tips.md)

### Q32. What does AdamW change, and why does it help?
**Answer**: It decouples weight decay from the gradient and applies $w\leftarrow w-\eta\lambda w$ directly, so the adaptive learning rate does not rescale the L2 term. This gives consistent regularization and is the default for large-model training.
→ See [Optimizers and training tips](../02-dl-fundamentals/06-optimizers-and-training-tips.md)

### Q33. Why do we need learning-rate warmup?
**Answer**: Early in training the parameters are random and gradients are unstable, and Adam's second-moment estimate is inaccurate; a large step can diverge. Warmup ramps the learning rate up gradually. Transformers especially depend on it.
→ See [Optimizers and training tips](../02-dl-fundamentals/06-optimizers-and-training-tips.md)

### Q34. Why does mixed-precision training need loss scaling?
**Answer**: FP16 underflows and turns small gradients into zero. Loss scaling multiplies the loss by a large constant so gradients stay in range, then divides before the update; dynamic scaling adjusts the factor automatically. BF16 has a wide exponent range and usually needs no scaling.
→ See [Optimizers and training tips](../02-dl-fundamentals/06-optimizers-and-training-tips.md)

### Q35. How does dropout work, and how do training and inference differ?
**Answer**: During training, units are zeroed with probability $p$ and the survivors are scaled by $1/(1-p)$ (inverted dropout); at inference nothing is dropped. It can be viewed as an ensemble of exponentially many subnetworks plus noise regularization.
→ See [Activations and initialization](../02-dl-fundamentals/02-activations-and-initialization.md)

### Q36. How do you compute a convolution's output size?
**Answer**: $H_{out}=\lfloor (H_{in}+2P-K)/S\rfloor+1$; for dilation, replace $K$ with $K_{eff}=K+(K-1)(d-1)$.
→ See [CNN and RNN](../02-dl-fundamentals/04-cnn-and-rnn.md)

### Q37. How do you compute the receptive field, and how do you increase it quickly?
**Answer**: Iteratively: $RF_l=RF_{l-1}+(K_l-1)\prod_{i<l}S_i$. To increase it: stack layers, use stride, dilated convolutions, or pooling.
→ See [CNN and RNN](../02-dl-fundamentals/04-cnn-and-rnn.md)

### Q38. What are 1x1 convolutions for?
**Answer**: Cross-channel linear mixing and dimensionality changes, reducing compute with bottlenecks, adding nonlinearity, and acting as a per-pixel fully connected layer; MobileNet's depthwise separable convolution uses them for channel mixing.
→ See [CNN and RNN](../02-dl-fundamentals/04-cnn-and-rnn.md)

### Q39. What is the complexity of self-attention, and why divide by $\sqrt{d_k}$?
**Answer**: $O(n^2d)$ time and $O(n^2)$ memory for the attention matrix. Dot-product variance grows linearly with $d_k$; dividing by $\sqrt{d_k}$ brings the variance back to 1 and avoids softmax saturation and vanishing gradients.
→ See [Transformer basics](../02-dl-fundamentals/05-transformer-basics.md)

### Q40. What are the five components of an MDP?
**Answer**: $(S,A,P,R,\gamma)$: states, actions, transition probabilities, rewards, and the discount factor. The goal is a policy that maximizes expected discounted return $E[\sum_t\gamma^t r_t]$.
→ See [MDP and Bellman equations](../03-reinforcement-learning/01-mdp-and-bellman-equations.md)

## C. Reinforcement Learning and Alignment (Q41–Q58)

### Q41. What is the difference between the Bellman expectation and optimality equations?
**Answer**: The expectation equation is for a fixed policy and is a linear system in $V^\pi$; the optimality equation contains $\max$, $V^*=\max_a[R+\gamma\sum P V^*]$, is nonlinear, and is solved by value or policy iteration.
→ See [MDP and Bellman equations](../03-reinforcement-learning/01-mdp-and-bellman-equations.md)

### Q42. What is the difference between value iteration and policy iteration?
**Answer**: Value iteration applies the optimal Bellman operator directly, one sweep per iteration. Policy iteration alternates full policy evaluation with policy improvement; it usually needs fewer outer iterations but each is more expensive. Both converge to an optimal policy.
→ See [Value based methods](../03-reinforcement-learning/02-value-based-methods.md)

### Q43. What is the difference between Monte Carlo and TD?
**Answer**: MC estimates values from complete returns: unbiased, high variance, and requires episode termination. TD bootstraps from one-step reward plus an estimate: biased, lower variance, and works online or in continuing tasks. TD($\lambda$) interpolates with eligibility traces.
→ See [Value based methods](../03-reinforcement-learning/02-value-based-methods.md)

### Q44. What is the difference between Q-learning and SARSA?
**Answer**: Q-learning is off-policy and updates with $\max_{a'}Q(s',a')$; SARSA is on-policy and updates with the action actually taken. Q-learning is more aggressive and can overestimate; SARSA is safer (the cliff-walking example).
→ See [Value based methods](../03-reinforcement-learning/02-value-based-methods.md)

### Q45. What are the two key tricks in DQN?
**Answer**: Experience replay (breaks correlation and improves sample efficiency) and a target network (stabilizes the bootstrapping target). Also common: Huber loss, gradient clipping, and frame stacking.
→ See [Value based methods](../03-reinforcement-learning/02-value-based-methods.md)

### Q46. Why does DQN overestimate Q-values, and how is it mitigated?
**Answer**: The max operator is positively biased under noise: $E[\max Q]\ge\max E[Q]$. Double DQN selects with the online network and evaluates with the target network; Dueling decomposes value and advantage; C51/QR-DQN use distributional values.
→ See [Value based methods](../03-reinforcement-learning/02-value-based-methods.md)

### Q47. State the policy gradient theorem and explain it.
**Answer**: $\nabla_\theta J=\mathbb E_{\pi_\theta}[\nabla_\theta\log\pi_\theta(a|s)Q^{\pi}(s,a)]$. The log-derivative trick turns a derivative of a distribution into a weighted derivative of actions, and the expectation is estimated by sampling. A baseline leaves the expectation unchanged and reduces variance.
→ See [Policy gradient and actor critic](../03-reinforcement-learning/03-policy-gradient-and-actor-critic.md)

### Q48. What is GAE, and how is it computed?
**Answer**: Generalized advantage estimation: $\hat A_t^{GAE}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$ with $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. $\lambda=0$ is one-step TD (low variance, high bias), $\lambda=1$ is Monte Carlo (high variance, low bias); PPO commonly uses 0.95.
→ See [Policy gradient and actor critic](../03-reinforcement-learning/03-policy-gradient-and-actor-critic.md)

### Q49. How should the PPO clip objective be understood, and why does it work?
**Answer**: $L=\mathbb E[\min(r_t\hat A_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$ with $r_t=\pi_\theta/\pi_{old}$. It limits how far one update can move the policy and prevents collapse; the min prevents over-increasing probabilities for positive advantage and over-decreasing them for negative advantage.
→ See [PPO and TRPO](../03-reinforcement-learning/04-ppo-and-trpo.md)

### Q50. What is the core idea of TRPO, and how does it relate to PPO?
**Answer**: Constrained policy update inside a trust region: maximize $\mathbb E[\frac{\pi_\theta}{\pi_{old}}\hat A]$ subject to $\bar{KL}(\pi_{old}\|\pi_\theta)\le\delta$, solved with the Fisher matrix and conjugate gradients to guarantee monotonic improvement. PPO approximates it with clipping or a KL penalty, which is simpler and nearly as effective.
→ See [PPO and TRPO](../03-reinforcement-learning/04-ppo-and-trpo.md)

### Q51. What do the three stages of RLHF do?
**Answer**: 1) SFT: supervised fine-tuning on human demonstrations; 2) reward model: train a scorer on human preference pairs (A > B); 3) PPO: optimize the policy against the reward model with a KL constraint. The point is to align with human preferences rather than merely imitate.
→ See [RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md)

### Q52. What loss does the reward model use, and why Bradley-Terry?
**Answer**: $\mathcal L=-\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))$. Bradley-Terry assumes the preference probability is a logistic function of the score difference; pairwise comparison is more stable and closer to human judgment than absolute scoring.
→ See [RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md)

### Q53. What is the role of the KL penalty in RLHF, and what happens if it is too large?
**Answer**: It keeps the policy close to the SFT model, preventing reward hacking and preserving language quality and diversity. Too large: conservative, learns little preference. Too small: score-chasing and degenerate outputs. Adaptive KL is common.
→ See [RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md)

### Q54. What are the practical pitfalls of PPO in RLHF?
**Answer**: Reward hacking, length bias (longer answers score higher), KL collapse, a hard-to-train value network, high memory (four models), hyperparameter sensitivity, and offline data distribution issues. Mitigations: length penalties, reward-model ensembles, adaptive KL, and LoRA with a shared base model.
→ See [RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md)

### Q55. Derive the DPO loss.
**Answer**: The KL-regularized optimum has the closed form $\pi^*(y|x)\propto\pi_{ref}(y|x)e^{r(x,y)/\beta}$; solving for $r=\beta\log\frac{\pi^*}{\pi_{ref}}$ and substituting into Bradley-Terry gives $\mathcal L=-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$. No explicit reward model is needed; the policy is optimized directly.
→ See [DPO and preference optimization](../03-reinforcement-learning/06-dpo-and-preference-optimization.md)

### Q56. Compare DPO and PPO.
**Answer**: DPO is simple, stable, needs only preference data and two models, but is offline, overfits preference data, has no exploration, and suffers length bias. PPO explores online, generalizes through a reward model, and has a higher ceiling, but is complex, memory-hungry, and unstable.
→ See [DPO and preference optimization](../03-reinforcement-learning/06-dpo-and-preference-optimization.md)

### Q57. What is the difference between GRPO and PPO?
**Answer**: GRPO samples a group of answers per prompt and normalizes rewards within the group to obtain relative advantages, removing the value network (critic). This saves memory and suits verifiable rewards; PPO needs a critic to estimate the baseline.
→ See [GRPO and reasoning RL](../03-reinforcement-learning/07-grpo-and-reasoning-rl.md)

### Q58. What is RLVR, and why does it work for reasoning models?
**Answer**: RL with verifiable rewards: rewards come from automatically checkable answers (math results, unit tests, format), avoiding reward-model noise and hacking. With GRPO it substantially improves pass@1 on reasoning tasks; the debate is whether it only amplifies existing capabilities.
→ See [GRPO and reasoning RL](../03-reinforcement-learning/07-grpo-and-reasoning-rl.md)

## D. Large Language Models (Q59–Q90)

### Q59. How does BPE tokenization work, and why subwords?
**Answer**: Start from characters or bytes and repeatedly merge the most frequent adjacent pair until the vocabulary reaches the target size. Subwords balance vocabulary size and OOV: common words stay whole, rare words split into pieces, which mitigates OOV and controls sequence length.
→ See [Language models and tokenization](../04-llm/01-language-models-and-tokenization.md)

### Q60. Why are mainstream LLMs decoder-only?
**Answer**: Dense supervision (every token is a training signal), the same objective as generation, inference needs only a KV cache and autoregression, a unified engineering stack, good scaling behavior, and strong in-context learning. Encoder-decoder remains more natural when input and output modalities differ a lot.
→ See [LLM architecture evolution](../04-llm/02-llm-architecture-evolution.md)

### Q61. What is the KV cache, and why does it speed up decoding?
**Answer**: During autoregressive generation, the K/V of past tokens are cached; a new token only computes its own Q and attends to the cache, avoiding recomputation and reducing per-step cost from $O(n^2)$ to $O(n)$. The price is memory that grows linearly with sequence length.
→ See [Inference optimization and serving](../04-llm/08-inference-optimization-and-serving.md)

### Q62. How do you estimate KV cache memory?
**Answer**: Per token per layer: $2\times n_{kv\_heads}\times d_{head}\times$ bytes, multiplied by layers, batch size, and sequence length. Example: Llama-3-8B (32 layers, 8 KV heads, head_dim 128, bf16) is about 128 KB/token, roughly 1 GB per 8k-token sequence.
→ See [Inference optimization and serving](../04-llm/08-inference-optimization-and-serving.md)

### Q63. What is the difference between MHA, MQA, and GQA?
**Answer**: MHA gives every head its own K/V. MQA shares one K/V across all heads, cutting KV cache by $1/h$ at some quality cost. GQA groups query heads and shares K/V within each group, trading quality against memory; Llama 2/3 use it.
→ See [Attention and position encoding](../04-llm/03-attention-and-position-encoding.md)

### Q64. What is MLA, and what problem does it solve?
**Answer**: Multi-head Latent Attention (DeepSeek): K/V are compressed into a low-dimensional latent and projected back, with a decoupled RoPE dimension. It shrinks the KV cache dramatically (90%+ in practice) while matching or exceeding MHA expressiveness.
→ See [Attention and position encoding](../04-llm/03-attention-and-position-encoding.md)

### Q65. What is the core idea of FlashAttention?
**Answer**: Use the GPU memory hierarchy and tiling so the $n\times n$ attention matrix is never written to HBM; accumulate with online softmax and recompute in the backward pass. Memory drops from $O(n^2)$ to $O(n)$, and the speedup comes from reduced HBM traffic, not fewer FLOPs.
→ See [Attention and position encoding](../04-llm/03-attention-and-position-encoding.md)

### Q66. How does RoPE work, and why can it represent relative positions?
**Answer**: It rotates q/k in 2D subspaces by angles proportional to position, so $\langle R_m q,R_n k\rangle$ depends only on $m-n$; the long-range decay property helps extrapolation. In practice cos/sin tables are precomputed and indexed by position.
→ See [Attention and position encoding](../04-llm/03-attention-and-position-encoding.md)

### Q67. What methods exist for length extrapolation?
**Answer**: Position interpolation (PI), NTK-aware scaling, YaRN, ALiBi, sliding-window attention, and progressive context extension during training. The goal is a longer effective context without full retraining; quality still depends on the training distribution.
→ See [Attention and position encoding](../04-llm/03-attention-and-position-encoding.md)

### Q68. What is the Chinchilla conclusion?
**Answer**: For a fixed compute budget, model size and training tokens should scale together, roughly 20 tokens per parameter; earlier models such as GPT-3 were undertrained. Later models like LLaMA deliberately overtrain smaller models to reduce inference cost.
→ See [Scaling laws](../04-llm/05-scaling-laws.md)

### Q69. Is emergent ability real?
**Answer**: It is heavily debated: some emergence may come from discontinuous metrics (e.g., exact match jumps for long answers), while token-level metrics look smoother. In practice focus on the task distribution and evaluation method, not a hard capability threshold.
→ See [Scaling laws](../04-llm/05-scaling-laws.md)

### Q70. How does LoRA work, and why is it effective?
**Answer**: Freeze the original weights and inject a low-rank update $\Delta W=BA$ ($B\in\mathbb R^{d\times r},A\in\mathbb R^{r\times k}$, $r\ll d$), training only A and B. The assumption is that fine-tuning updates are low-rank; trainable parameters and optimizer memory drop sharply, and the update can be merged back with zero inference overhead.
→ See [Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md)

### Q71. What does QLoRA do?
**Answer**: It quantizes the base model to 4-bit (NF4), adds double quantization and a paged optimizer, and trains the LoRA adapters in bf16. This allows fine-tuning a 65B model on a single 48GB GPU with quality close to full fine-tuning.
→ See [Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md)

### Q72. How do you estimate full fine-tuning memory?
**Answer**: About 16–20 bytes per parameter: weights 2B + gradients 2B + Adam first/second moments 8B (fp32) + activations. A 7B model needs roughly 112–140 GB, requiring ZeRO/FSDP sharding. LoRA only accounts for trainable parameters and activations, much less.
→ See [Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md)

### Q73. What do the three ZeRO stages shard?
**Answer**: Stage 1 shards optimizer states; stage 2 also shards gradients; stage 3 also shards parameters (all-gathered on demand in the forward pass). Memory drops roughly by $1/N$ for $N$ devices, at the cost of more communication.
→ See [Distributed training](../04-llm/07-distributed-training.md)

### Q74. What is the difference between tensor parallelism and pipeline parallelism?
**Answer**: TP splits a single layer's matrices across devices with frequent intra-layer communication (all-reduce), best within a node over NVLink. PP puts different layers on different devices and pipelines micro-batches (1F1B); communication is light but bubbles appear, and it suits cross-node scaling.
→ See [Distributed training](../04-llm/07-distributed-training.md)

### Q75. Why does the SFT loss mask the prompt?
**Answer**: Only answer tokens contribute to cross-entropy, so the model does not learn to repeat the user input and the loss focuses on the distribution to generate. In practice prompt labels are set to -100.
→ See [Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md)

### Q76. Why does catastrophic forgetting happen after fine-tuning, and how do you mitigate it?
**Answer**: Parameter updates overwrite pretrained knowledge and the data distribution is narrow. Mitigations: PEFT such as LoRA, mixing general data, low learning rate and few steps, KL regularization to the base model, multi-task training, and periodic evaluation of general capabilities.
→ See [Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md)

### Q77. What does PagedAttention solve?
**Answer**: It manages the KV cache in fixed-size blocks, like virtual memory, allocating on demand and supporting prefix sharing and copy-on-write. This nearly eliminates fragmentation and gives vLLM several times the throughput.
→ See [Inference optimization and serving](../04-llm/08-inference-optimization-and-serving.md)

### Q78. Why does continuous batching achieve higher throughput than static batching?
**Answer**: It schedules at iteration granularity: finished requests leave immediately and new ones join at any step, avoiding padding and idle time caused by waiting for the longest sequence in a static batch. With PagedAttention, memory utilization is also higher.
→ See [Inference optimization and serving](../04-llm/08-inference-optimization-and-serving.md)

### Q79. How does speculative decoding work?
**Answer**: A small draft model proposes $k$ tokens; the large model verifies them in parallel and corrects sampling according to acceptance probabilities. The accept-reject scheme preserves the target distribution and can give a 2–3× speedup when the draft is good, trading memory for latency.
→ See [Inference optimization and serving](../04-llm/08-inference-optimization-and-serving.md)

### Q80. What is the difference between PTQ and QAT?
**Answer**: PTQ quantizes after training and uses calibration data to set scales; it is simple but loses accuracy, especially at W4. QAT simulates quantization during training for better accuracy at higher cost. LLMs commonly use GPTQ/AWQ (PTQ) or FP8 post-training quantization.
→ See [Quantization and compression](../04-llm/09-quantization-and-compression.md)

### Q81. What is the difference between GPTQ and AWQ?
**Answer**: GPTQ performs layer-wise error compensation using the Hessian to quantize weights to 3–4 bits. AWQ observes that a few salient channels matter and protects them via per-channel scaling based on activation magnitude, which is more robust and general at 4-bit.
→ See [Quantization and compression](../04-llm/09-quantization-and-compression.md)

### Q82. How do you detect and mitigate benchmark contamination in LLM evaluation?
**Answer**: Use n-gram overlap detection, training-data provenance, dynamic or private test sets, and paraphrased questions; report contamination risk. Inflated leaderboard scores often come from contamination and prompt overfitting.
→ See [Evaluation and hallucination](../04-llm/10-evaluation-and-hallucination.md)

### Q83. What biases does LLM-as-a-judge have, and how do you calibrate it?
**Answer**: Position bias (prefers the first answer), length bias, self-preference, and style bias. Mitigations: swap order, use multiple judges, provide reference answers, pairwise comparison with a rubric, align with human annotations, and spot-check regularly.
→ See [Evaluation and hallucination](../04-llm/10-evaluation-and-hallucination.md)

### Q84. What causes hallucination, and how do you mitigate it?
**Answer**: Causes: missing or outdated pretraining knowledge, stochastic decoding, alignment that encourages confidence, and retrieval errors in RAG. Mitigations: RAG with citations, factual SFT/RLHF, constrained decoding, uncertainty estimation, tool-based verification, and faithfulness evaluation.
→ See [Evaluation and hallucination](../04-llm/10-evaluation-and-hallucination.md)

### Q85. What is the full RAG pipeline?
**Answer**: Offline: parse documents, chunk, embed, index. Online: rewrite the query, retrieve (vector/BM25/hybrid), rerank, assemble context, generate, then cite and post-process. Evaluation splits into retrieval (recall@k) and generation (faithfulness/relevance).
→ See [Rag](../04-llm/11-rag.md)

### Q86. What are common chunking and reranking strategies?
**Answer**: Chunk by semantics, headings, or fixed length with overlap, keeping metadata; rerank the top-k with a cross-encoder to balance recall and precision. Parent-child chunks or sentence-window retrieval help with long documents.
→ See [Rag](../04-llm/11-rag.md)

### Q87. How does HNSW work?
**Answer**: A hierarchical navigable small-world graph: sparse upper layers for long jumps and a dense bottom layer for fine search; queries descend greedily from the top, about $O(\log n)$. Parameters M, efConstruction, and efSearch control recall and latency.
→ See [Rag](../04-llm/11-rag.md)

### Q88. How do you choose between long context and RAG?
**Answer**: Long context suits deep understanding of a few documents with less engineering; RAG suits large, frequently updated corpora that need permissions and citations at lower cost. In practice they are combined: retrieve first, then read with a long context, or use long context for reranking.
→ See [Rag](../04-llm/11-rag.md)

### Q89. How do you defend against prompt injection?
**Answer**: Separate instructions from data, apply least privilege, whitelist tools and require human confirmation, filter inputs and outputs, sandbox execution, isolate the system prompt, and detect injection in retrieved content. There is no silver bullet; use defense in depth.
→ See [Prompt engineering](../04-llm/12-prompt-engineering.md)

### Q90. Why do CoT and self-consistency work?
**Answer**: CoT externalizes implicit computation into intermediate tokens, giving the model more computation steps; self-consistency samples multiple reasoning paths and votes, reducing single-path errors. For small models or trivial tasks they can be neutral or harmful.
→ See [Prompt engineering](../04-llm/12-prompt-engineering.md)

## E. AI Agents and Engineering (Q91–Q100)

### Q91. What are the usual components of an agent?
**Answer**: LLM (decision core) + planning (task decomposition/reflection) + tools (function calling/code execution) + memory (short-term context/long-term store) + runtime (loop, state, error handling).
→ See [Agent fundamentals and architecture](../05-agents/01-agent-fundamentals-and-architecture.md)

### Q92. What is the ReAct loop?
**Answer**: Thought → Action → Observation: the model reasons about the next step, calls a tool, observes the result, and continues until it produces a final answer. Interleaving reasoning and acting reduces hallucination and supports multi-step tasks.
→ See [Planning and reasoning](../05-agents/03-planning-and-reasoning.md)

### Q93. How does function calling work under the hood?
**Answer**: Tool descriptions (JSON Schema) are placed in the context; the model emits structured calls; the runtime parses them, executes, and feeds results back as messages. Parallel calls and streaming arguments are supported; it is essentially constrained format alignment.
→ See [Tool use and function calling](../05-agents/02-tool-use-and-function-calling.md)

### Q94. What is MCP, and what problem does it solve?
**Answer**: Model Context Protocol: a standard way for LLM applications to connect to external tools and data sources. It defines client/server roles, resources/tools/prompts, and transports (stdio/SSE/HTTP), reducing fragmented integrations and making tools reusable.
→ See [Tool use and function calling](../05-agents/02-tool-use-and-function-calling.md)

### Q95. How should agent memory be layered?
**Answer**: Short term: conversation and tool results in the context window. Long term: vector stores (semantic), structured databases (facts/user profiles), episodic memory (past trajectories), and procedural memory (skills/prompt templates). Writes need summarization and deduplication; retrieval needs relevance and recency.
→ See [Memory systems](../05-agents/04-memory-systems.md)

### Q96. When should you not use multi-agent systems?
**Answer**: When a single agent can do the task, when latency or cost is tight, or when strong consistency and debuggability are required. Multi-agent adds communication overhead, error propagation, and evaluation difficulty. Start with one agent plus tools; split only when justified.
→ See [Multi agent systems](../05-agents/05-multi-agent-systems.md)

### Q97. How do you evaluate an agent?
**Answer**: Task success rate, trajectory quality (steps/tool accuracy), cost (tokens/latency), robustness (perturbation/failure recovery), and safety. You need reproducible environments, fixed tool sets, LLM-as-a-judge plus human spot checks, and regression tests.
→ See [Evaluation and safety](../05-agents/07-evaluation-and-safety.md)

### Q98. What are the principles of good tool design?
**Answer**: Single responsibility, clear descriptions (when to use, inputs/outputs, limits), few stable parameters, idempotency and retryability, readable error messages, structured and truncated outputs, least privilege, and confirmation for destructive operations.
→ See [Tool use and function calling](../05-agents/02-tool-use-and-function-calling.md)

### Q99. How do you optimize agent cost?
**Answer**: Model routing (small models for easy steps), context compression/caching, step limits, parallel tool calls, truncating tool results, prompt caching, fast failure, and offline batch processing.
→ See [Evaluation and safety](../05-agents/07-evaluation-and-safety.md)

### Q100. What are the security risks of agents?
**Answer**: Prompt injection (especially from web pages or documents), tool abuse or privilege escalation, data exfiltration, sandbox escapes in code execution, supply-chain risks (MCP servers), and irreversible actions. Defenses: least privilege, sandboxing, human confirmation, audit logs, and output filtering.
→ See [Evaluation and safety](../05-agents/07-evaluation-and-safety.md)
