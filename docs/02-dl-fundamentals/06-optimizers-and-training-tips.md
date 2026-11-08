# Optimizers and Training Tricks

> **One-line summary**: The optimizer decides how to take a step once the gradient is available - SGD is stable but slow, while Adam adaptively scales each parameter's learning rate using first and second moments and converges fast; what actually keeps large-model training stable is usually not the optimizer itself but the combination of warmup, gradient clipping, mixed precision, and gradient accumulation.
>
> **Keywords**: `SGD` `Momentum` `Adam` `AdamW` `bias correction` `warmup` `cosine` `gradient clipping` `mixed precision` `gradient accumulation`

## 1. Core concepts

Training means solving an optimization problem:

$$\min_{\theta}\;L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell\big(f_\theta(x_i),y_i\big)$$

Every first-order optimizer can be written in the same template:

$$\theta_{t+1}=\theta_t-\eta_t\,u_t$$

The only difference is how the update direction/scale $u_t$ is computed from gradient history:

- **SGD**: $u_t=g_t$, one learning rate for all parameters.
- **Momentum**: an exponential moving average of gradients that exploits past directions.
- **Adaptive methods** (AdaGrad/RMSProp/Adam): scale the learning rate per parameter using historical squared gradients.
- **Second-order methods**: correct the direction with Hessian information (rarely used directly in deep learning because of cost).

## 2. Principles and derivations

### 2.1 SGD and mini-batch

Estimate the gradient with a mini-batch $\mathcal{B}$:

$$g_t=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\nabla_\theta\ell_i(\theta_t), \qquad \theta_{t+1}=\theta_t-\eta g_t$$

$g_t$ is an unbiased estimate of the full gradient, with variance inversely proportional to batch size. SGD is simple and often generalizes better; the downsides are one shared learning rate for all parameters and severe oscillation on poorly conditioned (ravine-shaped) loss surfaces.

### 2.2 Momentum and Nesterov

$$v_t=\mu v_{t-1}+g_t, \qquad \theta_{t+1}=\theta_t-\eta v_t$$

$v_t$ is an exponential moving average of gradients, equivalent to adding inertia: dimensions with a consistent direction accumulate speed, while oscillating dimensions cancel out. In steady state the effective learning rate is about $\eta/(1-\mu)$ (about 10x when $\mu=0.9$).

**Nesterov** is "look ahead along the momentum, then compute the gradient there":

$$v_t=\mu v_{t-1}+\nabla_\theta L(\theta_t-\eta\mu v_{t-1})$$

It is more sensitive to overshooting, has a better theoretical convergence bound, and is often slightly more stable than standard momentum in practice.

### 2.3 AdaGrad and RMSProp

**AdaGrad** accumulates all historical squared gradients:

$$G_t=G_{t-1}+g_t^2, \qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t}+\epsilon}\,g_t$$

Each parameter's learning rate decreases monotonically during training, which suits sparse features (rare features get larger updates), but $G_t$ only grows: the learning rate approaches 0 late in training and progress stalls prematurely.

**RMSProp** replaces "accumulate" with an "exponential moving average":

$$s_t=\beta s_{t-1}+(1-\beta)g_t^2, \qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{s_t}+\epsilon}\,g_t$$

The learning rate no longer decays monotonically, making it a common improvement for non-stationary objectives.

### 2.4 Adam and bias correction

Adam = Momentum + RMSProp:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t, \qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat{m}_t=\frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t=\frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1}=\theta_t-\eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**Why bias correction is needed**: with $m_0=0$, expanding gives $m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$. For stationary gradients, $\mathbb{E}[m_t]=(1-\beta_1^t)\,\mathbb{E}[g]$, so early estimates are systematically shrunk (with $\beta_1=0.9$ the first step is only $0.1$ of the true gradient). Dividing by $1-\beta_1^t$ gives $\mathbb{E}[\hat{m}_t]=\mathbb{E}[g]$, and the same holds for $v$. Without correction the first few updates are abnormally small, especially for $v$ with $\beta_2=0.999$, where the bias is worse.

**Default hyperparameters**: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, with $\eta$ often $10^{-3}$ (LLM training commonly uses $\beta_2=0.95$, $\epsilon=10^{-8}$, and a smaller $\eta$).

### 2.5 AdamW: decoupled weight decay

**L2 regularization + Adam (coupled)**: add $\lambda\theta$ to the gradient, then divide everything by $\sqrt{\hat{v}}$:

$$g_t\leftarrow g_t+\lambda\theta_t \quad\Rightarrow\quad \theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t+\lambda\theta_t}{\sqrt{\hat{v}_t}+\epsilon}$$

The problem: parameters with large gradients have large $\sqrt{\hat{v}}$, so the regularization term is adaptively shrunk; parameters with small gradients are regularized harder. **The assumption that regularization strength is independent of the parameter is broken.**

**AdamW (decoupled)**:

$$\theta_{t+1}=\theta_t-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}}_t+\epsilon}+\lambda\theta_t\right)$$

Weight decay acts directly on the parameters without adaptive scaling, so the regularization strength is controllable. This is the standard recipe for Transformer/LLM training.

### 2.6 Learning-rate schedules

| Schedule | Formula / recipe | Characteristics |
| --- | --- | --- |
| Step decay | Multiply by $\gamma$ every $k$ epochs | Simple, requires manually set milestones |
| Cosine | $\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$ | Smooth annealing, common for LLMs |
| Linear warmup | Linearly rise from 0 to peak over the first $T_w$ steps | Stabilizes early training |
| Inverse sqrt | $\eta_t\propto1/\sqrt{t}$ | Used by the original Transformer |

**Why warmup is needed**: early in training Adam's second-moment estimate is still inaccurate ($v$ has seen only a few batches), so the gradient direction is noisy; with a large learning rate, a large model can have its pretrained/initialized state destroyed in one update, or the loss can spike. Starting with a small learning rate lets the statistics settle before rising to the peak, the cheapest stabilization trick available. Warmup is typically 1%-5% of total steps.

### 2.7 Training tricks

**Gradient clipping**: clip by global norm to prevent individual batches from exploding:

$$g\leftarrow g\cdot\min\left(1,\frac{c}{\lVert g\rVert_2}\right)$$

$c$ is usually 1.0. It only handles exploding gradients, not vanishing ones.

**Mixed precision**:

- **fp16**: 5 exponent bits and 10 mantissa bits, a small dynamic range, so gradients easily underflow/overflow and **loss scaling** is needed (scale up the loss, then scale the gradients back down).
- **bf16**: 8 exponent bits and 7 mantissa bits, the same dynamic range as fp32, usually no loss scaling needed, slightly lower precision but more stable training - the first choice for modern large-model training.
- Typically paired with fp32 master weights and fp32 for sensitive ops such as normalization and softmax.

**Gradient accumulation**: when a large batch does not fit in memory, accumulate gradients over $k$ micro-batches and update once:

$$B_{eff}=B_{micro}\times k\times N_{GPU}$$

This is approximately equivalent to a large batch (note that BN statistics are still computed per micro-batch, so it is not strictly equivalent).

**Other**: weight EMA (use smoothed parameters at inference), label smoothing, dropout, and gradient checkpointing (trade compute for memory).

### 2.8 Tuning experience

- **Learning rate is the first priority**, usually searched on a $\log$ scale ($10^{-5}\sim10^{-1}$).
- **Batch size and lr**: linear scaling $\eta\propto B$ (common for large batches) or square-root scaling $\eta\propto\sqrt{B}$; returns diminish beyond a critical batch size.
- **Weight decay**: often $0.1$ for LLMs, and **no decay on bias/LN parameters**; $10^{-4}\sim10^{-2}$ for small models.
- **Adam $\beta_2$**: LLMs often use $0.95$ instead of $0.999$, so the second moment adapts faster and loss spikes are reduced.
- **Gradient clipping**: $1.0$ is a safe default; when training diverges, lower the lr first and then inspect the data.

## 3. Implementation notes

```python
import torch
from torch.optim import AdamW

model = torch.nn.Linear(128, 10)

# parameter groups: no weight decay for bias / norm
decay, no_decay = [], []
for name, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)
opt = AdamW([
    {"params": decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)

# typical loop with gradient accumulation + clipping + mixed precision (bf16)
scaler = torch.cuda.amp.GradScaler(enabled=False)   # bf16 usually needs no scaler
accum_steps = 8
opt.zero_grad(set_to_none=True)
for step, (x, y) in enumerate(loader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = loss_fn(model(x), y) / accum_steps     # average first to avoid scaling gradients up
    loss.backward()
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```

Note: during gradient accumulation the loss must be divided by `accum_steps`, otherwise the effective learning rate is multiplied.

## 4. Interview Q&A

### Q1. How do you choose between SGD and Adam?
**Answer**: Adam converges fast, is more robust to the learning rate and sparse gradients, and needs almost no tuning, which suits Transformers/LLMs and large-scale training; SGD + momentum often generalizes better and uses less memory in CV (especially small data and fine-tuning), but needs more careful lr and schedule tuning. Rule of thumb: AdamW for large models, sparse gradients, or fast iteration; try SGD for vision tasks where final accuracy matters and you have time to tune.
**Follow-up**: Why does Adam sometimes generalize worse than SGD? One explanation is that adaptive learning rates make effective step sizes differ too much across parameters, favoring sharper minima; others point to different hyperparameter and regularization settings, and the debate is ongoing.

### Q2. Why does Momentum accelerate convergence?
**Answer**: It replaces the gradient with an exponential moving average; dimensions with a consistent direction accumulate and the effective learning rate rises to about $\eta/(1-\mu)$, while oscillating dimensions cancel and noise is smoothed. On a ravine-shaped loss surface it therefore accelerates along the long axis and damps oscillation along the short axis.
**Follow-up**: What is the difference between Momentum and Nesterov? Nesterov looks ahead along the momentum before computing the gradient, making it more sensitive to overshoot and giving a better convergence bound.

### Q3. Why does Adam need bias correction?
**Answer**: $m_t,v_t$ are initialized to 0, so the early moving averages are systematically underestimated: $\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]$, and with $\beta_1=0.9$ the first step is only 0.1 of the true value; the bias is even larger for $v$ with $\beta_2=0.999$. Dividing by $1-\beta^t$ makes the estimate unbiased; otherwise the first few updates are abnormally small or even directionally distorted.
**Follow-up**: What is the cost of bias correction? Just two divisions, negligible; late in training $1-\beta^t\to1$ and the correction disappears automatically.

### Q4. How do Adam and AdamW differ, and why do Transformers use AdamW?
**Answer**: Adam adds L2 regularization to the gradient, which is then scaled together with the gradient by $\sqrt{\hat{v}}$, so parameters with large gradients get weak regularization and those with small gradients get strong regularization - regularization strength runs out of control. AdamW decouples weight decay and applies it directly to the parameters, so the strength is independent of gradient magnitude. Transformers/LLMs generally use AdamW because it trains more stably, generalizes better, and makes weight decay interpretable.
**Follow-up**: Are L2 regularization and weight decay equivalent under SGD? Under plain SGD they are (both equal $\theta\leftarrow(1-\eta\lambda)\theta-\eta g$); under adaptive methods such as Adam they are not, which is the motivation for AdamW.

### Q5. What is the difference between AdaGrad and RMSProp?
**Answer**: AdaGrad accumulates **all** historical squared gradients, $G_t=G_{t-1}+g_t^2$, so the learning rate decreases monotonically; it suits sparse data but stalls late in training. RMSProp switches to an exponential moving average $s_t=\beta s_{t-1}+(1-\beta)g_t^2$, so the learning rate does not decay forever, which suits non-stationary deep-network training. Adam's $v_t$ is exactly RMSProp's second moment.
**Follow-up**: Where is AdaGrad still useful? On traditional NLP or recommendation tasks with strong sparse features, and as an object of theoretical analysis.

### Q6. Why does large-model training need learning-rate warmup?
**Answer**: (1) Early on, Adam's second-moment estimate is inaccurate and the gradient direction is noisy. (2) Large models are sensitive to the distribution in the initial phase, and one large-lr step can damage the parameters and cause a loss spike. (3) Warmup lets the model enter a more stable region with small steps before rising to the peak. It is a cheap way to trade a few steps for training stability.
**Follow-up**: What is wrong with too long a warmup? It wastes training steps and leaves the learning rate too low later; 1%-5% is usually enough, depending on model size and optimizer.

### Q7. What does gradient clipping solve, and how do you choose the threshold?
**Answer**: It solves exploding gradients - when a batch's gradient norm is abnormally large, one update can blow up the parameters (NaN or a sudden loss jump). Clip the global norm $\lVert g\rVert_2$ to a threshold $c$, keeping the direction and only shrinking the scale. $c$ is usually 1.0; too small slows convergence, too large becomes ineffective. It does not solve vanishing gradients.
**Follow-up**: Is clipping the same as gradient normalization? No: clipping only scales when the threshold is exceeded and otherwise keeps the original scale; normalization always forces the norm to a fixed value.

### Q8. How does mixed-precision training work, and how do you choose between fp16 and bf16?
**Answer**: Forward and backward passes use low precision for matrix multiplication (Tensor Core acceleration, less memory), while parameter updates and sensitive ops (softmax, LN, loss) stay in fp32. fp16 has few exponent bits and a small dynamic range, so loss scaling is needed to prevent gradient underflow; bf16 has the same exponent bits as fp32 and enough range, usually needs no scaling, has slightly lower precision, and barely affects training. Large-model training prefers bf16; on older GPUs that do not support it, use fp16 + a scaler.
**Follow-up**: Why are fp32 master weights still needed? Updating directly in low precision loses precision (especially for small gradients); keeping the master weights in fp32 and converting back after the update is the standard practice.

### Q9. Why can gradient accumulation simulate a large batch, and where is it not equivalent?
**Answer**: Summing/averaging gradients over $k$ micro-batches before one update matches the gradient estimate of a $k$-times-larger batch (in the averaging sense), so it bypasses the memory limit of a single GPU. It is not equivalent because BN statistics are still computed per micro-batch and dropout masks differ, so it is not strictly the same; SyncBN or other adjustments are needed to get close.
**Follow-up**: Does gradient accumulation affect the learning rate? Yes: if the loss is not divided by the number of accumulation steps, it is equivalent to multiplying the lr by $k$; the correct approach is to average the loss first.

### Q10. How are batch size and learning rate related?
**Answer**: Within a range, keeping $\eta\propto B$ (linear scaling) maintains similar training dynamics; a more conservative heuristic is $\eta\propto\sqrt{B}$. Beyond a point the gradient noise is already small and adding batch size yields diminishing returns (the critical batch size); at that stage, increase the number of training steps or the model size instead.
**Follow-up**: Why does linear scaling not hold indefinitely? Too large an lr destabilizes training, and the error of large-batch gradient estimates decreases more slowly; in practice this is paired with warmup and lr search.

### Q11. Why is second-order optimization rarely used in deep learning?
**Answer**: Newton's method requires the Hessian $H$ and its inverse; with $10^8\sim10^{11}$ parameters that is $O(n^2)$ memory and $O(n^3)$ compute, completely infeasible, and deep-network losses are non-convex with possibly indefinite Hessians. Compromises include L-BFGS (approximate inverse Hessian, $O(nk)$ memory, suitable for small models or full batches), K-FAC, and Shampoo, but on large models they are still less practical than first-order adaptive methods.
**Follow-up**: Is second-order information still useful in deep learning? Yes, for analyzing the loss landscape, estimating curvature, and pruning/quantization (for example diagonal Hessian approximations), just not as the main optimizer.

### Q12. When training loss spikes or becomes NaN, in what order do you debug?
**Answer**: First check for exploding gradients (print grad norm, enable clipping, lower the lr); then check mixed precision (fp16 overflow - switch to bf16 or adjust the loss scale); then the data (outlier samples, label errors, sudden length changes); then optimizer state ($\beta_2$ too large, eps too small); finally insufficient warmup or initialization/normalization issues. Diagnostic tools: per-layer grad norm, activation norm, and bisecting by step to reproduce.
**Follow-up**: Why do LLM trainings often use $\beta_2=0.95$? A shorter second-moment memory lets $v$ adapt faster to changes in gradient scale and reduces loss spikes, at the cost of slightly noisier estimates.

## 5. Pitfalls and follow-ups

- **Forgetting to divide by the accumulation steps**: the effective learning rate is multiplied and the loss takes off.
- **Applying weight decay to bias/LN in AdamW**: usually these should be grouped out to avoid unnecessary shrinkage.
- **No loss scaling with fp16**: gradients underflow to 0 and training looks fine but never converges.
- **Clipping after `step()`**: clip before the update, otherwise the order makes it ineffective.
- **Warmup and cosine not joined up**: peak learning rate, total steps, and minimum lr must line up, otherwise the annealing shape is wrong.

## 6. Cheat sheet

| Optimizer | Update core | Key hyperparameters | Characteristics |
| --- | --- | --- | --- |
| SGD | $g_t$ | $\eta$ | Simple, generalizes well, needs lr tuning |
| Momentum | $v_t=\mu v_{t-1}+g_t$ | $\mu=0.9$ | Accelerates, damps oscillation |
| Nesterov | Look-ahead gradient + momentum | $\mu$ | Better convergence bound |
| AdaGrad | Accumulated $g^2$ | $\eta$ | Sparse-friendly, stalls late |
| RMSProp | EMA of $g^2$ | $\beta=0.9$ | Non-stationary objectives |
| Adam | First + second moments + bias correction | $\beta_1,\beta_2,\epsilon$ | Fast convergence, general-purpose |
| AdamW | Adam + decoupled weight decay | Same + $\lambda$ | Default for Transformers/LLMs |

| Trick | Effect | Default / recipe |
| --- | --- | --- |
| warmup | Stabilizes early training | Linear ramp over 1%-5% of steps |
| cosine decay | Smooth annealing | Peak → minimum lr |
| grad clip | Prevents gradient explosion | Global norm 1.0 |
| bf16/fp16 | Saves memory, speeds up | Prefer bf16 for large models |
| grad accumulation | Simulates a large batch | Divide the loss by accumulation steps |
| weight decay | Regularization | 0.1 (LLM), excluding bias/LN |

**Mnemonic**: Adam is fast, SGD is steady, AdamW decouples regularization; warmup prevents an early blow-up, clipping prevents a mid-training blow-up, bf16 prevents overflow.

## 7. References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983): cosine
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md): optimizer comparison
- [llm_interview_note · wdndev](https://github.com/wdndev/llm_interview_note): large-model training details
