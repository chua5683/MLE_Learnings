# PPO and TRPO

> **One-line summary**: TRPO uses a KL trust region plus a monotonic-improvement lower bound to guarantee that each update does not make the policy worse; PPO approximates the trust region with a clipped ratio and became the de facto standard for LLM RLHF, reusing data with importance sampling while controlling update size with clipping.
>
> **Keywords**: `trust region` `KL constraint` `monotonic improvement` `PPO-clip` `surrogate objective` `GAE` `RLHF`

## 1. Core concepts

Policy gradients (Chapter 3) have a fatal weakness: **step size is defined in parameter space, but policy quality is measured in distribution space**. A parameter update of the same size can change the policy distribution drastically and destroy performance, and one bad update is hard to recover from (the on-policy data is gone).

TRPO and PPO answer the same question: **how do we push the reward up while bounding how far the new policy can drift from the old one?**

| Algorithm | Constraint | Solver | Engineering complexity |
| --- | --- | --- | --- |
| TRPO | Hard constraint $\mathbb{E}[\mathrm{KL}]\le\delta$ | Conjugate gradient + line search | High (second order) |
| PPO | Clipped ratio / KL penalty (soft constraint) | First-order SGD, multiple epochs | Low (mainstream) |

## 2. Principles and derivations

### 2.1 From policy gradients to a surrogate objective

Let the old policy be $\pi_{\mathrm{old}}$ and the new policy be $\pi_\theta$. The sample form of the policy gradient is

$$
\nabla_\theta J=\mathbb{E}_{s,a\sim\pi_{\mathrm{old}}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}\nabla_\theta\log\pi_\theta(a\mid s)A^{\pi_{\mathrm{old}}}(s,a)\right] \qquad (1)
$$

(because $\pi_\theta\nabla\log\pi_\theta=\nabla\pi_\theta$). We can therefore construct a **surrogate objective**:

$$
L_{\pi_{\mathrm{old}}}(\pi_\theta)=\mathbb{E}_{s,a\sim\pi_{\mathrm{old}}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}A^{\pi_{\mathrm{old}}}(s,a)\right] \qquad (2)
$$

At $\theta=\theta_{\mathrm{old}}$, $L$ has the same value and gradient as $J$, and it **can be estimated from old-policy data** (importance sampling), which is the precondition for "sample once, update many times."

### 2.2 The performance difference lemma and the monotonic-improvement bound

The policy performance difference lemma:

$$
J(\pi')-J(\pi)=\mathbb{E}_{s\sim d_{\pi'}}\left[\sum_a\pi'(a\mid s)A^{\pi}(s,a)\right] \qquad (3)
$$

Replacing the state distribution $d_{\pi'}$ by $d_\pi$ gives $L$, and the gap can be controlled by KL (Schulman et al., 2015):

$$
\left|J(\pi')-L_\pi(\pi')\right|\le C\max_s\mathrm{KL}\left(\pi(\cdot\mid s)\,\|\,\pi'(\cdot\mid s)\right),\qquad
C=\frac{4\epsilon\gamma}{(1-\gamma)^2},\ \epsilon=\max_{s,a}|A^{\pi}(s,a)| \qquad (4)
$$

Therefore

$$
J(\pi')\ge L_\pi(\pi')-C\max_s\mathrm{KL}\left(\pi\,\|\,\pi'\right) \qquad (5)
$$

As long as KL is small enough, the right-hand side is a lower bound on $J(\pi')$; maximizing this lower bound gives the **monotonic improvement** guarantee. TRPO relaxes the $\max_s$ to an expectation and obtains the trust-region problem:

$$
\max_\theta\ L_{\pi_{\mathrm{old}}}(\pi_\theta)\quad
\text{s.t.}\quad \mathbb{E}_{s\sim d_{\pi_{\mathrm{old}}}}\left[\mathrm{KL}\left(\pi_{\mathrm{old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\right)\right]\le\delta \qquad (6)
$$

### 2.3 How TRPO solves it

Take a second-order Taylor expansion of Eq. (6): the objective is $\approx \mathbf{g}^\top(\theta-\theta_{\mathrm{old}})$ and the KL constraint is $\approx\frac12(\theta-\theta_{\mathrm{old}})^\top F(\theta-\theta_{\mathrm{old}})\le\delta$, where $F=\mathbb{E}[\nabla\log\pi\,\nabla\log\pi^\top]$ is the **Fisher information matrix** (the second derivative of KL). Solving the Lagrangian gives the natural gradient direction:

$$
\theta=\theta_{\mathrm{old}}+\sqrt{\frac{2\delta}{\mathbf{g}^\top F^{-1}\mathbf{g}}}\,F^{-1}\mathbf{g} \qquad (7)
$$

$F$ is too large to invert explicitly, so conjugate gradient (CG) only needs matrix-vector products; a line search along the direction then enforces the KL constraint and improves the objective. The cost: second-order information, complex implementation, and incompatibility with dropout/BN and similar tricks, which makes it unfriendly in practice.

### 2.4 PPO-clip: a first-order approximation of the trust region

PPO defines the probability ratio:

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)} \qquad (8)
$$

The **clipped surrogate objective**:

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\ \mathrm{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)A_t\right)\right] \qquad (9)
$$

Term-by-term analysis ($\epsilon=0.2$):

- $A_t>0$ (the action is better than average): once $r$ exceeds $1+\epsilon$, the $\min$ selects the clipped term $(1+\epsilon)A_t$, whose gradient is zero, so **the update no longer pushes the action probability higher**.
- $A_t<0$ (the action is worse than average): once $r$ drops below $1-\epsilon$, the $\min$ selects $(1-\epsilon)A_t$, again with zero gradient, so **the update cannot crush the probability in one step**.
- Inside $[1-\epsilon,1+\epsilon]$, the gradient is the ordinary importance-sampled policy gradient.

The role of $\min$ is to take a **pessimistic lower bound**: it optimizes the smaller of the unclipped and clipped terms, preventing a single update from pushing the policy too far from the old one.

Another equivalent implementation is the **KL penalty**:

$$
L^{\mathrm{KL}}(\theta)=\mathbb{E}_t\left[r_t(\theta)A_t-\beta\,\mathrm{KL}\left(\pi_{\theta_{\mathrm{old}}}\,\|\,\pi_\theta\right)\right] \qquad (10)
$$

In practice an adaptive $\beta$ is common: increase it when KL exceeds the target, decrease it when below. PPO-clip is more common because it needs no $\beta$ tuning and is simpler to compute.

### 2.5 The full PPO loss

$$
L(\theta)=-\mathbb{E}_t\left[L^{\mathrm{CLIP}}_t-c_1\left(V_\phi(s_t)-V^{\mathrm{target}}_t\right)^2+c_2\,\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right)\right] \qquad (11)
$$

- $c_1\approx0.5$: value-loss weight; $c_2\approx0.01$: entropy bonus.
- $V^{\mathrm{target}}_t$ comes from GAE: $V^{\mathrm{target}}_t=A_t+V_\phi(s_t)$ (with $\gamma=1$).
- The value function is often clipped too: $V^{\mathrm{clip}}=\mathrm{clip}(V,V_{\mathrm{old}}\pm\epsilon_v)$, to keep value updates in check.

## 3. Implementation notes

A minimal PPO update (sampling and GAE omitted):

```python
import torch

def ppo_loss(logp, old_logp, values, old_values, adv, ret, clip=0.2, vf_coef=0.5, ent_coef=0.01, entropy=None):
    ratio = torch.exp(logp - old_logp)                 # r_t(θ)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # value clipping (optional)
    v_clip = old_values + torch.clamp(values - old_values, -clip, clip)
    vf_loss = 0.5 * torch.max((values - ret) ** 2, (v_clip - ret) ** 2).mean()

    loss = policy_loss + vf_coef * vf_loss
    if entropy is not None:
        loss = loss - ent_coef * entropy.mean()
    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()  # unbiased KL estimate
    return loss, approx_kl.detach()
```

Engineering notes (follow these and it trains):
- Compute GAE first to get `adv` and `ret`. **Standardize adv within the batch**; do not standardize ret.
- Run 4~10 epochs over the same batch with several minibatches; compute the ratio from the cached `old_logp`.
- Monitor `approx_kl`: if it stays above 0.02 (or 1.5x the target), stop the epochs for this batch early.
- Gradient clipping ($\|g\|_2\le0.5$); actor and critic may share a trunk or be separate.
- PPO is sensitive to reward scale: apply running mean/std normalization first, or clip rewards.

## 4. Interview Q&A

### Q1. Why is PPO more stable than vanilla policy gradient?
**Answer**: Vanilla PG takes a fixed step in parameter space, so the policy distribution can move too far in one step, destroying performance with no way to roll back. PPO measures how much the new policy differs from the old one via the importance ratio and clips the ratio to $[1-\epsilon,1+\epsilon]$, bounding the size of one update. This acts like a first-order trust region, so training is more stable and the same data can be reused for several epochs.
**Follow-up**: Is the clip a hard constraint? No, it is a soft constraint on the objective; the actual ratio can still go out of range, but crossing the boundary gives no gradient reward.

### Q2. How does TRPO guarantee monotonic improvement?
**Answer**: The performance difference lemma gives $J(\pi')-J(\pi)=\mathbb{E}_{s\sim d_{\pi'}}[\sum_a\pi'(a\mid s)A^\pi(s,a)]$. Replacing $d_{\pi'}$ with $d_\pi$ gives the surrogate $L_\pi(\pi')$, and the gap is controlled by $C\max_s\mathrm{KL}(\pi\|\pi')$ (Eq. 4). Hence $J(\pi')\ge L(\pi')-C\max\mathrm{KL}$, and maximizing the lower bound inside the KL trust region guarantees $J$ does not decrease. $\gamma<1$ and bounded rewards are what keep the constant $C$ finite.
**Follow-up**: Is the bound tight? It is very loose ($C$ can be huge), so practice uses a fixed $\delta$ (e.g., 0.01) instead of the theoretical bound.

### Q3. Why constrain KL rather than the parameter distance?
**Answer**: Policy performance depends only on the action distribution, not on the parameters. Distance in parameter space has no semantics: different parameters can represent the same policy, and a tiny parameter change can drastically change the distribution. KL directly measures the behavioral difference between old and new policies in the same state and is directly tied to the upper bound on performance difference, making it the more sensible trust-region metric.
**Follow-up**: Why is the direction $\mathrm{KL}(\pi_{\mathrm{old}}\|\pi_\theta)$? It measures how much the new policy deviates on states sampled by the old policy, which is convenient to estimate from samples.

### Q4. Write the PPO-clip objective and explain the roles of min and clip.
**Answer**: $L=\mathbb{E}[\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$ with $r_t=\pi_\theta/\pi_{\mathrm{old}}$. Clipping bounds the ratio neighborhood so a single update cannot be too large; min takes the smaller of the unclipped and clipped terms, forming a pessimistic bound: when $A>0$, going beyond $1+\epsilon$ earns no extra reward, and when $A<0$, going below $1-\epsilon$ earns no extra penalty. This limits both over-pursuit and over-suppression.
**Follow-up**: What if $A=0$? That term has no gradient and the sample is wasted, which is one reason advantage normalization matters.

### Q5. How does PPO-clip differ from the KL-penalty version? Which should you use?
**Answer**: Clip truncates the ratio directly, needs no extra hyperparameter, and is simple to compute. The KL-penalty version puts KL in the loss and needs a (often adaptive) $\beta$. Both approximate a trust region; theoretically the KL penalty is closer to TRPO's constrained form, while in practice clip is more common and robust, especially in RLHF. They can also be combined (clip plus a small KL penalty).
**Follow-up**: What are the advantages of the KL-penalty version? It controls KL per step precisely; the downside is sensitivity to $\beta$ and noisy KL estimates.

### Q6. PPO is on-policy, so why can it train multiple epochs on the same data?
**Answer**: Because of importance sampling: the loss uses $r_t=\pi_\theta/\pi_{\mathrm{old}}$ to correct the distribution gap, and the estimate remains meaningful as long as $\pi_\theta$ has not drifted too far from $\pi_{\mathrm{old}}$. Clipping keeps the ratio in a neighborhood, so the data stays "approximately on-policy" across epochs; with too many epochs the ratio goes out of range and the gradient vanishes, effectively stopping training.
**Follow-up**: Why resample at all? Because the older the data, the larger the importance-sampling variance; after clipping, the usable information is limited, so fresh samples from the current policy are required.

### Q7. What are the key PPO hyperparameters and how do you tune them?
**Answer**: Clip range (0.1~0.3, usually 0.2), learning rate (around 3e-4, often linearly decayed), number of epochs (4~10), minibatch size, GAE's $\gamma$ (0.99, or 1 for LLMs) and $\lambda$ (0.95), entropy coefficient (0.0~0.01), value-loss coefficient (0.5), and gradient clipping (0.5). Priority: learning rate and clip determine stability, $\lambda$ controls bias-variance, and the entropy coefficient controls exploration.
**Follow-up**: What do you check first when training is unstable? Approximate KL, value loss, entropy, and the reward curve; common causes are reward scale, unnormalized advantages, and a clip range that is too large.

### Q8. What role does PPO play in RLHF? Why is a critic needed?
**Answer**: The third stage of RLHF uses PPO to optimize "reward-model score + KL constraint." The critic estimates per-token values and, with GAE, produces token-level advantages, distributing the sequence-level reward across tokens. The critic is as large as the actor and is the main source of RLHF memory and engineering complexity (4 models: actor/critic/reward/ref). See Chapter 5 for details.
**Follow-up**: Why can't the critic be dropped? PPO needs values for GAE; without it you need a group-relative reward (GRPO, see Chapter 7) or another baseline.

### Q9. How should GAE be configured in PPO / RLHF?
**Answer**: Standard PPO uses $\gamma=0.99,\lambda=0.95$. In RLHF a response has finite length and the reward arrives at the end, so $\gamma=1$ is common, with $\lambda=0.95\sim1.0$, recursing backward over tokens as $A_t=\delta_t+\gamma\lambda A_{t+1}$. Note that prompt tokens do not contribute to the loss (they are masked); the policy loss is computed only on response tokens.
**Follow-up**: Why does $\gamma=1$ not diverge? Because the episode is finite and rewards are bounded, so the return is a finite sum.

### Q10. What are common PPO failure modes?
**Answer**: (1) Entropy collapse: the policy becomes deterministic too early and exploration disappears. (2) Value loss explodes or fails to converge: the critic cannot keep up with the actor. (3) Reward hacking: the reward model is exploited (length, sycophancy). (4) KL explosion: the policy drifts too far from the reference model. (5) After many epochs the ratio goes out of range and gradients vanish. Countermeasures include entropy bonuses, value clipping, KL penalties, reward normalization, and early stopping.
**Follow-up**: How do you tell it is reward hacking? Reward rises while human/held-out evaluation falls, or proxy metrics such as length and formatting spike abnormally.

## 5. Pitfalls and follow-ups

- **Do not flip the ratio**: $r=\pi_\theta/\pi_{\mathrm{old}}$; the denominator is the sampling policy.
- **old_logp must be cached and excluded from gradients**.
- **The value target and adv come from the same GAE pass**: $V^{\mathrm{target}}=A+V$; do not add the discount twice.
- **Mask prompt tokens in the loss**: RLHF trains only on the response.
- **Clip is not a hard constraint**: monitor the actual KL instead of assuming clip holds the line.
- **$\epsilon$ is coupled with the learning rate**: large clip + large lr = oversized updates.
- **With $\gamma=1$ the return spans the whole sequence**: length normalization is critical.

## 6. Cheat sheet

| Concept | Formula / key point |
| --- | --- |
| Importance sampling | $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$ |
| TRPO constraint | $\max L$ s.t. $\mathbb{E}[\mathrm{KL}]\le\delta$ |
| Monotonic improvement | $J(\pi')\ge L_\pi(\pi')-C\max_s\mathrm{KL}$ |
| PPO-clip | $\min(rA,\mathrm{clip}(r,1-\epsilon,1+\epsilon)A)$ |
| KL-penalty version | $rA-\beta\mathrm{KL}$ |
| Full loss | $-$clip $+c_1$ value MSE $-c_2$ entropy |
| Typical hyperparameters | $\epsilon=0.2$, $\lambda=0.95$, epochs 4~10 |

Mnemonic: **"TRPO draws a KL circle, PPO draws a clip circle; when A is positive do not chase the high, when A is negative do not crush it; min takes the pessimistic bound, and multiple epochs rely on importance sampling."**

## 7. References

- [TRPO · Schulman et al., 2015](https://arxiv.org/abs/1502.05477)
- [PPO · Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [GAE · Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
- [The 37 Implementation Details of PPO (Huang et al.)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [Secrets of RLHF in LLMs Part I: PPO](https://arxiv.org/abs/2307.04964)
- [DeepSpeed-Chat · RLHF PPO code walkthrough](https://github.com/microsoft/DeepSpeedExamples)
- [OpenAI Spinning Up · PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
