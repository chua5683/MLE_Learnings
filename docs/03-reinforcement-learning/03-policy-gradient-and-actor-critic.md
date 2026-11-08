# Policy Gradients and Actor-Critic

> **One-line summary**: Policy gradients differentiate $\log\pi_\theta(a\mid s)$ directly and take gradient steps weighted by the return/advantage; Actor-Critic uses a critic to estimate the advantage, replacing "wait for the full episode" with one-step bootstrapping, and GAE gives a tunable continuum between bias and variance.
>
> **Keywords**: `policy gradient theorem` `REINFORCE` `baseline` `advantage` `Actor-Critic` `A2C/A3C` `GAE` `entropy regularization`

## 1. Core concepts

Value-based methods first estimate values and then derive a policy from them; policy gradients (PG) instead **parameterize the policy directly**, $\pi_\theta(a\mid s)$, and perform gradient ascent on the objective $J(\theta)$. They naturally support stochastic policies and continuous actions, and they are the common ancestor of PPO, RLHF, and GRPO.

The RL objective:

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[R(\tau)\right],\qquad
R(\tau)=\sum_{t=0}^{T-1}\gamma^{t}r_{t+1} \qquad (1)
$$

where the trajectory is $\tau=(s_0,a_0,s_1,a_1,\dots)$ and $p_\theta(\tau)=\rho(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)$.

Three roles:

| Role | Function | What it maps to in an LLM |
| --- | --- | --- |
| Actor | Outputs the action distribution $\pi_\theta(a\mid s)$ | The generative model (policy model) |
| Critic | Estimates $V_\phi(s)$ or $Q_\phi(s,a)$ | The value model (PPO's critic) |
| Reward | Provides the immediate reward | Reward model / rule-based score |

## 2. Principles and derivations

### 2.1 The policy gradient theorem (step by step)

Differentiate Eq. (1) and use $\nabla_\theta p_\theta=p_\theta\nabla_\theta\log p_\theta$ (the log-derivative trick):

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\nabla_\theta\int p_\theta(\tau)R(\tau)\,d\tau
=\int \nabla_\theta p_\theta(\tau)R(\tau)\,d\tau\\
&=\int p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)R(\tau)\,d\tau
=\mathbb{E}_{\tau\sim p_\theta}\left[\nabla_\theta\log p_\theta(\tau)R(\tau)\right]
\end{aligned} \qquad (2)
$$

The key step: $\log p_\theta(\tau)=\log\rho(s_0)+\sum_t\left[\log\pi_\theta(a_t\mid s_t)+\log P(s_{t+1}\mid s_t,a_t)\right]$. **The environment dynamics $P$ do not depend on $\theta$, so their gradient is zero**, hence

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R(\tau)\right] \qquad (3)
$$

Eq. (3) is the naive form of the policy gradient theorem: **weight each action's log-prob by the return of the whole trajectory**.

**Causality (reward-to-go)**: rewards before time $t$ do not depend on $a_t$ and can be dropped (the rigorous argument uses $\mathbb{E}_{a_t}[\nabla\log\pi]=0$):

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\sum_{t'\ge t}\gamma^{t'-t}r_{t'+1}\right] \qquad (4)
$$

This is **REINFORCE**: sample trajectories, weight log-probs by the return-to-go, and ascend the gradient.

### 2.2 Baseline: why subtracting a function does not change the expectation

For any $b(s_t)$ that depends only on the state:

$$
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\right]
=b(s_t)\nabla_\theta\sum_{a}\pi_\theta(a\mid s_t)
=b(s_t)\nabla_\theta 1=0 \qquad (5)
$$

Therefore subtracting $b(s_t)$ from Eq. (4) **leaves the expectation unchanged and reduces the variance** (it removes the action-independent fluctuation shared by all actions). With $b(s_t)=V^{\pi}(s_t)$, the weight becomes the advantage $A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)$:

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi}(s_t,a_t)\right] \qquad (6)
$$

Intuition: it is not "reward is high, so raise the probability" but "**better than expected**, so raise the probability."

### 2.3 Actor-Critic and A2C/A3C

Bootstrapping the advantage with a critic $V_\phi$ gives the one-step form:

$$
A_t\approx \delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t) \qquad (7)
$$

- **Actor update**: $\theta\leftarrow\theta+\alpha_A\,\delta_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$.
- **Critic update**: minimize $\left(r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)\right)^2$ (or an n-step / GAE target).

| Dimension | A2C | A3C |
| --- | --- | --- |
| Parallelism | Multiple environments sample synchronously, gradients are aggregated, then one update | Multiple threads update shared parameters asynchronously |
| Update consistency | Synchronous, stronger on-policy guarantee | Asynchronous, parameters can be stale |
| Engineering | Easy to reproduce, fits GPU batching | Used on CPU clusters, now rare |

### 2.4 GAE: writing the λ-return as an advantage

Define the TD error $\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$; the k-step advantage is

$$
A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^{l}\delta_{t+l}=-V(s_t)+\sum_{l=0}^{k-1}\gamma^{l}r_{t+l+1}+\gamma^{k}V(s_{t+k}) \qquad (8)
$$

**Generalized Advantage Estimation (GAE)** exponentially weights all k-step advantages:

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}=(1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}A_t^{(k)} \qquad (9)
$$

Substitute Eq. (8) and swap the summation order; the coefficient of $\delta_{t+l}$ is $(1-\lambda)\sum_{k=l+1}^{\infty}\lambda^{k-1}=\lambda^{l}$, which gives the most useful recursive form in practice:

$$
\boxed{\;A_t^{\mathrm{GAE}}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}\;}
\qquad (10)
$$

The backward recursion is $A_t=\delta_t+\gamma\lambda A_{t+1}$, computed from the end of the trajectory in $O(T)$.

**What λ means**:

| $\lambda$ | Equivalent to | Bias | Variance |
| --- | --- | --- | --- |
| 0 | One-step TD: $\delta_t$ | High (depends on $V$'s accuracy) | Low |
| $(0,1)$ | The advantage version of TD(λ) | Medium | Medium |
| 1 | MC: $\sum_l\gamma^l\delta_{t+l}=G_t-V(s_t)$ | Low | High |

So GAE tunes the bias-variance tradeoff continuously with a single hyperparameter $\lambda$; PPO defaults to $\lambda=0.95$, and in LLM RLHF, since $\gamma=1$, $\lambda=0.95\sim1.0$ is common.

### 2.5 Entropy regularization

To encourage exploration and prevent the policy from collapsing to a deterministic one too early, add policy entropy to the objective:

$$
J_{\mathrm{ent}}(\theta)=\mathbb{E}\left[\sum_t r_t\right]+\beta\,\mathbb{E}\left[\sum_t\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right)\right] \qquad (11)
$$

The gradient of entropy pushes the distribution toward uniform (exploration); $\beta$ controls the strength, usually 0.0~0.01 in PPO/GRPO. Note that entropy regularization differs from KL regularization: entropy is "absolute randomness," while KL is "how far from the reference policy."

## 3. Implementation notes

A minimal REINFORCE implementation (with a baseline):

```python
import torch

def reinforce_loss(log_probs, returns, baselines=None):
    # log_probs: [T] log π(a_t|s_t) for each action
    # returns:   [T] reward-to-go (normalization optional)
    adv = returns - baselines if baselines is not None else returns
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # variance reduction
    return -(log_probs * adv.detach()).sum()
```

Actor-Critic (A2C) loss:

```python
def a2c_loss(log_probs, values, rewards, next_values, dones, gamma=0.99):
    with torch.no_grad():
        targets = rewards + gamma * (1 - dones) * next_values
        adv = targets - values                 # TD error = advantage estimate
    policy_loss = -(log_probs * adv.detach()).mean()
    value_loss = 0.5 * (values - targets).pow(2).mean()
    return policy_loss + 0.5 * value_loss, adv.mean()
```

Engineering notes:
- **The advantage must be detached**, otherwise the policy loss sends gradients into the critic.
- Apply running normalization to returns (especially when the reward scale is unknown).
- When using GAE, recurse backward along the trajectory and truncate at `done` boundaries.
- Learning rates: actor and critic can differ; the critic often uses a larger lr.
- Decay the entropy coefficient $\beta$ during training to avoid excessive randomness late in training.

## 4. Interview Q&A

### Q1. Derive the policy gradient theorem.
**Answer**: The objective is $J=\int p_\theta(\tau)R(\tau)d\tau$. Differentiating with respect to $\theta$ and substituting $\nabla p_\theta=p_\theta\nabla\log p_\theta$ gives $\mathbb{E}[\nabla\log p_\theta(\tau)R(\tau)]$. In $\log p_\theta(\tau)$ only $\sum_t\log\pi_\theta(a_t\mid s_t)$ depends on $\theta$ (the initial distribution and transition probabilities do not), so $\nabla J=\mathbb{E}[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)]$. Applying causality to drop future rewards yields REINFORCE.
**Follow-up**: Why is no derivative through the environment needed? Because the trajectory distribution depends on $\theta$ only through the policy; the environment dynamics are a black box, which is exactly why PG is model-free.

### Q2. Why use the log-derivative trick? What problem does it solve?
**Answer**: $\nabla_\theta p_\theta(\tau)$ cannot be estimated by sampling directly (we do not know the analytic form of $p_\theta$). Rewriting it as $p_\theta\nabla\log p_\theta$ makes the expectation an average over samples from the current policy, which is a Monte Carlo quantity we can estimate. This is the mathematical basis of all policy gradient methods.
**Follow-up**: What does the gradient of $\log\pi$ look like? For a softmax policy, $\nabla_\theta\log\pi_\theta(a\mid s)=\nabla_\theta z_a-\sum_{a'}\pi(a'\mid s)\nabla_\theta z_{a'}$, i.e., "the selected action's logit minus the expected logit."

### Q3. Why does a baseline reduce variance without adding bias?
**Answer**: Because $\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]=b(s)\nabla\sum_a\pi(a\mid s)=b(s)\nabla 1=0$ holds for any state-dependent $b$, so subtracting it leaves the expectation unchanged (unbiased). It removes the action-independent part of the return, concentrating the weights and lowering variance. The optimal baseline is approximately $V^{\pi}(s)$, in which case the weight is the advantage.
**Follow-up**: Can the baseline be arbitrary? It must be independent of the action; if it depends on the action, the expectation is no longer 0 and bias appears.

### Q4. What is the difference between REINFORCE and Actor-Critic?
**Answer**: REINFORCE estimates the gradient with full returns (MC): unbiased but high variance, must wait for the episode to end, and is on-policy only. Actor-Critic bootstraps TD errors/GAE with a critic as the advantage: lower variance and online updates, at the cost of critic bias. In practice AC/GAE is almost the default, and pure REINFORCE is mainly for teaching and small-scale tasks.
**Follow-up**: What if the critic is inaccurate? The advantage is biased and the update direction can be wrong; stabilize with critic warmup, a value-loss weight, and target normalization.

### Q5. How is GAE derived? Why the $(\gamma\lambda)^l$ weights?
**Answer**: The k-step advantage is $A_t^{(k)}=\sum_{l<k}\gamma^l\delta_{t+l}$, and GAE weights the k's geometrically: $A_t=(1-\lambda)\sum_k\lambda^{k-1}A_t^{(k)}$. After swapping sums, the coefficient of $\delta_{t+l}$ is $(1-\lambda)\sum_{k>l}\lambda^{k-1}=\lambda^l$; multiplying by $\gamma^l$ gives $(\gamma\lambda)^l$. The recursive implementation is $A_t=\delta_t+\gamma\lambda A_{t+1}$.
**Follow-up**: What does $\lambda=1$ give? $A_t=\sum_l\gamma^l\delta_{t+l}=G_t-V(s_t)$, the MC advantage.

### Q6. How does GAE's $\lambda$ affect bias and variance? How should it be tuned?
**Answer**: Larger $\lambda$ brings the advantage closer to the MC return: less bias (less dependence on the critic) and more variance; smaller $\lambda$ approaches one-step TD: more bias (strong dependence on the critic) and less variance. Usually 0.9~0.95; increase it if the critic is accurate and rewards are sparse, decrease it if training is jittery.
**Follow-up**: Is there a difference between $\gamma$ and $\lambda$? $\gamma$ defines the return discount (part of the problem), while $\lambda$ is only a statistical tradeoff when estimating the advantage (an algorithm hyperparameter) and does not change the true objective.

### Q7. What is the difference between A2C and A3C? Why is A3C rarely used now?
**Answer**: A2C is the synchronous version: multiple environments sample in parallel and all workers' gradients are aggregated before one update. A3C is the asynchronous version: each thread interacts with its own environment and updates shared parameters asynchronously (Hogwild). A3C's parameters can be stale and it is hard to reproduce; A2C exploits GPU batching, trains more stably, and with GAE is the precursor of modern PPO.
**Follow-up**: What does A3C's asynchrony buy? It does not wait for the slowest worker, so throughput is higher, but stale gradients hurt the on-policy assumption.

### Q8. Why add entropy regularization? How is it different from a KL penalty?
**Answer**: Entropy regularization keeps the policy stochastic and prevents premature convergence to a suboptimal deterministic policy; it is an exploration tool. A KL penalty keeps the new policy close to a reference policy (for example, preventing drift from the SFT model in RLHF); it is a stability/fidelity tool. Mathematically, entropy is $\mathcal{H}(\pi)$ and KL is $\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$: the former is reference-free, the latter needs a reference model.
**Follow-up**: What if the entropy coefficient is too large? The policy stays random and never learns a deterministic optimal behavior, so reward does not improve.

### Q9. Why is policy gradient on-policy? Can it be made off-policy?
**Answer**: The expectations in Eqs. (2)(3) are over trajectories sampled from the current policy $p_\theta$; once the policy updates, the old data distribution is off, so strictly speaking it is on-policy. Making it off-policy requires importance sampling $\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}$ (as in PPO's ratio) or corrections such as V-trace/Retrace; otherwise the gradient is biased.
**Follow-up**: What does PPO's clip do? It bounds the ratio, controls importance-sampling variance, and makes multi-epoch updates on "approximately on-policy" data feasible (see Chapter 4).

### Q10. Why is $\gamma$ often set to 1 in LLM RL?
**Answer**: An LLM response is a finite-horizon episode and the reward usually arrives only at the end (or per token from a reward model), so there is no infinite-horizon discounting need; $\gamma=1$ makes the rewards of the whole response add up with equal weight, matching the semantics of "how good is this whole answer." The price is higher return variance, which is counteracted by GAE's $\lambda<1$ and advantage normalization.
**Follow-up**: How do you compare responses of different lengths? Usually average per token or length-normalize; otherwise longer answers accumulate more reward by construction (one source of reward hacking).

## 5. Pitfalls and follow-ups

- **Do not use the whole-trajectory return as reward-to-go**: $a_t$ can only be judged by rewards after $t$; using $R(\tau)$ adds irrelevant noise (it remains unbiased, though).
- **The baseline must be action-independent**: using $Q(s,a)$ as a baseline introduces bias.
- **The log-prob must come from the current policy**: for multi-epoch PPO, cache the old policy's log-probs (`old_log_probs`).
- **Advantage normalization**: standardizing advantages within a batch is one of the key tricks for stable PPO training.
- **Entropy vs KL**: do not conflate the two.
- **$\gamma=1$ with long sequences**: the return scale grows with length, so length normalization or token-level rewards are required.

## 6. Cheat sheet

| Concept | Formula | Note |
| --- | --- | --- |
| Policy objective | $J=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$ | Optimize the policy directly |
| PG theorem | $\nabla J=\mathbb{E}[\sum_t\nabla\log\pi_\theta(a_t\mid s_t)R(\tau)]$ | No derivative through dynamics |
| reward-to-go | $\sum_{t'\ge t}\gamma^{t'-t}r_{t'+1}$ | Causality |
| baseline | $b(s_t)$ independent of the action | Unbiased, lowers variance |
| advantage | $A=Q-V$ | "How much better than expected" |
| TD error | $\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$ | One-step advantage |
| GAE | $A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ | Recursion $A_t=\delta_t+\gamma\lambda A_{t+1}$ |
| entropy regularization | $J+\beta\mathbb{E}[\mathcal{H}(\pi)]$ | Encourages exploration |

Mnemonic: **"PG multiplies log-prob by return; subtracting a baseline does not change the expectation; AC bootstraps with a critic; GAE links TD and MC with $(\gamma\lambda)^l$."**

## 7. References

- [Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) — Chapter 13, Policy Gradient
- [Williams, 1992 · Simple Statistical Gradient-Following Algorithms (REINFORCE)](https://link.springer.com/article/10.1007/BF00992696)
- [Sutton et al., 2000 · Policy Gradient Methods for RL with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
- [GAE · Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
- [A3C · Mnih et al., 2016](https://arxiv.org/abs/1602.01783)
- [OpenAI Spinning Up · VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- [Easy-RL · Policy Gradient](https://datawhalechina.github.io/easy-rl/#/chapter9/chapter9)
