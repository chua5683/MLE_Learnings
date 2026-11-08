# Value-Based Methods

> **One-line summary**: Value-based methods first estimate $V/Q$ by sampling or dynamic programming and then derive a policy from the values; the main line runs DP -> MC/TD -> DQN -> distributional/continuous-action methods, and the core tension is always bias, variance, and sample efficiency.
>
> **Keywords**: `policy iteration` `value iteration` `Monte Carlo` `TD(0)` `TD(λ)` `Sarsa` `Q-learning` `DQN` `Double DQN` `Dueling` `PER` `Rainbow` `DDPG/TD3/SAC`

## 1. Core concepts

Value-based methods have two steps:

1. **Prediction (policy evaluation)**: given a policy $\pi$, estimate $V^{\pi}$ or $Q^{\pi}$.
2. **Control**: improve the policy using the estimated values, alternating until convergence.

A 2x2 map based on "needs an environment model" and "bootstraps or not":

| | Needs a model | Does not need a model |
| --- | --- | --- |
| **No bootstrap** (uses the full return) | — | Monte Carlo (MC) |
| **Bootstrap** (uses estimated values) | Dynamic programming (DP) | TD / DQN |

- **DP**: knows $P,R$ and performs full backups (sweeps) with the Bellman equations.
- **MC**: rolls out complete episodes and updates with the unbiased estimate $G_t$.
- **TD**: samples one step and uses $r+\gamma V(s')$ as the target, trading bias for variance.

## 2. Principles and derivations

### 2.1 Policy evaluation, policy iteration, and value iteration

**Policy evaluation** (iterative form):

$$
V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma V_k(s')\right] \qquad (1)
$$

**Policy improvement**: $\pi'(s)=\arg\max_a Q^{\pi}(s,a)$. The policy improvement theorem guarantees $V^{\pi'}(s)\ge V^{\pi}(s)$, and if no further improvement is possible the policy is optimal. Policy iteration = evaluate to convergence + improve, repeat.

**Value iteration**: treat the optimal backup itself as the update

$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma V_k(s')\right] \qquad (2)
$$

Value iteration is policy iteration with "one evaluation step per improvement," and it converges to $V^{*}$.

### 2.2 Monte Carlo (MC)

Estimate the value function from actual returns:

$$
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right] \qquad (3)
$$

- Unbiased ($G_t$ is an unbiased sample of $V^{\pi}$) but **high variance**; must wait for the episode to end.
- Only works for episodic tasks; needs exploring starts or $\varepsilon$-soft behavior to ensure coverage.

### 2.3 TD(0), n-step returns, and TD(λ)

TD(0) uses a one-step bootstrapped target:

$$
V(S_t)\leftarrow V(S_t)+\alpha\left[\underbrace{R_{t+1}+\gamma V(S_{t+1})}_{\text{TD target}}-V(S_t)\right] \qquad (4)
$$

The term in brackets is the **TD error** $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$. It is **biased** ($V$ itself is being estimated) but has **low variance**, updates online, and does not wait for the episode to end.

The n-step return interpolates the two:

$$
G_t^{(n)}=R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^{n}V(S_{t+n}) \qquad (5)
$$

TD(λ) uses the geometrically weighted λ-return:

$$
G_t^{\lambda}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)} \qquad (6)
$$

$\lambda=0$ reduces to TD(0) and $\lambda=1$ reduces to MC. The forward view needs to see the future; the backward view implements it online with **eligibility traces**:

$$
E_t(s)=\gamma\lambda E_{t-1}(s)+\mathbb{1}(S_t=s),\qquad V\leftarrow V+\alpha\delta_t E_t \qquad (7)
$$

The two views are equivalent in expectation (Sutton & Barto, Chapter 12).

### 2.4 Sarsa and Q-learning

**Sarsa (on-policy)**: update with the next action $A_{t+1}$ actually taken

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right] \qquad (8)
$$

The name comes from the tuple $(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})$.

**Q-learning (off-policy)**: update with the max over greedy actions

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\right] \qquad (9)
$$

The classic cliff-walking difference: Sarsa learns the "safe path," while Q-learning learns the "optimal but cliff-hugging path" (because the behavior policy explores, and Sarsa prices that exploration risk into its values).

**Overestimation (maximization bias)**: $\mathbb{E}[\max_i(Q_i+\epsilon_i)]\ge\max_iQ_i$ (Jensen's inequality). Noise makes the max systematically too high, so Q-learning overestimates.

### 2.5 DQN: making Q-learning work with neural networks

The DQN loss:

$$
L(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(r+\gamma\max_{a'}Q(s',a';\theta^{-})-Q(s,a;\theta)\right)^2\right] \qquad (10)
$$

Two key engineering pieces:

| Component | Problem it solves | How it works |
| --- | --- | --- |
| Experience replay (replay buffer) | Correlated samples, wasted data | Breaks temporal correlation, approximates i.i.d., and allows reuse |
| Target network | Target drifts with the parameters, training diverges | Computes the target with old parameters $\theta^{-}$, synced every $C$ steps or via Polyak averaging |

**Why does DQN work while "vanilla Q-learning + a neural network" collapses?** Because function approximation + bootstrapping + off-policy learning together create the **deadly triad**: value-estimation errors get amplified by bootstrapping. Replay and the target network mitigate correlation and target drift respectively.

### 2.6 The DQN improvement family

| Method | Core change | Problem it solves |
| --- | --- | --- |
| Double DQN | Online network selects the action, target network evaluates it: $a^{*}=\arg\max_{a'}Q_\theta(s',a')$, $y=r+\gamma Q_{\theta^{-}}(s',a^{*})$ | Decouples selection from evaluation, suppresses overestimation |
| Dueling DQN | $Q(s,a)=V(s)+A(s,a)-\frac{1}{\lvert\mathcal{A}\rvert}\sum_{a'}A(s,a')$ | Separates state value from action advantage; more stable with many actions |
| Prioritized Replay | Sample by $p_i\propto\lvert\delta_i\rvert^{\alpha}$ with importance weight $w_i\propto(1/(N\cdot P(i)))^\beta$ | Learns more from "very wrong" samples, improving sample efficiency |
| Noisy Net | Inject parameter noise into the network instead of $\varepsilon$-greedy | Learnable, state-dependent exploration |
| C51 / distributional RL | Learn the return distribution rather than its expectation | Preserves uncertainty, more stable |
| n-step DQN | Use n-step returns as the target | Bias-variance tradeoff |
| Rainbow | Combine all of the above | The SOTA bundle of its time |

### 2.7 Continuous actions: DDPG / TD3 / SAC

With continuous actions $\max_{a'}Q(s',a')$ cannot be enumerated, so an actor outputs actions directly:

- **DDPG**: deterministic actor $\mu_\theta(s)$ + critic $Q_\phi$; off-policy, replay, target networks, and action noise for exploration. Target: $y=r+\gamma Q_{\phi^{-}}(s',\mu_{\theta^{-}}(s'))$.
- **TD3**: three fixes, namely clipped double-Q (take the smaller of two critics), delayed policy updates, and target policy smoothing (add noise to the target action).
- **SAC**: maximum-entropy objective

$$
J(\pi)=\mathbb{E}_{\pi}\left[\sum_t r_t+\alpha\mathcal{H}\left(\pi(\cdot\mid s_t)\right)\right] \qquad (11)
$$

The soft Q update replaces the max with $V(s')-\alpha\log\pi(a'\mid s')$, and the temperature $\alpha$ can be tuned automatically; exploration comes from the stochastic policy itself, which makes SAC the most common baseline in continuous control.

## 3. Implementation notes

A single DQN update (the core 20 lines):

```python
import torch, torch.nn.functional as F

def dqn_update(batch, q_net, target_net, optimizer, gamma=0.99):
    s, a, r, s2, done = batch          # tensors; done: float 0/1
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: online network selects, target network evaluates
        a_star = q_net(s2).argmax(dim=1, keepdim=True)
        q_next = target_net(s2).gather(1, a_star).squeeze(1)
        target = r + gamma * (1.0 - done) * q_next
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
```

Engineering notes:
- The replay buffer usually holds $10^5\sim10^6$ transitions; train after warmup. Hard-sync the target network every $10^3\sim10^4$ steps (or use Polyak averaging with $\tau=0.005$).
- Use a Huber loss (`smooth_l1`) to resist outlier TD errors, and gradient clipping to prevent explosions.
- Evaluate with $\varepsilon=0$; train with $\varepsilon$ linearly decayed from 1.0 to about 0.05.
- Normalizing observations and actions is critical in continuous control; SAC is relatively robust to hyperparameters.

## 4. Interview Q&A

### Q1. What is the difference between policy iteration and value iteration? Which is faster?
**Answer**: Policy iteration = full policy evaluation (solve a linear system or iterate to convergence) + policy improvement; each round is expensive but there are few rounds. Value iteration = one optimal backup per round; each round is cheap but there are many rounds. Both converge to $V^{*}$. In practice, with many states prefer value iteration/asynchronous DP; when the action set is small and evaluation has a closed form, policy iteration is cheaper.
**Follow-up**: What is generalized policy iteration (GPI)? Alternating evaluation and improvement at any granularity; value iteration and policy iteration are both special cases.

### Q2. What is the difference between MC and TD? When do you use each?
**Answer**: MC uses the full return $G_t$: unbiased, high variance, waits for the episode end, and only fits episodic tasks. TD uses $R_{t+1}+\gamma V(S_{t+1})$: biased, low variance, online/incremental, and fits continuing tasks. TD converges faster on most tabular tasks (Sutton & Barto's random-walk experiment), while MC is more stable when returns themselves have low variance.
**Follow-up**: Where does TD's bias come from? The target contains $V(S_{t+1})$, which is computed from current estimates; if the estimate is off, the target is biased.

### Q3. Why is Q-learning off-policy while Sarsa is on-policy?
**Answer**: Q-learning's target is $\max_{a'}Q(s',a')$, independent of which action is actually taken and of the behavior policy, so data from any behavior policy can be used. Sarsa's target is $Q(s',a')$, where $a'$ is the action the current policy would actually take, so behavior and target policies are tied together. The cliff-walking result, where Sarsa learns the safe path and Q-learning the optimal path, is the direct consequence.
**Follow-up**: What about expected Sarsa? It takes an expectation over $a'$, $\sum_{a'}\pi(a'\mid s')Q(s',a')$, which has lower variance than Sarsa and is still on-policy.

### Q4. Why does DQN need experience replay and a target network?
**Answer**: Experience replay breaks the temporal correlation of samples, making SGD approximately i.i.d. and improving data utilization. The target network freezes the "moving target," preventing the target from chasing the parameters during bootstrapping and causing divergence. The two address correlation and target drift in the deadly triad respectively.
**Follow-up**: What are the downsides of replay? It only fits off-policy methods; stale data introduces distribution shift; a large buffer costs memory.

### Q5. Why does DQN overestimate? How does Double DQN fix it?
**Answer**: The target uses $\max_{a'}Q_{\theta^{-}}(s',a')$, and since $Q$ is noisy, $\mathbb{E}[\max]$ is larger than $\max\mathbb{E}$ (Jensen), so the estimate is systematically high. Double DQN lets the online network choose the action and the target network evaluate it: $a^{*}=\arg\max Q_\theta(s',a')$, $y=r+\gamma Q_{\theta^{-}}(s',a^{*})$, reducing the positive bias that comes from using one network to both select and evaluate.
**Follow-up**: Does Double DQN eliminate overestimation? No, it only mitigates it, and it can introduce mild underestimation.

### Q6. Why is the Dueling DQN architecture effective?
**Answer**: It decomposes $Q$ into a state value $V(s)$ and an action advantage $A(s,a)$, subtracting the mean to keep the decomposition identifiable. In many states actions hardly differ (an empty corridor, say), so learning $V$ separately is more efficient, and the advantage branch only needs to learn relative quality. The gain is clear when there are many actions and many are irrelevant.
**Follow-up**: Why subtract the mean rather than the max? Subtracting the mean makes $V$ the mean of $Q$ and keeps gradients stable; subtracting the max is also identifiable but trains less smoothly.

### Q7. How are PER priorities set? Why are importance weights needed?
**Answer**: $p_i\propto|\delta_i|^{\alpha}$ ($\alpha$ controls the strength of prioritization), with sampling probability $P(i)=p_i^{\alpha}/\sum_k p_k^{\alpha}$. Prioritized sampling changes the data distribution and biases Q-values, so an importance weight $w_i=(1/(N\cdot P(i)))^\beta$ corrects for it, annealing $\beta$ from 0 to 1.
**Follow-up**: What are PER's pitfalls? It is sensitive to noisy rewards and can overfit high-TD-error samples; implementations usually use a sum-tree.

### Q8. How do the forward and backward views of TD(λ) relate?
**Answer**: The forward view uses the λ-return $G_t^{\lambda}=(1-\lambda)\sum_n\lambda^{n-1}G_t^{(n)}$ as the target and needs multiple future steps. The backward view uses eligibility traces $E_t=\gamma\lambda E_{t-1}+\mathbb{1}(S_t=s)$ to distribute each TD error to recently visited states, enabling online updates. They are equivalent in expectation, and the backward view is the standard implementation form.
**Follow-up**: What does $\lambda$ do? $\lambda=0$ means high bias and low variance (TD), $\lambda=1$ means unbiased but high variance (MC), and intermediate values (often 0.9 to 0.95) trade off between them.

### Q9. Why can't DQN be used directly for continuous actions? What is the difference between DDPG/TD3/SAC?
**Answer**: DQN's target needs $\max_{a'}Q(s',a')$, which cannot be enumerated over continuous actions, and argmax itself is not differentiable. DDPG uses a deterministic actor that outputs actions directly plus a critic to estimate values, trained off-policy. TD3 fixes DDPG's overestimation (take the smaller of two Qs), delays updates, and smooths targets. SAC uses a maximum-entropy objective with a stochastic policy, explores better, and is more robust to hyperparameters, so it is usually the default choice.
**Follow-up**: How is SAC's $\alpha$ set? It can be fixed or tuned automatically to maintain a target entropy (Haarnoja et al., 2018b).

## 5. Pitfalls and follow-ups

- **The Q-table in Q-learning**: initialization affects exploration; optimistic initialization encourages exploration but can overestimate.
- **Do not backpropagate through the TD target**: the $V(s')$ / target net inside the target must be `detach`ed.
- **Handling `done`**: for terminal states the target must be just $r$, with no $\gamma\max Q(s',\cdot)$ added.
- **Reward scaling**: the reward scale affects the learning rate and Q-value scale; normalize before training.
- **Deadly triad**: function approximation + bootstrapping + off-policy is unstable together, and DQN's three components target exactly that.

## 6. Cheat sheet

| Algorithm | Update target | Policy type | One-liner |
| --- | --- | --- | --- |
| MC | $G_t$ | on | Unbiased, high variance, waits for the episode |
| TD(0) | $r+\gamma V(s')$ | on | Biased, low variance, online |
| Sarsa | $r+\gamma Q(s',a')$ | on | Prices in exploration risk |
| Q-learning | $r+\gamma\max_{a'}Q(s',a')$ | off | Learns the optimum, overestimates |
| DQN | $r+\gamma\max Q_{\theta^{-}}$ | off | Replay + target network |
| Double DQN | $r+\gamma Q_{\theta^{-}}(s',\arg\max Q_\theta)$ | off | Decouples selection and evaluation |
| Dueling | $V+A-\bar{A}$ | off | Value/advantage separation |
| PER | Sample by TD error | off | Hard samples first |
| DDPG | $r+\gamma Q(s',\mu(s'))$ | off | Deterministic actor |
| TD3 | clipped double-Q | off | Fixes DDPG overestimation |
| SAC | $r+\gamma(V-\alpha\log\pi)$ | off | Maximum entropy, automatic temperature |

Mnemonic: **"DP needs a model, MC waits for the end, TD takes one step; Q-learning takes the greedy max while Sarsa follows the policy; DQN adds replay and a target net, Double decouples, Dueling splits, PER picks the hard cases."**

## 7. References

- [Sutton & Barto, Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN · Mnih et al., 2015 (Nature)](https://www.nature.com/articles/nature14236)
- [Double DQN · van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)
- [Dueling DQN · Wang et al., 2016](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay · Schaul et al., 2016](https://arxiv.org/abs/1511.05952)
- [Rainbow · Hessel et al., 2018](https://arxiv.org/abs/1710.02298)
- [SAC · Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
- [TD3 · Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)
