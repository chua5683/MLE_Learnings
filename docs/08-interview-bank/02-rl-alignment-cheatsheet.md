# RL and Alignment Cheat Sheet

> **How to use**: 10-minute final sprint before a post-training, alignment, or reasoning-model interview. For full derivations, follow the links to [Chapter 3](../03-reinforcement-learning/README.md).

## 1. Classical RL backbone (must state cleanly)

| Concept | Formula / key point |
| --- | --- |
| Bellman expectation | $V^\pi(s)=\mathbb{E}_\pi[r + \gamma V^\pi(s')]$, $Q^\pi(s,a)=\mathbb{E}[r+\gamma V^\pi(s')]$ |
| Policy gradient | $\nabla_\theta J(\theta)=\mathbb{E}\big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat{A}_t\big]$ (baseline $b(s_t)$ lowers variance without bias) |
| GAE | $\hat{A}_t^{\mathrm{GAE}}=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}$, $\delta_t = r_t+\gamma V(s_{t+1})-V(s_t)$; $\lambda=0$ low var/high bias, $\lambda=1$ MC |
| PPO-clip | $\mathcal{L}^{\mathrm{CLIP}}=\mathbb{E}\big[\min(r_t\hat{A}_t,\;\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)\big]$, $r_t=\pi_\theta/\pi_{\mathrm{old}}$ |
| On-policy vs. Off-policy | PPO/GRPO/REINFORCE = on-policy (fresh rollouts); DQN/SAC/DPO = off-policy/offline |

## 2. LLM alignment & post-training comparison

| Algorithm | Models in memory | Online rollouts? | Reward source | Core strength | Main failure mode |
| --- | --- | --- | --- | --- | --- |
| **PPO (RLHF)** | 4 (actor, critic, RM, ref) | Yes | Learned RM (Bradley-Terry) | Token-level GAE; general preferences | Heavy VRAM; critic tuning; RM hacking |
| **DPO** | 2 (actor, ref) | No (offline pairs) | Implicit via $\beta\log(\pi/\pi_{\mathrm{ref}})$ | Simple, cheap, stable | Distribution shift; likelihood displacement |
| **SimPO / ORPO** | 1 (actor only) | No (offline pairs) | Length-normalized log-prob margin | No reference model; length-controlled | Sensitive to margin $\gamma$ & offline data quality |
| **KTO** | 2 (actor, ref) | No (unpaired +/-) | Prospect-theory utility | Works without paired rankings | Needs accurate positive/negative ratio |
| **GRPO** | 2–3 (actor, ref, verifier/RM) | Yes ($G$ per prompt) | Verifiable rules (RLVR) or RM | Drops critic; shines on math/code | Difficulty & length bias; entropy collapse |
| **Dr. GRPO / DAPO** | 2–3 | Yes | Verifiable rules / RM | Fixes std/length bias, clip-higher, dynamic sampling | Higher rollout cost from filtering zero-var groups |
| **RLOO / REINFORCE++** | 2–3 | Yes | Verifiable rules / RM | Unbiased leave-one-out baseline (RLOO); simple | Sequence-level credit assignment |
| **GSPO / CISPO** | 2–3 | Yes | Verifiable rules / RM | Sequence-level ratio (GSPO) / clipped IS (CISPO) stabilizes MoE & long CoT | Newer tuning recipes |

## 3. Essential formulas to recite

**Bradley-Terry Reward Model loss**:

$$
\mathcal{L}_{\mathrm{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big]
$$

**KL-regularized RLHF optimum** (the bridge from PPO to DPO):

$$
\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right) \Longrightarrow r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)
$$

**DPO loss** ($Z(x)$ cancels in pairwise difference):

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]
$$

**GRPO advantage & k3 KL estimator**:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)} \text{ (drop std in Dr. GRPO)}, \qquad \widehat{\mathrm{KL}}_{\mathrm{k3}} = \frac{\pi_{\mathrm{ref}}}{\pi_\theta} - \log\frac{\pi_{\mathrm{ref}}}{\pi_\theta} - 1 \ge 0
$$

**Evaluation metrics (`pass@k` vs `pass^k`)**:

$$
\mathrm{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \text{ (at least 1)}, \qquad \mathrm{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}} \approx \hat{p}^k \text{ (all } k \text{ succeed)}
$$

## 4. Six rapid-fire interview answers

1. **Why can GRPO drop the critic?** All $G$ answers to prompt $x$ share the same initial state, so the group mean is an unbiased Monte Carlo baseline of $V(x)$ — replacing the value network at the cost of $G\times$ rollout sampling and sequence-level granularity.
2. **Why does DPO sometimes lower *both* chosen and rejected log-probabilities?** Because the loss only depends on the *difference* $\log\pi(y_w)-\log\pi(y_l)$; both can fall as long as $y_l$ drops faster (**likelihood displacement**). Mitigate with an SFT/NLL anchor term or online/iterative DPO.
3. **Why did DAPO add Clip-Higher ($\epsilon_{\mathrm{high}} = 0.28 > \epsilon_{\mathrm{low}} = 0.20$)?** Symmetric clipping starves low-probability exploratory tokens from rising quickly, causing rapid **entropy collapse** during long reasoning RL.
4. **How do you align non-verifiable tasks (writing, instruction-following, safety) without RM hacking?** Use **rubric-based rewards** + **Generative Reward Models (GenRM)** (judge critiques before scoring), pair with KL constraints, refresh RM on new policy outputs, and keep held-out human evals.
5. **In multi-turn / agentic RL, what breaks if you compute loss on all tokens?** Including tool-output / environment tokens in the policy gradient trains the actor to predict tool errors instead of actions — always **mask non-assistant tokens** out of the loss.
6. **Does RLVR expand reasoning or only sharpen `pass@1`?** It reliably boosts `pass@1` by concentrating mass on effective reasoning/verification trajectories; whether `pass@k` (large $k$) rises depends on base-model coverage, training scale, and preventing entropy collapse.

## 5. Diagnostics dashboard (what to check when RL diverges)

| Symptom | Likely cause | First fix |
| --- | --- | --- |
| Reward ↑, held-out quality ↓ | Reward hacking (verbosity, format gaming) | Inspect samples, add rubric/length checks, raise KL $\beta$ |
| Entropy drops near 0 | Over-tight clipping or high LR | DAPO clip-higher, lower LR, entropy bonus, dynamic sampling |
| Zero gradient on many prompts | Too easy (all 1s) or too hard (all 0s) | Difficulty curriculum + dynamic sampling |
| Rollout vs. trainer KL spike | Weight-sync bug, FP8 mismatch, or re-tokenization | Pass raw token IDs directly; verify chat template & IS ratio |

## 6. Related chapters

- [04 PPO and TRPO](../03-reinforcement-learning/04-ppo-and-trpo.md) · [05 RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md) · [06 DPO](../03-reinforcement-learning/06-dpo-and-preference-optimization.md) · [07 GRPO & Reasoning RL](../03-reinforcement-learning/07-grpo-and-reasoning-rl.md) · [08 Alignment compared](../03-reinforcement-learning/08-alignment-algorithms-compared.md) · [04-07 Post-training pipeline](../04-llm/07-post-training-pipeline.md) · [07-05 RL Post-Training Infra](../07-ml-infra/05-rl-posttraining-infrastructure.md)
