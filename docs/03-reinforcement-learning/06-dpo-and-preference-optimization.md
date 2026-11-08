# DPO and Preference Optimization

> **One-line summary**: DPO solves the KL-regularized reward maximization for a closed-form optimal policy, substitutes that solution back into Bradley-Terry, and obtains a classification loss that only needs preference pairs, with no reward model and no online sampling; it replaces RL with classification while inheriting the distribution shift and length bias of offline methods.
>
> **Keywords**: `DPO` `implicit reward` `Bradley-Terry` `IPO` `KTO` `ORPO` `SimPO` `length bias` `offline preference optimization`

## 1. Core concepts

The pain point of RLHF (Chapter 5) is that you first train an RM, then use PPO with online sampling: 4 models, unstable training, and a complex implementation. DPO (Direct Preference Optimization) asks:

> Since the KL-regularized objective has a **closed-form optimal policy**, can we eliminate "reward model + RL" entirely and optimize the policy directly from preference pairs?

Yes. DPO takes the same input as the second stage of RLHF (preference pairs $(x,y_w,y_l)$) and outputs a directly trainable policy, with **no RM and no sampling during training**.

| Dimension | RLHF (PPO) | DPO |
| --- | --- | --- |
| Models needed | actor + critic + reward + ref | policy + ref |
| Sampling | Online during training | Offline preference data |
| Loss | Policy gradient + clip | Binary cross-entropy (sigmoid) |
| Reward model | Trained explicitly | Implicit (from the policy/ref log-ratio) |
| Stability | Sensitive to hyperparameters | Relatively stable |
| Reusable artifact | A reusable RM | Only the policy |

## 2. Principles and derivations

### 2.1 From the KL-regularized objective to a closed-form optimal policy

The RLHF objective (same as Eq. (4) in Chapter 5):

$$
\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},y\sim\pi(\cdot\mid x)}\left[r(x,y)\right]-\beta\,\mathrm{KL}\left(\pi(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\right) \qquad (1)
$$

Maximize pointwise for each $x$. Expand the expectation into a sum, add the constraint $\sum_y\pi(y\mid x)=1$, and form the Lagrangian:

$$
\mathcal{L}=\sum_y\pi(y\mid x)r(x,y)-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\lambda\left(1-\sum_y\pi(y\mid x)\right) \qquad (2)
$$

Differentiate with respect to $\pi(y\mid x)$ and set it to zero:

$$
r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}-\beta-\lambda=0
\quad\Longrightarrow\quad
\pi^{*}(y\mid x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right) \qquad (3)
$$

where $Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)\exp\left(r(x,y)/\beta\right)$ is the normalization constant (partition function). **Eq. (3) is the entire starting point of DPO**: the optimal policy reweights the reference policy by $\exp(r/\beta)$.

### 2.2 Invert the reward and substitute into Bradley-Terry

Take the log of Eq. (3) and rearrange:

$$
r(x,y)=\beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x) \qquad (4)
$$

Define the **implicit reward** $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$. Substitute it into Bradley-Terry (Eq. (1) in Chapter 5):

$$
P(y_w\succ y_l\mid x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
=\sigma\left(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)\right) \qquad (5)
$$

**The key**: $\beta\log Z(x)$ is the same constant for $y_w$ and $y_l$, so it cancels in the difference; there is no need to know $Z(x)$, and therefore no need for an explicit reward model.

### 2.3 The DPO loss

Maximize the likelihood of preference data (minimize the negative log-likelihood) to obtain:

$$
\boxed{\;\mathcal{L}_{\mathrm{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]\;} \qquad (6)
$$

Writing $z=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$, we have $\mathcal{L}=-\log\sigma(z)$, whose gradient is

$$
\nabla_\theta\mathcal{L}_{\mathrm{DPO}}=-\beta\,\sigma(-z)\left(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\right) \qquad (7)
$$

Term by term:

- $\sigma(-z)$ is an **adaptive weight**: when the model already ranks $y_w$ above $y_l$ ($z$ large), the weight is small and the sample barely updates; when the ranking is wrong, the weight is large and the sample is corrected.
- The gradient direction is to **raise the log-prob of $y_w$ and lower that of $y_l$**, normalized by the reference log-ratio so the absolute probabilities are not pushed too high or too low.
- No sampling, no RM, no value function; one forward pass computes the loss, so training is much cheaper than PPO.

### 2.4 The role of $\beta$

$\beta$ appears in both the objective and the loss, and its role is to **control how far the policy may deviate from the reference model**:

- Large $\beta$ pulls the optimal policy toward $\pi_{\mathrm{ref}}$ and makes updates conservative.
- Small $\beta$ pursues preferences more aggressively and easily overfits the preference data, degrading language quality.
- Common values are 0.1~0.5 (conventions differ; some implementations put $\beta$ in the denominator).

Viewed through Eq. (3), $\beta$ is a "temperature": as $\beta\to0$ the optimal policy puts all probability on the highest-reward answer (mode collapse); as $\beta\to\infty$ it reduces to $\pi_{\mathrm{ref}}$.

### 2.5 DPO variants at a glance

| Method | Core change | Needs ref? | Problem it solves |
| --- | --- | --- | --- |
| DPO | The classification loss of Eq. (6) | Yes | Baseline |
| IPO | Regress the log-ratio difference to a fixed margin $1/(2\beta)$ with a squared loss | Yes | Prevents overfitting to deterministic preferences (DPO keeps pushing as $z\to\infty$) |
| KTO | Uses only binary "good/bad" labels, no pairs; introduces asymmetric prospect-theory weights | Yes | Pairwise data is hard to collect; single feedback still works |
| ORPO | Odds-ratio loss + SFT NLL trained jointly, **no ref** | No | Removes the reference model and does SFT + alignment in one step |
| SimPO | Uses length-normalized average log-prob as the implicit reward with a margin $\gamma$, **no ref** | No | Saves memory and mitigates length bias |
| cDPO / robust DPO | Probabilistic correction for label noise (label smoothing) | Yes | More robust when human labels are wrong |

A few key formulas (the idea is enough):

- **IPO**: $\min\left(\log\frac{\pi_\theta(y_w)\pi_{\mathrm{ref}}(y_l)}{\pi_\theta(y_l)\pi_{\mathrm{ref}}(y_w)}-\frac{1}{2\beta}\right)^2$, fixing the preference margin to avoid DPO's "ever more confident" behavior.
- **ORPO**: $\mathcal{L}=\mathcal{L}_{\mathrm{SFT}}+\lambda\,\mathcal{L}_{\mathrm{OR}}$, where $\mathcal{L}_{\mathrm{OR}}=-\log\sigma\left(\log\frac{\mathrm{odds}_\theta(y_w)}{\mathrm{odds}_\theta(y_l)}\right)$ and $\mathrm{odds}_\theta(y)=\frac{\pi_\theta(y)}{1-\pi_\theta(y)}$.
- **SimPO**: $\mathcal{L}=-\log\sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l)-\gamma\right)$, no ref, averaged over length.

### 2.6 Length bias and the limits of DPO

**Length bias**: the implicit reward is $\beta\sum_t\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$, summed over tokens. Longer answers have more log-prob difference terms and are easier to push up, so after DPO training answers generally get longer, a very consistent empirical phenomenon. Mitigations: length normalization (SimPO), an explicit length penalty (R-DPO subtracts $\alpha(|y_w|-|y_l|)$), and controlling the length distribution in the data.

**Other limitations**:
- **Offline**: only fixed preference data, no online exploration; when the data distribution no longer matches the current policy, the "preference" DPO learns can become stale (distribution shift).
- **Depends on ref**: if $\pi_{\mathrm{ref}}$ and the sampling policy of the preference data differ (e.g., using a public dataset rather than your own samples), the implicit reward is biased; in practice the chosen responses are used for a short SFT to initialize ref.
- **Over-optimization**: training too long lowers the absolute probabilities of both chosen and rejected answers (only the relative gap is constrained), which can degrade language quality.
- **Cannot use non-pairwise/process rewards**: it only consumes $(y_w,y_l)$; RLVR and tool-call outcomes require PPO/GRPO.

## 3. Implementation notes

The DPO loss (including log-prob computation, about 20 lines):

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy, ref, batch, beta=0.1):
    # batch: input_ids [B, T], labels [B, T] (prompt positions are -100)
    logp_w = policy.log_prob(batch.chosen).sum(-1)      # [B]
    logp_l = policy.log_prob(batch.rejected).sum(-1)
    with torch.no_grad():
        ref_w = ref.log_prob(batch.chosen).sum(-1)
        ref_l = ref.log_prob(batch.rejected).sum(-1)
    z = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    loss = -F.logsigmoid(z).mean()
    # monitor: implicit reward gap, chosen/rejected log-probs, lengths
    return loss, (logp_w - ref_w).mean().detach()
```

Implementation notes:
- **Freeze the ref model** and use it for forward only; 4-bit quantization or offloading saves memory.
- Sum log-probs only over response tokens (mask the prompt).
- For `log_prob` use `log_softmax` + `gather`; do not use the cross-entropy mean (the reward here is a sum).
- Data mix: balance the length and quality of chosen/rejected; preference data should ideally come from samples of the current SFT model.
- Monitor whether the implicit reward gap diverges and whether the absolute log-probs of chosen/rejected fall together.

## 4. Interview Q&A

### Q1. Derive DPO: from the KL-regularized objective to a classification loss.
**Answer**: The objective $\max_\pi\mathbb{E}[r]-\beta\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$ is optimized pointwise for each $x$; the Lagrangian gives $\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}$. Inverting gives $r=\beta\log(\pi^*/\pi_{\mathrm{ref}})+\beta\log Z$. Substituting into Bradley-Terry cancels $Z$ in the difference and yields $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)})$, the DPO loss.
**Follow-up**: Which step matters most? $Z(x)$ depends only on $x$, not on $y$, so it cancels in $r_w-r_l$; this is the fundamental reason no reward model is needed.

### Q2. Why does DPO need no reward model? What is the implicit reward?
**Answer**: Because the optimal policy and the reward are invertible (Eq. 3), the reward can be written as the policy's log-ratio against the reference: $\hat r_\theta=\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$. DPO treats this as the reward and trains the policy with the BT likelihood of preference pairs; after training, $\hat r_\theta$ is the learned reward, but it is tied to the policy and cannot be used to score other models independently.
**Follow-up**: Can a DPO model be used as an RM? You can approximately rank with $\hat r_\theta$, but it is not independently calibrated and cross-model comparisons are unreliable.

### Q3. How should the DPO gradient be understood?
**Answer**: $\nabla\mathcal{L}=-\beta\sigma(-z)(\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l))$. $\sigma(-z)$ is a per-sample weight: already-correctly-ranked samples get small weight and misranked ones get large weight; the direction raises the chosen log-prob and lowers the rejected one, constrained by the reference log-ratio. DPO thus behaves like contrastive learning with adaptive weights.
**Follow-up**: Why is it said to do RL implicitly? Because the optimal solution in Eq. (3) guarantees that maximizing the implicit reward under the KL constraint is equivalent to maximizing the true reward; DPO fits that optimal policy.

### Q4. How does DPO differ from PPO? When is each appropriate?
**Answer**: PPO samples online, needs RM + critic, uses 4 models, and can explore; DPO is offline, needs only policy + ref, uses 2 models, and is stable and cheap. Choose DPO when the data is fixed, preferences are clear, and the budget is limited; choose PPO/GRPO when the RM must be reused, online exploration matters, or the reward is not a pairwise preference (safety rules, tool outcomes). Many teams iterate quickly with DPO and then refine with online RL.
**Follow-up**: Is DPO always worse than PPO? No; DPO often matches or beats PPO on single-turn dialogue and summarization, but PPO/GRPO have a higher ceiling on tasks that need exploration and long-horizon rewards.

### Q5. What does $\beta$ do in DPO? What happens if it is too large or too small?
**Answer**: $\beta$ controls how far the policy may drift from the reference and is the temperature in Eq. (3). Large $\beta$ keeps the optimal policy close to $\pi_{\mathrm{ref}}$: conservative updates, slower learning. Small $\beta$ is more aggressive and prone to overfitting preferences, degrading language quality or even causing mode collapse. Around 0.1 is common (implementation conventions differ); tune it with data size and ref quality.
**Follow-up**: Is $\beta$ the same as the KL coefficient? It plays the same role: a penalty coefficient in the KL-regularized objective and the scaling of the implicit reward in DPO.

### Q6. Why do DPO answers get longer after training? How do you fix it?
**Answer**: The implicit reward is a sum of per-token log-ratios, so longer answers have more terms and the model can amplify the reward gap by writing more; moreover, human preference data often favors more detailed answers, correlating length with "good." Mitigations: length normalization (SimPO uses average log-prob), explicit length penalties (R-DPO), length matching/debiasing in the data, and monitoring length during training.
**Follow-up**: Is longer always bad? Not necessarily, but "longer for the sake of longer" with falling information density is a sign of reward hacking.

### Q7. What problems do IPO/KTO/ORPO/SimPO each solve?
**Answer**: IPO uses a squared loss to fix the preference margin, preventing DPO from overfitting deterministic preferences; KTO relaxes pairwise comparison into single "good/bad" binary feedback, suiting data that is hard to pair; ORPO combines SFT and preference alignment in one step and removes the ref model, saving memory; SimPO uses a length-normalized average log-prob plus a margin, needs no ref, and mitigates length bias. They are different patches for DPO's "offline, needs ref, prone to length bias" issues.
**Follow-up**: Can these variants fully replace DPO? There is no universal winner; it depends on data shape (paired/single), memory budget, and task, and paper conclusions often vary with the setup.

### Q8. What are DPO's limitations? When does it fail?
**Answer**: (1) Offline distribution shift: preference data comes from an old policy, so after training it no longer represents current preferences. (2) It only learns relative ranking and can lower the absolute probabilities of both chosen and rejected, hurting language modeling. (3) Length/format bias. (4) It cannot use process rewards or non-pairwise signals. (5) It is sensitive to the ref model and $\beta$. (6) Noise in preference data (label errors) is amplified. It fails when exploration is needed, when multi-turn/tool feedback is needed, or when the reward is verifiable but not a preference.
**Follow-up**: How do you mitigate the offline problem? Iterative DPO (sample new preference pairs with the current policy), on-policy DPO, or go straight to online RL.

### Q9. Why can the probabilities of both chosen and rejected drop in DPO? Is that a bug?
**Answer**: No. The DPO loss constrains only the **difference** $z$, not absolute probabilities; if $\pi_\theta$ drifts as a whole from $\pi_{\mathrm{ref}}$ (for example, ref and the data distribution disagree), the gradient can lower both while keeping the difference. As long as the difference grows, preference ranking is learned correctly; but if absolute probabilities fall too much, generation quality degrades, so monitor it and pair with an SFT loss, a lower lr, or a better ref.
**Follow-up**: How do you avoid it? Initialize ref as the SFT model from the same distribution as the preference data, limit training steps, and add NLL regularization (the ORPO idea).

### Q10. How can DPO be made "online"?
**Answer**: Iterative DPO: sample several answers with the current policy, build new preference pairs scored by an RM/humans/rules, and run another DPO round; or on-policy DPO resamples before each update round. This mitigates distribution shift but reintroduces sampling and labeling cost, essentially moving toward the PPO online paradigm. Another direction combines verifiable rewards (RLVR) with the preference loss.
**Follow-up**: How does online DPO differ from PPO? Online DPO still lacks an explicit value/critic and importance sampling, so its exploration is weaker than PPO/GRPO.

## 5. Pitfalls and follow-ups

- **$Z(x)$ cannot be dropped**: it appears in Eq. (4) and only happens to cancel in the BT difference; when computing the implicit reward alone it is not negligible (though implementations usually do not care).
- **Where $\beta$ sits**: different papers/code put $\beta$ in the numerator or denominator; align the convention before reading formulas.
- **Use sum, not mean, of log-probs**: the DPO reward is the sum of log-ratios for the whole sequence.
- **ref must be frozen** and ideally from the same distribution as the data.
- **Data leakage**: preference data must not overlap the evaluation set.
- **Loss alone is not enough**: monitor the implicit reward gap, length, absolute log-probs, and human evaluation together.

## 6. Cheat sheet

| Concept | Formula / key point |
| --- | --- |
| Optimal policy | $\pi^*=\frac{1}{Z}\pi_{\mathrm{ref}}e^{r/\beta}$ |
| Implicit reward | $\hat r_\theta=\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$ |
| DPO loss | $-\log\sigma(\hat r_w-\hat r_l)$ |
| Gradient weight | $\sigma(-z)$, larger for misranked samples |
| $\beta$ | How tightly to stay near ref (temperature) |
| Variants | IPO (fixed margin), KTO (binary), ORPO (no ref), SimPO (no ref + length normalization) |
| Main issues | Offline shift, length bias, falling absolute probabilities |

Mnemonic: **"KL regularization gives a closed-form policy; invert the reward and plug it into BT; Z cancels and the RM disappears; raise chosen, lower rejected, and σ(-z) does the weighting."**

## 7. References

- [DPO · Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- [IPO · Azar et al., 2023](https://arxiv.org/abs/2310.12036)
- [KTO · Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)
- [ORPO · Hong et al., 2024](https://arxiv.org/abs/2403.07691)
- [SimPO · Meng et al., 2024](https://arxiv.org/abs/2405.14734)
- [Is DPO Superior to PPO? · Xu et al., 2024](https://arxiv.org/abs/2404.10719)
- [llm_interview_note · DPO](https://github.com/wdndev/llm_interview_note)
