# Normalization Techniques

> **One-line summary**: Normalization pulls each layer's input (or each sample's features) back to a stable mean and variance, letting deep networks use larger learning rates and become less sensitive to initialization; BN relies on batch statistics and suits CV, while LN/RMSNorm are batch-independent and therefore dominate Transformers.
>
> **Keywords**: `BatchNorm` `LayerNorm` `RMSNorm` `GroupNorm` `Pre-LN/Post-LN` `running statistics`

## 1. Core concepts

A direct cause of unstable deep-network training is that every parameter update changes the input distribution of the next layer, and the deeper the network the more that drift compounds (early papers called this internal covariate shift, ICS). The normalization layer's approach is simple: **before the nonlinearity, restandardize the activations to mean 0 and variance 1, then restore expressiveness with learnable scale and shift**.

Designing a normalization layer means answering four questions:

1. **On which axes are the statistics computed?** (batch axis / feature axis / channel groups / spatial axes)
2. **Whose statistics are used?** (current batch / running average / the sample itself)
3. **Is training consistent with inference?**
4. **On which side of the residual does it sit?** (Pre-LN / Post-LN)

## 2. Principles and derivations

### 2.1 BatchNorm

For a mini-batch $\mathcal{B}=\{x_1,\dots,x_m\}$ and a given channel (or feature dimension):

$$\mu_\mathcal{B}=\frac{1}{m}\sum_{i=1}^{m}x_i, \qquad \sigma_\mathcal{B}^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_\mathcal{B})^2$$

$$\hat{x}_i=\frac{x_i-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}, \qquad y_i=\gamma\,\hat{x}_i+\beta$$

$\gamma,\beta$ are learnable parameters that let the network "undo" normalization when needed. **Training** uses the current batch statistics while also updating global statistics with a running average:

$$\mu_{run}\leftarrow(1-\alpha)\mu_{run}+\alpha\,\mu_\mathcal{B}, \qquad \sigma^2_{run}\leftarrow(1-\alpha)\sigma^2_{run}+\alpha\,\sigma^2_\mathcal{B}$$

$\alpha$ is usually $0.1$. **Inference** uses $\mu_{run},\sigma^2_{run}$ and ignores the current input, so a single sample can be processed and the result is deterministic. This also means BN can be **folded into the preceding convolution/linear layer** at inference time: $y=\frac{\gamma(Wx+b-\mu)}{\sqrt{\sigma^2+\epsilon}}+\beta$ is equivalent to a new $W',b'$ with zero extra cost.

**Why it works** (a common follow-up; answer in layers):

- **The original explanation (ICS)**: it stabilizes each layer's input distribution and breaks the chain reaction where parameter updates shift the distribution. This explanation was later challenged - experiments showed BN still helps even when distribution shift is injected artificially.
- **The better-accepted explanation (loss landscape)**: BN makes the loss surface smoother (smaller Lipschitz constant) and gradients more predictable, which allows larger learning rates and more tolerance for initialization.
- **Regularization effect**: batch statistics add noise that acts like mild regularization and may improve generalization (which is why smaller batches mean more noise).

**Cost**: it depends on batch size and sample correlation; in sequence tasks different positions have different lengths, so batch statistics are meaningless; distributed training needs SyncBN to be equivalent to a large batch.

### 2.2 LayerNorm

Compute mean and variance over **a single sample's** feature dimensions (in a Transformer, over $d_{model}$):

$$\mu=\frac{1}{d}\sum_{j=1}^{d}x_j, \qquad \sigma^2=\frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2, \qquad y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

It is independent of the batch and behaves identically in training and inference, which makes it suitable for variable-length sequences, small batches, and autoregressive generation. For a Transformer, each token's vector is normalized independently, which naturally matches the view that "each position is a sample".

### 2.3 RMSNorm

A simplified LayerNorm: **do not subtract the mean, just divide by the root mean square**.

$$\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}, \qquad y=\gamma\odot\frac{x}{\mathrm{RMS}(x)}$$

It drops both the mean computation and the shift parameter $\beta$. In LLM training the FLOP share is small but it runs in every layer, so the savings add up; in practice RMSNorm matches LN, which is why models such as LLaMA and Qwen use it widely.

### 2.4 GroupNorm and InstanceNorm

- **InstanceNorm (IN)**: normalize each sample and each channel over the spatial dimensions $(H,W)$; common in style transfer.
- **GroupNorm (GN)**: split the $C$ channels into $G$ groups and normalize within each group ($C/G$ channels x spatial). It does not depend on the batch and is more stable than BN in **small-batch** settings such as detection and segmentation, making it a common BN replacement.

### 2.5 Pre-LN and Post-LN

There are two classic placements in a Transformer ($x$ is the residual stream, $F$ the sublayer):

$$\text{Post-LN:}\quad x_{l+1}=\mathrm{LN}\big(x_l+F(x_l)\big)$$

$$\text{Pre-LN:}\quad x_{l+1}=x_l+F\big(\mathrm{LN}(x_l)\big)$$

- **Post-LN** (the original Transformer): every layer's output is normalized, which keeps representations well-behaved, but backward gradients must pass through the LN Jacobian, making deep models unstable; learning-rate warmup is usually required to train.
- **Pre-LN** (mainstream since GPT-2): the residual stream is a **pure identity path**, so gradients flow from the last layer back to the first without attenuation and training is very stable with little dependence on warmup. The price is that the norm of the residual stream grows layer by layer, so a final LN is usually added, and at equal depth it may trail a well-tuned Post-LN.

### 2.6 Weight normalization (WeightNorm)

Instead of normalizing activations, reparameterize the weights:

$$w=\frac{g}{\lVert v\rVert}\,v$$

This decouples "direction" $v/\lVert v\rVert$ from "length" $g$ and makes the gradient insensitive to weight scale. It has been used in GANs and RNNs and is less common than BN/LN today, but the idea (decoupling scale from direction) is worth knowing.

## 3. Implementation notes

```python
import torch, torch.nn as nn

# BatchNorm: switching train/eval is critical
bn = nn.BatchNorm1d(128)
bn.train();  y = bn(torch.randn(32, 128))   # uses current batch stats and updates running stats
bn.eval();   y = bn(torch.randn(1, 128))    # uses running_mean/running_var, works on a single sample

# LayerNorm / RMSNorm
ln = nn.LayerNorm(128)                      # normalizes the last dimension
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # x: (..., d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

# Fold Conv+BN at inference time (common in deployment)
# W' = W * gamma / sqrt(var+eps);  b' = (b - mean) * gamma / sqrt(var+eps) + beta
```

## 4. Interview Q&A

### Q1. How does BN differ between training and inference, and why not use current batch statistics at inference?
**Answer**: Training uses the current mini-batch mean and variance while accumulating global statistics with a running average; inference uses the accumulated $\mu_{run},\sigma^2_{run}$. The reason is that inference may see a single sample (batch statistics are unstable or undefined) and the output must be deterministic and reproducible; using batch statistics would make the same image produce different outputs in different batches, which is unacceptable online.
**Follow-up**: What happens if you forget `model.eval()` at inference? BN keeps using current batch statistics and updates the running stats; with small batches the outputs jitter and metrics collapse - a very common bug.

### Q2. Why does BN work?
**Answer**: Three layers of explanation. (1) The original explanation is that it reduces internal covariate shift and stabilizes each layer's input distribution. (2) A stronger explanation is that BN makes the loss surface smoother with a smaller gradient Lipschitz constant, so larger learning rates are allowed and initialization matters less. (3) Batch statistics inject noise, giving a mild regularization effect. The ICS hypothesis itself is debated, so in an interview the more defensible version is "smoother optimization plus larger learning rates".
**Follow-up**: Why can a larger learning rate be used with BN? Because gradient scale is constrained by normalization, the effective step size becomes more predictable and one step is less likely to leave the stable region.

### Q3. Why do Transformers/NLP use LayerNorm instead of BatchNorm?
**Answer**: Three reasons. (1) Sequence lengths vary and padding is common; statistics over the batch axis would mix tokens at different positions and of different lengths, so the statistics have no clear meaning. (2) Autoregressive generation often runs with batch size 1, where BN cannot work. (3) LN normalizes each token independently, behaves identically in training and inference, needs no running stats, and is simpler to implement in a distributed setting.
**Follow-up**: Why is BN still mainstream in CV? Spatial statistics per channel within an image batch are stable, batches are usually large, BN works well, and it folds into convolutions for zero inference overhead.

### Q4. What is the essential difference between BatchNorm and LayerNorm?
**Answer**: The difference is **which axes are normalized**. BN computes statistics for the same channel across the batch (and spatial positions), i.e. "one feature over all samples"; LN computes statistics for all features of the same sample, i.e. "inside one sample". The former depends on batch composition and differs between training and inference; the latter is batch-independent and consistent.
**Follow-up**: Are the learnable parameters the same? Both have a scale $\gamma$; BN also has a shift $\beta$; LN generally has $\beta$ too (RMSNorm drops it).

### Q5. How does RMSNorm differ from LayerNorm, and why do LLMs use RMSNorm?
**Answer**: RMSNorm does not subtract the mean; it only divides by the root mean square $\sqrt{\mathrm{mean}(x^2)+\epsilon}$ and usually drops the shift parameter. It saves one mean computation and one subtraction, with a simpler and faster kernel and fewer memory accesses; empirically it matches or slightly beats LN (one view is that LN's re-centering is not essential and the scaling is what matters). Hence its wide adoption in the LLaMA family and elsewhere.
**Follow-up**: RMSNorm is insensitive to a mean shift in the input - does that lose information? In practice the Transformer residual stream is already close to zero-mean, so re-centering buys little; that is the empirical basis for dropping the mean.

### Q6. What is the difference between Pre-LN and Post-LN, and why is Pre-LN standard now?
**Answer**: Post-LN is $\mathrm{LN}(x+F(x))$, normalizing after the residual addition; Pre-LN is $x+F(\mathrm{LN}(x))$, normalizing before the sublayer input. Pre-LN's residual stream is a pure identity mapping, giving backward gradients a direct path, so deep training is stable and depends little on warmup; Post-LN keeps representations better-behaved but gradients must cross the LN Jacobian, so deep models diverge easily and need careful warmup. Large-model training is expensive and stability comes first, so Pre-LN (and its variants) became the default.
**Follow-up**: What does Pre-LN cost? The residual-stream norm grows with depth, so a final LN is needed; at equal depth the final quality can trail a well-tuned Post-LN, which is why compromises such as DeepNorm exist.

### Q7. What do $\gamma$ and $\beta$ do in BN, and can they be removed?
**Answer**: Normalization forces activations to mean 0 and variance 1, which limits expressiveness (sigmoid would stay in its near-linear region forever). $\gamma,\beta$ provide a learnable scale and shift so the network can decide the distribution range each layer needs; in principle they can restore the identity mapping ($\gamma=\sigma,\beta=\mu$). Removing them usually hurts, especially in deep networks.
**Follow-up**: Is the bias of the layer before BN still useful? No: subtracting the mean cancels it, so Conv+BN commonly sets `bias=False`, saving parameters and memory.

### Q8. Is BN sensitive to batch size, and what do you do for small batches?
**Answer**: Yes. Small batches make the mean and variance estimates noisy, so training jitters and quality drops; in the extreme case of batch=1 the variance is 0. Remedies: SyncBN (synchronize statistics across GPUs, equivalent to a large batch), batch-independent normalizations such as GroupNorm/LayerNorm, gradient accumulation (increases the effective batch but does not improve BN statistics - an important distinction), or freezing BN to pretrained statistics during fine-tuning.
**Follow-up**: Can gradient accumulation make BN equivalent to a large batch? No. BN statistics are still computed per micro-batch unless you use SyncBN or tricks such as GhostBN.

### Q9. Should normalization go before or after the activation?
**Answer**: The classic CV order is Conv → BN → ReLU, i.e. normalization before the activation so the nonlinearity operates on standardized inputs; putting BN after ReLU would destroy ReLU's sparsity. In Transformers, LN sits **before** the sublayer (attention/FFN) in Pre-LN, normalizing the sublayer input. Conclusion: there is no universally correct order, but "normalization before the nonlinearity, inside the residual branch" is a common and stable choice.
**Follow-up**: Why is BN better before the activation? Because ReLU outputs are non-negative and skewed, and normalizing them would wash out the sparse structure, whereas linear-layer outputs are approximately symmetric and better suited to standardization.

### Q10. How do BN, dropout, initialization, and learning rate relate?
**Answer**: All of them control training stability, but at different points. BN stabilizes each layer's input distribution, reducing sensitivity to initialization scale and allowing larger learning rates; dropout is a regularization tool against overfitting; initialization sets the starting variance. Note that BN and dropout can interact (dropout noise affects BN's variance estimate), so some architectures replace BN with dropout or reorder them.
**Follow-up**: Is weight decay still needed with BN? Yes. BN does not constrain weight norms, and weight decay still controls model complexity.

### Q11. Why can BN be "folded" into a convolution at inference, and how?
**Answer**: At inference BN is an affine transform $y=\gamma\frac{Wx+b-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, which can be merged into a new convolution weight $W'=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W$ and bias $b'=\frac{\gamma(b-\mu)}{\sqrt{\sigma^2+\epsilon}}+\beta$. After folding there is no BN operator in the inference graph, saving one memory pass and some compute - a standard deployment optimization.
**Follow-up**: Can it be folded during training? No: training statistics change with each batch and must be computed online; only inference with fixed running stats can be folded.

## 5. Pitfalls and follow-ups

- **Mixing up training and inference modes**: BN and dropout both require an explicit `train()/eval()` switch.
- **Forcing BN onto small batches**: statistics are noisy and GN/LN may work better.
- **Assuming LN has running stats**: it does not; training and inference are identical.
- **Forgetting the final LN with Pre-LN**: the residual-stream norm accumulates, so normalize once more before the output layer.
- **Treating BN as the main regularizer**: its regularization effect is weak and cannot replace weight decay, dropout, or data augmentation.

## 6. Cheat sheet

| Method | Statistics axes | Batch-dependent | Train/infer consistent | Typical use |
| --- | --- | --- | --- | --- |
| BatchNorm | Across samples, same channel | Yes | No (running stats) | CNNs, large batches |
| LayerNorm | Single sample, all features | No | Yes | Transformers, RNNs |
| RMSNorm | Single sample, root mean square | No | Yes | LLaMA/Qwen and other LLMs |
| InstanceNorm | Single sample, single-channel spatial | No | Yes | Style transfer |
| GroupNorm | Single sample, channel groups | No | Yes | Small-batch detection/segmentation |
| WeightNorm | Weight direction/norm | No | Yes | GANs, some RNNs |

| Placement | Formula | Characteristics |
| --- | --- | --- |
| Post-LN | $\mathrm{LN}(x+F(x))$ | Original Transformer, needs warmup, hard to train deep |
| Pre-LN | $x+F(\mathrm{LN}(x))$ | Direct gradient path, stable, default for large models |

**Mnemonic**: CV uses BN (large batch), NLP uses LN, LLMs use RMSNorm; Pre-LN inside the residual, stability first.

## 7. References

- [Batch Normalization paper](https://arxiv.org/abs/1502.03167)
- [Layer Normalization paper](https://arxiv.org/abs/1607.06450)
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [llm_interview_note · layer normalization](https://github.com/wdndev/llm_interview_note)
