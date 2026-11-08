# Activation Functions and Initialization

> **One-line summary**: Activation functions decide whether a signal can pass through the nonlinearity without loss, and initialization decides whether that signal keeps its variance across tens of layers; together they determine whether the network trains at all - most vanishing/exploding-gradient problems trace back to these two choices.
>
> **Keywords**: `sigmoid` `ReLU` `GELU` `SwiGLU` `dying ReLU` `Xavier` `He/Kaiming` `gradient checking`

## 1. Core concepts

**Activation function**: the nonlinear mapping applied after a neuron's linear transform. Practical requirements: nonlinear, as non-saturating as possible (derivative not approaching 0), cheap to compute, and with a simple derivative.

**Saturation**: when the input is very large or very small, the output enters a flat region and the derivative approaches 0. sigmoid/tanh saturate on both sides; ReLU saturates "hard" only on the negative half (its derivative is exactly 0).

**Two goals of initialization**:

1. **Break symmetry**: different neurons must start with different weights, otherwise they will always learn the same thing.
2. **Preserve variance**: keep the variance of activations and gradients roughly constant across layers, avoiding exponential growth or decay with depth.

## 2. Principles and derivations

### 2.1 Common activation functions

**Sigmoid**

$$\sigma(z)=\frac{1}{1+e^{-z}}, \qquad \sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)\in\left(0,\tfrac14\right]$$

The output lies in $(0,1)$ and can be read as a probability, but it is not zero-mean (which biases the gradients of later layers) and its derivative peaks at only $0.25$.

**Tanh**

$$\tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}, \qquad \tanh'(z)=1-\tanh^2(z)\in(0,1]$$

Zero-mean and steeper than sigmoid, but it still saturates on both ends.

**ReLU family**

$$\mathrm{ReLU}(z)=\max(0,z), \qquad \mathrm{ReLU}'(z)=\begin{cases}1,&z>0\\0,&z<0\end{cases}$$

$$\mathrm{LeakyReLU}(z)=\max(\alpha z,z), \qquad \mathrm{LeakyReLU}'(z)=\begin{cases}1,&z>0\\\alpha,&z<0\end{cases}\ (\alpha\ \text{around}\ 0.01)$$

ReLU has derivative exactly 1 on the positive half, so it never saturates and is extremely cheap; the price is zero gradient on the negative half (**dying ReLU**) and a non-zero-mean output.

**GELU (Gaussian Error Linear Unit)**

$$\mathrm{GELU}(z)=z\,\Phi(z)\approx 0.5z\left(1+\tanh\left[\sqrt{\tfrac{2}{\pi}}\left(z+0.044715z^3\right)\right]\right)$$

where $\Phi$ is the standard normal CDF. It acts as a soft gate based on the input magnitude, is smooth everywhere, and is used by models such as BERT and GPT-2.

**SwiGLU (gated linear unit)**

$$\mathrm{SwiGLU}(x)=\mathrm{Swish}(xW_1)\odot(xV), \qquad \mathrm{Swish}(z)=z\,\sigma(z)$$

It turns the FFN from "two linear layers plus an activation" into "three projections plus a multiplicative gate", as used by LLaMA and others.

### 2.2 Deriving initialization variance

Consider a linear layer $z_i=\sum_{j=1}^{n_{in}}w_{ij}x_j$ (ignoring the bias) and assume $w_{ij}$ are i.i.d. with mean 0 and variance $\sigma_w^2$, while $x_j$ has mean 0, variance $\sigma_x^2$, and is independent of $w$:

$$\mathrm{Var}(z_i)=\sum_{j=1}^{n_{in}}\mathrm{Var}(w_{ij}x_j)=n_{in}\,\sigma_w^2\,\sigma_x^2$$

For $\mathrm{Var}(z_i)=\sigma_x^2$ (variance unchanged across layers) we need

$$\sigma_w^2=\frac{1}{n_{in}}$$

**The backward pass is symmetric**: gradients travel from layer $l$ back to $l-1$ multiplied by $W^T$, so keeping their variance unchanged requires $\sigma_w^2=1/n_{out}$.

**Xavier/Glorot**: balances forward and backward, taking the harmonic mean of the two:

$$\sigma_w^2=\frac{2}{n_{in}+n_{out}}$$

For a uniform distribution $w\sim U(-a,a)$ we have $\sigma_w^2=a^2/3$, hence $a=\sqrt{6/(n_{in}+n_{out})}$. This suits zero-mean, nearly linear activations such as tanh/sigmoid.

**He/Kaiming**: ReLU zeros the entire negative half. If $z$ is zero-mean and symmetric, then $\mathbb{E}[\mathrm{ReLU}(z)^2]=\tfrac12\mathrm{Var}(z)$, so the variance is halved. Compensating by a factor of 2 requires

$$\sigma_w^2=\frac{2}{n_{in}}$$

This is He initialization (Kaiming normal), the default choice for ReLU networks.

**Intuition**: if $\sigma_w^2$ is slightly above $1/n_{in}$, the variance after 30 layers grows by $(n\sigma_w^2)^{30}$ and explodes; slightly below, it decays exponentially to 0. Initialization is not "random for the sake of random" - it is matching the statistics of the activation function.

### 2.3 Gradient checking

Approximate the numerical gradient with a central difference and compare it with your own backward pass:

$$\frac{\partial L}{\partial\theta_i}\approx\frac{L(\theta_i+\epsilon)-L(\theta_i-\epsilon)}{2\epsilon}$$

A relative error $\frac{|g_{num}-g_{ana}|}{\max(1,|g_{num}|+|g_{ana}|)}$ below roughly $10^{-6}$ usually means the implementation is correct. Note: you must use **float64**, a small network, and $\epsilon\approx10^{-6}$; otherwise float32 rounding error drowns out the difference.

## 3. Implementation notes

```python
import torch, torch.nn as nn

# 1) call an activation directly
act = nn.GELU(approximate="tanh")   # matches the tanh approximation above

# 2) SwiGLU FFN (LLaMA style): note hidden is usually 8/3*d, rounded to a nice size
class SwiGLUFFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)  # gate branch
        self.v  = nn.Linear(d, hidden, bias=False)  # value branch
        self.w2 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.v(x))

# 3) initialization: PyTorch's default Linear uses U(-1/sqrt(fan_in), 1/sqrt(fan_in))
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)          # tanh/sigmoid networks
        # nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # ReLU networks
        nn.init.zeros_(m.bias)

# 4) gradient check (always double precision + small inputs)
from torch.autograd import gradcheck
layer = nn.Linear(4, 3).double()
x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
print(gradcheck(layer, (x,), eps=1e-6, atol=1e-4))   # True means it passed
```

## 4. Interview Q&A

### Q1. Why does sigmoid easily cause vanishing gradients?
**Answer**: The maximum of $\sigma'(z)=\sigma(z)(1-\sigma(z))$ is only $0.25$, so every backward layer multiplies by a factor of at most $0.25$ and the gradient decays exponentially across layers. On top of that, sigmoid outputs are always positive and not zero-mean, which biases the gradient direction of later weights and slows convergence.
**Follow-up**: Why is tanh a bit better than sigmoid? tanh is zero-mean and its derivative peaks at 1, so it decays more slowly, but it still saturates on both sides and deep vanishing remains, which is why the ReLU family replaced it.

### Q2. What advantages does ReLU have over sigmoid/tanh?
**Answer**: Its derivative is exactly 1 on the positive half, so it does not saturate and deep gradients are not attenuated by the activation; it only needs a comparison and a multiply, making it very fast; and it outputs 0 on the negative half, yielding sparse activations. The downsides are zero gradient on the negative half (dying ReLU), a non-zero-mean output, and a non-differentiable point (subgradients are fine in practice).
**Follow-up**: What does ReLU's non-zero mean affect? It shifts the input distribution of the next layer; in practice this is mitigated by BN/LN or zero-mean initialization.

### Q3. What is dying ReLU, and how do you mitigate it?
**Answer**: A neuron whose weights have moved so that every training sample falls on the negative half outputs 0 and has gradient 0 forever, effectively dying permanently. Common causes: too large a learning rate pushing weights very negative, too large an initialization variance, or bad samples. Mitigations: a smaller learning rate, He initialization, LeakyReLU/ELU/GELU to keep a small negative-side gradient, and BN to stabilize the input distribution.
**Follow-up**: Is dying ReLU data-dependent? Yes: it is more likely when many samples activate negatively in some pattern, which is one reason GELU/SwiGLU have become more popular in LLMs.

### Q4. Why does initialization need to control variance?
**Answer**: In the forward pass $\mathrm{Var}(z^{(l)})=n_{in}\sigma_w^2\mathrm{Var}(a^{(l-1)})$, and in the backward pass the gradient variance is multiplied by $n_{out}\sigma_w^2$. If $n\sigma_w^2\neq1$, variance grows or shrinks exponentially with depth, and training either explodes (NaN) or stalls. Controlling variance means keeping the signal "conserved" across tens of layers.
**Follow-up**: Do we still need good initialization with BN/LN? Yes. Normalization standardizes each layer's input but does not guarantee the magnitude of backward gradients on the residual branch; poor initialization can still destabilize training.

### Q5. Derive Xavier initialization.
**Answer**: Preserving forward variance requires $n_{in}\sigma_w^2=1$; preserving backward gradient variance requires $n_{out}\sigma_w^2=1$. Both cannot hold at once unless $n_{in}=n_{out}$, so we take the compromise $\sigma_w^2=2/(n_{in}+n_{out})$, which corresponds to $U(-\sqrt{6/(n_{in}+n_{out})},\sqrt{6/(n_{in}+n_{out})})$, whose variance is exactly $2/(n_{in}+n_{out})$.
**Follow-up**: Is Xavier suitable for ReLU? No. It assumes a nearly linear, zero-mean activation; ReLU halves the variance, so Xavier would decay layer by layer - use He instead.

### Q6. Why is He initialization $2/n_{in}$?
**Answer**: ReLU zeros the negative half; if the input is zero-mean and symmetric, then $\mathbb{E}[\mathrm{ReLU}(z)^2]=\tfrac12\mathrm{Var}(z)$, so each layer halves the variance. To compensate, the weight variance must be $2/n_{in}$ instead of $1/n_{in}$, which preserves both the forward activation variance and the backward gradient variance.
**Follow-up**: Does He initialization only work for ReLU? LeakyReLU also works (it keeps a small negative slope) with a slightly different coefficient; GELU is fine with the approximation.

### Q7. Why can't we initialize everything to zero? What about all to the same constant?
**Answer**: All-zero or all-identical weights make every neuron in a layer produce the same forward output and receive the same backward gradient, and they stay identical after the update, equivalent to a single neuron and destroying feature diversity. Random initialization is required. Biases can be zero because they do not participate in symmetry breaking across the input dimension.
**Follow-up**: Can weights be initialized to a very small random value? Too small and the signal decays layer by layer (vanishing gradients); too large and it explodes. Match the variance to fan_in and the activation.

### Q8. How does GELU differ from ReLU, and why do LLMs prefer GELU?
**Answer**: ReLU is a hard gate ($z>0$ passes fully, otherwise blocked); GELU is a soft gate $z\Phi(z)$ that smoothly decides how much to keep based on the input magnitude, is differentiable everywhere, keeps a non-zero gradient on the negative half, and avoids dead neurons. GELU/GLU variants in LLMs tend to beat ReLU slightly on language modeling and train more smoothly.
**Follow-up**: Is GELU more expensive? The exact version needs erf; in practice the tanh approximation is used and the overhead is acceptable.

### Q9. What is SwiGLU? Why use it in the FFN? How do you adjust the parameter count?
**Answer**: SwiGLU uses two projections $xW_1$ and $xV$, applies Swish to one branch, multiplies it elementwise with the other, and projects back with $W_2$. Compared with "linear-activation-linear" it introduces a multiplicative gate, which strengthens expressiveness and works better in models such as LLaMA. The cost is one extra projection matrix, so the FFN hidden size is usually reduced from $4d$ to about $\tfrac{8}{3}d$ to keep the total parameter count on par with a standard FFN.
**Follow-up**: Why 8/3 instead of 4? A standard FFN has two $d\times4d$ matrices; SwiGLU has three $d\times h$ matrices, so setting $3h=8d$ gives $h\approx2.67d$.

### Q10. How do you do a gradient check, and why must it use double precision?
**Answer**: Compute numerical gradients per parameter with central differences and compare the relative error against your backward implementation; keep the network small, set $\epsilon$ around $10^{-6}$, and use float64. float32 has only about 7 significant digits, so the finite difference itself introduces errors on the order of $10^{-3}$ and cannot distinguish a $10^{-6}$ discrepancy.
**Follow-up**: Is there a built-in tool in PyTorch? `torch.autograd.gradcheck`, which also works on custom `autograd.Function` objects.

### Q11. How do initialization, normalization, and learning rate relate?
**Answer**: All three control the "signal scale". Initialization sets the variance at the starting point, normalization continuously pulls each layer's input back to a stable distribution during training, and the learning rate sets the size of each update. Bad initialization plus a large learning rate is a recipe for exploding on the first step; normalization buys tolerance for initialization, but you should still pick Xavier/He based on the activation.
**Follow-up**: What initialization is common in Transformers? Xavier or normal $\mathcal{N}(0,0.02)$, with residual output projections scaled down further (for example GPT-2 scales by $1/\sqrt{2N}$) to control residual accumulation in deep stacks.

## 5. Pitfalls and follow-ups

- **Mistaking `kaiming_uniform_` for He normal**: PyTorch's default `Linear` initialization is actually Kaiming uniform ($a=\sqrt{5}$), not "standard" He normal; specify it explicitly.
- **Using Xavier with ReLU**: variance decays layer by layer, so deep gradients shrink.
- **Thinking BN replaces initialization**: BN stabilizes the input distribution, not the backward gradient magnitude.
- **Ignoring the output activation**: linear for regression, sigmoid for binary classification, softmax for multiclass (do not apply softmax manually when using cross-entropy).
- **Running gradient checks in float32 or on large networks**: numerical noise drowns the signal and the conclusion is unreliable.

## 6. Cheat sheet

| Activation | Formula | Derivative | Main issue | Typical use |
| --- | --- | --- | --- | --- |
| sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)\le0.25$ | Vanishing gradients, non-zero mean | Binary output, gates |
| tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2$ | Still saturates | RNN hidden state |
| ReLU | $\max(0,z)$ | 0 or 1 | Dying ReLU, non-zero mean | CNNs, MLPs |
| LeakyReLU | $\max(\alpha z,z)$ | 1 or $\alpha$ | $\alpha$ needs tuning | Mitigating dying ReLU |
| GELU | $z\Phi(z)$ | $\Phi(z)+z\phi(z)$ | Slightly more expensive | BERT/GPT |
| SwiGLU | $\mathrm{Swish}(xW_1)\odot(xV)$ | - | One extra projection matrix | LLaMA FFN |

| Initialization | Weight variance | Suitable for |
| --- | --- | --- |
| Xavier/Glorot | $2/(n_{in}+n_{out})$ | tanh/sigmoid, near-linear |
| He/Kaiming | $2/n_{in}$ | ReLU family |
| Small-variance normal | e.g. $\mathcal{N}(0,0.02)$ | Common in Transformers |

**Mnemonic**: activation is about saturation, initialization is about variance; sigmoid pairs with Xavier, ReLU pairs with He, and deep networks fall back on normalization and residuals.

## 7. References

- [Deep Learning · Goodfellow et al.](https://www.deeplearningbook.org/): Sections 6.3 and 8.4
- [Understanding the difficulty of training deep feedforward neural networks (Xavier)](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers (He initialization)](https://arxiv.org/abs/1502.01852)
- [Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md)
