# Attention and positional encoding

> **One-line summary**: Every attention variant optimizes the same thing - **KV cache memory and bandwidth** (MHA -> MQA -> GQA -> MLA) - while FlashAttention optimizes **memory IO** and positional encoding (represented by RoPE) optimizes **length extrapolation**.
>
> **Keywords**: `KV cache` `MQA` `GQA` `MLA` `FlashAttention` `online softmax` `RoPE` `NTK` `YaRN` `ALiBi`

## 1. Core concepts

### 1.1 A quick recap of standard self-attention

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

Here $Q\in\mathbb{R}^{T\times d_h}$ and $M$ is the causal mask (upper triangle set to $-\infty$). A single head costs $O(T^2d_h)$, and after concatenating heads the total is about $O(T^2d)$. The standard derivation of Transformers is in [02 · Transformer basics](../02-dl-fundamentals/05-transformer-basics.md).

### 1.2 The two phases of inference

- **Prefill**: process the whole prompt at once; $T$ is large and the work is matrix-multiply heavy, so it is **compute-bound**.
- **Decode**: generate one token per step but attend to the historical K/V, so it is **memory-bound** (most time goes into moving the KV cache).

That is why half of inference optimization focuses on the KV cache: PagedAttention, continuous batching, KV quantization, and the attention variants in this note. Deployment details are in [08 Inference optimization and deployment](08-inference-optimization-and-serving.md).

### 1.3 Four attention variants

Let $L$ be the number of layers, $n_h$ the number of query heads, $n_{kv}$ the number of KV heads, $d_h$ the per-head dimension, and $T$ the sequence length.

| Variant | KV heads | Cached elements per layer per token | Relative to MHA | Examples |
| --- | --- | --- | --- | --- |
| MHA | $n_h$ | $2n_hd_h$ | $1\times$ | GPT-3, Llama-2-7B |
| MQA | $1$ | $2d_h$ | $1/n_h$ | PaLM, Falcon |
| GQA | $G$ ($1<G<n_h$) | $2Gd_h$ | $G/n_h$ | Llama-3, Qwen2.5, Mistral |
| MLA | low-rank latent | $d_c+d_R$ | about $1/50$ | DeepSeek-V2/V3 |

## 2. Principles and derivations

### 2.1 The KV cache memory formula (worth memorizing)

$$
\text{KV bytes}=2\times L\times n_{kv}\times d_h\times T\times B\times \text{bytes\_per\_elem}
$$

The leading 2 is one copy each for K and V. **Example: Llama-3-8B** ($L=32$, $n_{kv}=8$, $d_h=128$), $T=8192$, bf16 (2 bytes), batch=1:

$$
2\times32\times8\times128\times8192\times2=1\,\mathrm{GiB}
$$

In other words, a single 8k context already needs 1 GiB, and 128k needs 16 GiB; with MHA instead ($n_{kv}=32$), 8k would need 4 GiB. That is the motivation for GQA.

**The MLA accounting** (DeepSeek-V2, $L=60$, $d_c=512$, $d_R=64$, $n_h=128$, $d_h=128$): MLA caches $512+64=576$ elements per layer per token; the MHA equivalent is $2\times128\times128=32768$ elements, a compression of about **57x**. At 4k context and batch=1 that is about 283 MB versus 16 GiB.

### 2.2 Implementing GQA: repeat KV

```python
import torch

def repeat_kv(x, n_rep):           # x: [B, n_kv, T, d_h]
    B, n_kv, T, d = x.shape
    if n_rep == 1:
        return x
    # Replicate each KV head n_rep times for the corresponding Q heads
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, d)
    return x.reshape(B, n_kv * n_rep, T, d)
```

At **training** time GQA is equivalent to setting the number of KV heads to $G$, so $n_h/G$ query heads share one set of K/V; at inference you simply expand as above. It does not change the attention formula, only the head counts, so it is cheap to implement and effective - which is why it became the default.

### 2.3 FlashAttention: tiling + online softmax

Standard attention writes the $T\times T$ score matrix back to HBM, costing $O(T^2)$ memory and heavy IO. FlashAttention tiles Q/K/V into SRAM, computes block by block, and uses **online softmax** to update incrementally, avoiding materializing a whole row:

For each row, maintain a running maximum $m$, a running sum $\ell$, and a running output $O$. When processing a new block $\{x_j\}$:

$$
m'=\max(m,\max_j x_j),\qquad
\ell'=\ell\,e^{m-m'}+\sum_j e^{x_j-m'}
$$

$$
O'=\frac{\ell\,e^{m-m'}}{\ell'}\,O+\frac{\sum_j e^{x_j-m'}V_j}{\ell'}
$$

This satisfies both **numerical stability** (subtracting the max) and **incremental merging**, and the result is **exactly equal** to standard attention (not an approximation). Cost-wise: FLOPs are unchanged, HBM accesses drop from $\Theta(Nd+N^2)$ to $\Theta(N^2d^2/M)$ ($M$ is the SRAM size), and extra memory drops from $O(N^2)$ to $O(N)$. Backward does not store the score matrix but **recomputes** it, then backpropagates through the recomputed blocks - trading memory for compute, which still pays off overall. The causal mask skips upper-triangle blocks and saves roughly another half.

### 2.4 Sparse and sliding-window attention

- **Sliding window**: each token looks only at the most recent $w$ tokens, giving $O(Tw)$ complexity and a KV cache of size $w$. Mistral-7B uses a 4096 window.
- **Global + local**: Longformer/BigBird use a few global tokens to aggregate information, balancing long-range coverage.
- **Attention sinks**: StreamingLLM found that keeping the KV of the first few tokens stabilizes long streaming generation.
- In practice sliding-window layers are interleaved with full-attention layers (local layers save cost, global layers preserve capability).

### 2.5 Positional encoding: from absolute to relative

| Scheme | Form | Extrapolation | Notes |
| --- | --- | --- | --- |
| Sinusoidal absolute | fixed $\sin/\cos$ | weak | the original Transformer |
| Learned absolute | one vector per position | none | GPT-2/3 |
| Relative bias | add $b_{i-j}$ to attention logits | medium | T5 buckets, Shaw |
| **RoPE** | rotate Q/K | strong (extendable) | Llama/Qwen/DeepSeek |
| ALiBi | subtract $m\cdot(i-j)$ from logits | strong | BLOOM |
| NoPE | no positional encoding | strong | implicit via the causal mask |

### 2.6 Deriving RoPE

Group the $d$ dimensions into pairs and rotate pair $i$ by angle $\theta_i m$, with $\theta_i=10000^{-2i/d}$:

$$
R_{\Theta,m}=\mathrm{diag}\!\left(\begin{pmatrix}\cos m\theta_1&-\sin m\theta_1\\ \sin m\theta_1&\cos m\theta_1\end{pmatrix},\dots\right)
$$

For $q_m=R_{\Theta,m}W_qx_m$ and $k_n=R_{\Theta,n}W_kx_n$ we have

$$
q_m^\top k_n=(R_mq)^\top(R_nk)=q^\top R_m^\top R_nk=q^\top R_{n-m}k
$$

That is, **the inner product depends only on the relative displacement $n-m$**, which is why RoPE encodes relative position. In code you just concatenate with `rotate_half` instead of building the matrix explicitly:

```python
def rotate_half(x):                     # x: [..., d]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):            # cos/sin: [T, d]
    return x * cos + rotate_half(x) * sin
```

### 2.7 Length extrapolation: PI, NTK, YaRN

A model is trained only on length $L$ but you want $L'>L$:

- **Position interpolation (PI)**: rescale position $m$ to $m\cdot L/L'$, effectively squeezing unseen positions back into the training distribution; the cost is compressed high-frequency detail and reduced short-range resolution.
- **NTK-aware**: keep positions but change the base: $\text{base}'=\text{base}\cdot s^{\,d/(d-2)}$, with $s=L'/L$. High-frequency dimensions barely move while low-frequency ones are stretched, preserving more local information than PI.
- **YaRN**: NTK-by-parts - split by wavelength, interpolate low frequencies, leave high frequencies intact, and transition smoothly in between; then multiply attention logits by an $s$-dependent temperature (e.g. $\sqrt{1/t}$ or mscale) to compensate for the entropy change. It usually pairs with a small amount of long-text continued pretraining and works best.

## 3. Implementation notes

- `F.scaled_dot_product_attention` (PyTorch 2.x) selects the Flash/Memory-Efficient kernel automatically; when writing it by hand, avoid materializing the $T\times T$ matrix.
- In a KV cache implementation, keep the concatenation dimension of `past_key_value` separate from `position_ids` (RoPE uses absolute positions; the cache only appends).
- When training long context with `flash_attn`, keep `seqlen` aligned (usually a multiple of 8 or 128) or it falls back to a slow path.

## 4. Interview Q&A

### Q1. Why divide attention by $\sqrt{d_h}$?
**Answer**: If each dimension of $q,k$ is independent with mean 0 and variance 1, then $q^\top k$ has variance $d_h$, so the dot product grows with dimension. Without scaling the softmax input has large variance and saturates into a one-hot distribution with near-zero gradients. Dividing by $\sqrt{d_h}$ normalizes the variance to 1 and keeps training stable.
**Follow-up**: Why not divide by something else? Some implementations use QK-norm to normalize q/k directly, which is even more stable.

### Q2. How do you estimate KV cache memory? Give a concrete example.
**Answer**: The formula is $2\cdot L\cdot n_{kv}\cdot d_h\cdot T\cdot B\cdot\text{bytes}$. Take Llama-3-8B (32 layers, 8 KV heads, $d_h=128$) with 8k context, bf16, batch=1: $2\times32\times8\times128\times8192\times2\approx1$ GiB; at 128k it is 16 GiB.
**Follow-up**: Why does inference OOM as batch grows? The KV cache grows linearly with batch while the weights stay fixed.

### Q3. How do you choose between MHA, MQA, GQA, and MLA?
**Answer**: Choose MHA when quality matters most and memory is ample; MQA when memory is extremely tight, accepting slightly worse quality and stability; **GQA is the default sweet spot** (KV heads between $n_h/4$ and $n_h/8$), close to MHA at a fraction of the cost; choose MLA for extreme compression if you can afford implementation complexity. The choice also depends on whether your inference framework has the matching kernels.
**Follow-up**: Why can MLA compress harder than GQA? GQA only shares heads, whereas MLA caches a low-rank latent vector and compresses "the per-token dimension".

### Q4. Why is GQA close to MHA in quality?
**Answer**: GQA only makes multiple query heads share KV representations; the query side keeps all heads, and most of the scoring diversity comes from the queries. The shared K/V is shaped by gradients from multiple heads during training, which makes it more robust. Experimentally, 8 KV groups already approach MHA, and quality only drops noticeably at 1 group (MQA).
**Follow-up**: Does GQA need special treatment during training? No - it is equivalent to setting $n_{kv}$ smaller, i.e. reducing the output dimension of the linear layers.

### Q5. Is FlashAttention an approximation? Why is it faster?
**Answer**: **It is not an approximation**; the math is identical to standard attention. The speedup comes from two things: 1) tiled computation + online softmax avoids writing and reading the $T\times T$ score matrix to/from HBM, cutting IO; 2) backward recomputes instead of storing intermediate matrices. Attention's bottleneck is memory bandwidth, not FLOPs, so saving IO translates into a large speedup.
**Follow-up**: Why is FlashAttention not always faster on short sequences? Tiling and recomputation have fixed overhead that can cancel the gain when sequences are short.

### Q6. Why does FlashAttention save memory but not necessarily FLOPs?
**Answer**: It does not change the number of attention multiplications; FLOPs are the same as the standard implementation (slightly more due to recomputation). What it saves is HBM traffic and the $O(T^2)$ intermediate activation memory, which lets you use larger batches and longer contexts and thus raises throughput indirectly.
**Follow-up**: What changed in FlashAttention-2/3? FA-2 improves work partitioning and reduces the share of non-matmul FLOPs; FA-3 targets Hopper with TMA/WGMMA asynchronous pipelines.

### Q7. Derive why RoPE represents relative position.
**Answer**: RoPE rotates Q/K in 2-D blocks, and the rotation matrices satisfy $R_m^\top R_n=R_{n-m}$, so $q_m^\top k_n=q^\top R_{n-m}k$ and the inner product depends only on the relative displacement. It also preserves vector norms, so the attention distribution is not "drowned" by positional information.
**Follow-up**: Is RoPE additive or multiplicative? Multiplicative (rotating Q/K), unlike additive absolute positional encodings.

### Q8. Why does direct RoPE extrapolation break? How do PI/NTK/YaRN fix it?
**Answer**: Training only sees positions in $[0,L]$; beyond that the rotation angles enter unseen regions and the attention distribution drifts. PI squeezes new positions back into the training range at the cost of resolution; NTK changes the base to stretch low frequencies while preserving high frequencies; YaRN handles frequency bands separately with temperature compensation, usually plus a little long-text continued pretraining.
**Follow-up**: Which is simplest? PI is one line of position_ids manipulation, but its quality is mediocre; NTK/YaRN work better and are more complex.

### Q9. What is the difference between ALiBi and RoPE?
**Answer**: ALiBi adds no positional vectors; it subtracts a bias proportional to distance, $m\cdot(i-j)$, directly from attention logits, so it is parameter-free and naturally supports any length. RoPE rotates Q/K and carries richer relative-position information. RoPE is usually stronger on long context, while ALiBi is simpler and extrapolates "mindlessly".
**Follow-up**: Why does ALiBi extrapolate well? The distance penalty is linear, so the model can apply the same decay rule to distances it never saw.

### Q10. Why does everyone use RoPE instead of learned absolute positional encoding?
**Answer**: Learned absolute positions only cover the training length and are OOV beyond it; RoPE encodes relative displacement, adds no parameters, works naturally with the KV cache (rotate by position only), and supports PI/NTK/YaRN extensions. It wins on both engineering and quality, so it became the de facto standard.
**Follow-up**: Why might NoPE also work? The causal mask itself provides positional information (token $t$ can only see the first $t$), and some models find they can learn order without explicit positions.

### Q11. How does sliding-window attention save KV cache, and what capability does it lose?
**Answer**: Each token keeps only the most recent $w$ KV entries, so the cache drops from $O(T)$ to $O(w)$ and compute from $O(T^2)$ to $O(Tw)$. The cost is that information outside the window is only accessible indirectly through inter-layer propagation, weakening exact long-range retrieval; hence the common pattern of alternating local and global layers.
**Follow-up**: Can sliding windows be combined with positional extension? Yes, but the window limits the effective visible distance, so the benefit is limited.

### Q12. Why is prefill compute-bound and decode memory-bound?
**Answer**: Prefill processes $T$ tokens at once with large matrix multiplies, giving high arithmetic intensity and making it compute-limited; decode handles one token per step but must read all weights and the KV cache, giving low arithmetic intensity and making it memory-bandwidth-limited. So prefill optimization looks at FLOPs utilization, while decode optimization looks at batching and bandwidth (see [08](08-inference-optimization-and-serving.md)).
**Follow-up**: Which techniques does this explain? Continuous batching improves decode batch utilization; PD disaggregation splits the two workloads onto different machines.

## 5. Pitfalls and follow-ups

- **The KV cache exists for every layer**: do not forget the factor $L$, nor that K and V are separate copies.
- **GQA saves KV heads, not query heads**: attention FLOPs barely change; what is saved is memory and bandwidth.
- **FlashAttention does not change the math**: treat it as an IO optimization, not an approximate attention.
- **A larger RoPE base is not always better**: it must match the training length and continued pretraining; changing it blindly loses points.
- **"Extrapolation" differs from "interpolation"**: PI maps new positions back into the training interval (interpolation); NTK/YaRN change the frequency distribution to extrapolate.

## 6. Cheat sheet

| Topic | One-liner |
| --- | --- |
| KV cache formula | $2Ln_{kv}d_hTB\cdot$bytes |
| MHA -> GQA | KV heads from $n_h$ to $G$, cache down by $G/n_h$ |
| MQA | KV heads = 1, cheapest but slightly worse |
| MLA | caches a $d_c+d_R$ latent, about 50x compression |
| FlashAttention | tiling + online softmax, exact, saves IO and memory |
| Online softmax | maintain $m,\ell,O$, rescale incrementally |
| RoPE | rotates Q/K; inner product depends only on relative displacement |
| PI | rescale positions into the training interval |
| NTK | change the base to stretch frequencies |
| YaRN | frequency bands + temperature compensation |
| ALiBi | linear distance penalty on logits, parameter-free |
| prefill/decode | the former is compute-bound, the latter memory-bound |

## 7. References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Efficiently Scaling Transformer Inference (MQA/KV cache analysis)](https://arxiv.org/abs/2211.05102)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [Train Short, Test Long: ALiBi](https://arxiv.org/abs/2108.12409)
- [DeepSeek-V2 Technical Report (MLA)](https://arxiv.org/abs/2405.04434)
- [llm_interview_note · attention / positional encoding](https://github.com/wdndev/llm_interview_note)
