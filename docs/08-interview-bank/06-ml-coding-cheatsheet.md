# ML Coding Cheat Sheet

> **How to use**: Reference templates and edge-case checklists for the **ML Coding** interview round (45–60 min, PyTorch or NumPy). Every algorithm below also has a zero-dependency runnable implementation and unit tests in `ml_coding/` at the repository root (`python3 -m unittest discover -s ml_coding -v`).

## 1. Universal ML coding checklist (say these out loud)

1. **Confirm shapes & dtypes first**: write tensor shapes as comments (`# [B, T, D]`) on every line.
2. **Numerical stability**: subtract `max` before `exp` (softmax / log-sum-exp), clamp probabilities before `log`, add `eps` inside `sqrt` / division.
3. **Train vs. eval & masking**: `training` flag for BatchNorm/Dropout; mask with `-inf` (or `-1e9`) **before** softmax, never multiply by `0` after without renormalizing.
4. **Edge cases**: empty input, all-same labels (AUC), empty clusters (KMeans), ties in ranking, `top_p=1.0` / `temperature -> 0`, `unbiased=False` in variance.

## 2. Ten canonical templates

### 2.1 Numerically stable Softmax & Cross-Entropy

```python
import numpy as np

def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    # logits: [N, C], targets: [N] int class indices
    log_p = log_softmax(logits, axis=-1)
    return float(-np.mean(log_p[np.arange(len(targets)), targets]))
```

### 2.2 Logistic Regression (binary cross-entropy + L2)

```python
def sigmoid(z):
    # Numerically stable piecewise sigmoid
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def fit_logreg(X, y, lr=0.1, l2=1e-4, steps=200):
    # X: [N, D], y: [N] in {0, 1}
    N, D = X.shape
    w, b = np.zeros(D), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        diff = (p - y) / N
        w -= lr * (X.T @ diff + l2 * w)     # do NOT regularize bias b
        b -= lr * np.sum(diff)
    return w, b
```

### 2.3 K-Means (vectorized + empty-cluster guard)

```python
def kmeans(X, k, max_iter=100, tol=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(max_iter):
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)  # [N, K]
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            new_centers[j] = X[mask].mean(axis=0) if np.any(mask) else X[rng.integers(len(X))]
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, labels
```

### 2.4 ROC-AUC in $O(N \log N)$ with exact tie handling

```python
def roc_auc(y_true, y_score) -> float:
    # Mann-Whitney U / average rank for ties
    order = np.argsort(y_score)
    y_sorted = np.asarray(y_true)[order]
    s_sorted = np.asarray(y_score)[order]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined when only one class is present")
    ranks = np.empty(len(s_sorted), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0      # 1-based average rank for tied group
        ranks[i : j + 1] = avg_rank
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[y_sorted == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
```

### 2.5 BatchNorm1d, LayerNorm, and RMSNorm (PyTorch)

```python
import torch, torch.nn as nn

class SimpleBatchNorm1d(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.eps, self.momentum = eps, momentum

    def forward(self, x):                    # x: [B, D]
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                var_unbiased = x.var(dim=0, unbiased=True) if x.size(0) > 1 else var
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var_unbiased, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

def rms_norm(x, weight, eps=1e-6):           # x: [B, T, D]
    return weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
```

### 2.6 Multi-Head Self-Attention with Causal Mask & KV Cache (PyTorch)

```python
class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):     # x: [B, T, D]
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)        # [B, H, T, dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)               # [B, H, S, dh]
            v = torch.cat([kv_cache[1], v], dim=2)
        S = k.size(2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)    # [B, H, T, S]
        if T > 1:                                                # causal mask for prefill/train
            mask = torch.triu(torch.ones(T, S, device=x.device, dtype=torch.bool), diagonal=S - T + 1)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), (k, v)
```

### 2.7 Temperature + Top-k + Top-p (Nucleus) Sampling

```python
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    # logits: [V]
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits / temperature
    if 0 < top_k < len(z):
        idx = np.argpartition(z, -top_k)[-top_k:]
        mask = np.full_like(z, -np.inf)
        mask[idx] = z[idx]
        z = mask
    probs = np.exp(log_softmax(z))
    if top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = cum - probs[order] < top_p    # always keep at least the top-1 token
        filtered = np.zeros_like(probs)
        filtered[order[keep]] = probs[order[keep]]
        probs = filtered / filtered.sum()
    return int(np.random.choice(len(probs), p=probs))
```

### 2.8 LoRA Linear Layer (with zero-init $B$ and $\alpha/r$ scaling)

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r)) # zero init => ΔW=0 at step 0
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
```

### 2.9 Quick-recall formulas

| Problem | Key formula / edge case |
| --- | --- |
| Conv2d output size | $H_{\mathrm{out}} = \lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$ |
| BPE merge step | Count adjacent symbol pairs → merge most frequent pair → repeat; preserve merge priority order for encoding |
| GQA head sharing | `k.repeat_interleave(n_q // n_kv, dim=1)` before `q @ k.T` |
| Gradient check | Finite difference $\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}$ in `float64` (`eps=1e-6`), compare relative error $< 10^{-5}$ |

## 3. Practice locally

Run the self-contained Python implementations and test suite:

```bash
python3 -m unittest discover -s ml_coding -v
```

See [02 Interview preparation playbook](../00-roadmap/02-interview-preparation-playbook.md) for how to pace a 45-minute ML coding round.
