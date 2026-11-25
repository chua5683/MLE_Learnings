"""Zero-dependency pure-Python reference implementations of canonical ML coding problems.

Designed for local practice and verification with `python3 -m unittest discover -s ml_coding -v`.
For idiomatic NumPy/PyTorch interview templates, see `docs/08-interview-bank/06-ml-coding-cheatsheet.md`.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def log_softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        raise ValueError("logits must be non-empty")
    m = max(logits)
    shifted = [x - m for x in logits]
    log_sum = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum for x in shifted]


def softmax(logits: Sequence[float]) -> List[float]:
    return [math.exp(v) for v in log_softmax(logits)]


def cross_entropy(logits_batch: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    if not logits_batch or len(logits_batch) != len(targets):
        raise ValueError("logits_batch and targets must be non-empty and same length")
    total = 0.0
    for logits, y in zip(logits_batch, targets):
        lp = log_softmax(logits)
        total -= lp[y]
    return total / len(targets)


def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic_regression(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    lr: float = 0.1,
    l2: float = 1e-4,
    steps: int = 300,
) -> Tuple[List[float], float]:
    n = len(X)
    if n == 0 or n != len(y):
        raise ValueError("X and y must be non-empty and aligned")
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(steps):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b)
            err = p - yi
            gb += err
            for j in range(d):
                gw[j] += err * xi[j]
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)
    return w, b


def kmeans(
    X: Sequence[Sequence[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> Tuple[List[List[float]], List[int]]:
    n = len(X)
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 1 <= k <= len(X)")
    d = len(X[0])
    rng = random.Random(seed)
    centers = [list(X[idx]) for idx in rng.sample(range(n), k)]
    labels = [0] * n
    for _ in range(max_iter):
        for i, xi in enumerate(X):
            best_c = 0
            best_dist = float("inf")
            for c_idx, c in enumerate(centers):
                dist = sum((a - b) ** 2 for a, b in zip(xi, c))
                if dist < best_dist:
                    best_dist = dist
                    best_c = c_idx
            labels[i] = best_c
        new_centers: List[List[float]] = []
        shift = 0.0
        for c_idx in range(k):
            members = [X[i] for i, lbl in enumerate(labels) if lbl == c_idx]
            if not members:
                nc = list(X[rng.randrange(n)])
            else:
                nc = [sum(m[j] for m in members) / len(members) for j in range(d)]
            shift += sum((a - b) ** 2 for a, b in zip(nc, centers[c_idx]))
            new_centers.append(nc)
        centers = new_centers
        if math.sqrt(shift) < tol:
            break
    return centers, labels


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("y_true and y_score must be non-empty and same length")
    paired = sorted(zip(y_score, y_true), key=lambda p: p[0])
    n_pos = sum(1 for _, y in paired if y == 1)
    n_neg = len(paired) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both positive and negative classes")
    rank_sum_pos = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average rank for ties
        pos_in_group = sum(1 for idx in range(i, j + 1) if paired[idx][1] == 1)
        rank_sum_pos += pos_in_group * avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def layer_norm(x: Sequence[float], gamma: Sequence[float], beta: Sequence[float], eps: float = 1e-5) -> List[float]:
    d = len(x)
    mean = sum(x) / d
    var = sum((v - mean) ** 2 for v in x) / d
    inv_std = 1.0 / math.sqrt(var + eps)
    return [g * (v - mean) * inv_std + b for v, g, b in zip(x, gamma, beta)]


def rms_norm(x: Sequence[float], weight: Sequence[float], eps: float = 1e-6) -> List[float]:
    ms = sum(v * v for v in x) / len(x)
    inv_rms = 1.0 / math.sqrt(ms + eps)
    return [w * v * inv_rms for v, w in zip(x, weight)]


def scaled_dot_product_attention(
    Q: Sequence[Sequence[float]],
    K: Sequence[Sequence[float]],
    V: Sequence[Sequence[float]],
    causal: bool = False,
) -> List[List[float]]:
    t_q = len(Q)
    t_k = len(K)
    dk = len(Q[0])
    dv = len(V[0])
    scale = 1.0 / math.sqrt(dk)
    out: List[List[float]] = []
    for i in range(t_q):
        scores = []
        for j in range(t_k):
            if causal and j > i + (t_k - t_q):
                scores.append(float("-inf"))
            else:
                scores.append(sum(Q[i][m] * K[j][m] for m in range(dk)) * scale)
        # Handle softmax with -inf safely
        max_s = max(s for s in scores if s != float("-inf"))
        exps = [0.0 if s == float("-inf") else math.exp(s - max_s) for s in scores]
        denom = sum(exps)
        weights = [e / denom for e in exps]
        row = [sum(weights[j] * V[j][m] for j in range(t_k)) for m in range(dv)]
        out.append(row)
    return out


def sample_next_token(
    logits: Sequence[float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: Optional[random.Random] = None,
) -> int:
    if not logits:
        raise ValueError("logits must be non-empty")
    if temperature <= 0.0:
        return max(range(len(logits)), key=lambda i: logits[i])
    rng = rng or random.Random()
    scaled = [z / temperature for z in logits]
    indexed = sorted(enumerate(scaled), key=lambda p: p[1], reverse=True)
    if 0 < top_k < len(indexed):
        indexed = indexed[:top_k]
    probs_kept = softmax([z for _, z in indexed])
    if top_p < 1.0:
        filtered: List[Tuple[int, float]] = []
        cum = 0.0
        for (tok_id, _), p in zip(indexed, probs_kept):
            filtered.append((tok_id, p))
            cum += p
            if cum >= top_p:
                break
        total = sum(p for _, p in filtered)
        tok_ids = [t for t, _ in filtered]
        probs = [p / total for _, p in filtered]
    else:
        tok_ids = [t for t, _ in indexed]
        probs = probs_kept
    u = rng.random()
    acc = 0.0
    for tok_id, p in zip(tok_ids, probs):
        acc += p
        if u <= acc:
            return tok_id
    return tok_ids[-1]


def train_bpe(corpus_words: Sequence[str], num_merges: int) -> List[Tuple[str, str]]:
    """Minimal BPE trainer returning the ordered list of learned symbol-pair merges."""
    vocab: Counter[Tuple[str, ...]] = Counter()
    for word in corpus_words:
        if word:
            vocab[tuple(word) + ("</w>",)] += 1
    merges: List[Tuple[str, str]] = []
    for _ in range(num_merges):
        pair_counts: Counter[Tuple[str, str]] = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best_pair)
        merged_sym = best_pair[0] + best_pair[1]
        new_vocab: Counter[Tuple[str, ...]] = Counter()
        for symbols, freq in vocab.items():
            out: List[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best_pair:
                    out.append(merged_sym)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab
    return merges


def conv2d_output_shape(
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
) -> Tuple[int, int]:
    h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return h_out, w_out
