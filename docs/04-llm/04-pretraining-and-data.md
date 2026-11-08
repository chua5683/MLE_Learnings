# Pretraining and data engineering

> **One-line summary**: Pretraining is three things - **objective x data engineering x training stability**. The CLM objective was settled long ago; what actually separates models is data quality/mixing and the engineering details that keep the loss stable.
>
> **Keywords**: `CLM` `MLM` `span corruption` `FIM` `deduplication` `quality filtering` `packing` `bf16` `z-loss` `loss spike` `data contamination`

## 1. Core concepts

### 1.1 Comparing pretraining objectives

| Objective | Method | Supervision | Examples | Status |
| --- | --- | --- | --- | --- |
| CLM (causal LM) | predict token $t$ from the previous $t-1$ | every token | GPT, Llama, Qwen | the LLM mainstream |
| MLM (masked LM) | mask 15% at random, predict the masked tokens | masked positions only | BERT, RoBERTa | understanding tasks/embeddings |
| Span corruption | delete random spans, mark them with sentinels, reconstruct | deleted positions only | T5, mT5 | encoder-decoder |
| FIM (Fill-in-the-Middle) | split a document into prefix/middle/suffix and rearrange | the middle span | CodeLlama, StarCoder | required for code completion |
| Prefix LM | bidirectional prefix, causal suffix | suffix tokens | GLM, U-PaLM | a few models |

The CLM loss is just standard cross-entropy:

$$
\mathcal{L}_{\text{CLM}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

### 1.2 The two FIM permutations

Given a document with prefix $P$, middle $M$, and suffix $S$: **PSM** concatenates `<PRE>P<SUF>S<MID>M` (prefix, then suffix, then middle); **SPM** moves the suffix first, `<SUF>S<PRE>P<MID>M`. Both preserve the causal structure of "generate the middle last", so the model learns to fill in from the surrounding context, which is what code completion (cursor in the middle) needs. Cut points are chosen at random during training; FIM ratios are typically 50%-90%, and too high a ratio hurts ordinary CLM ability.

### 1.3 The data engineering pipeline

```mermaid
flowchart LR
    A["Raw web pages/books/code"] --> B["Parse and extract<br/>HTML -> text"]
    B --> C["Language ID / rule-based cleaning"]
    C --> D["Exact dedup<br/>hash"]
    D --> E["Fuzzy dedup<br/>MinHash-LSH"]
    E --> F["Quality filtering<br/>rules + classifiers"]
    F --> G["Safety/compliance filtering"]
    G --> H["Mixing and sampling"]
    H --> I["Tokenize + packing"]
    I --> J["Training"]
```

Each step answers a concrete question:

- **Deduplication**: duplicate documents cause memorization, waste compute, and distort scaling laws (see 2.1).
- **Quality filtering**: web text is full of templates, ads, and garbage. The usual two-stage setup is "rules (length, symbol ratio, repeated n-grams, language probability) + model scoring (fastText classifiers, small-LLM judges)".
- **Mixing**: the ratio of English/Chinese/code/math determines the capability mix. Llama-3's published mix is roughly **50% general, 25% math and reasoning, 17% code, 8% multilingual**.
- **Packing**: concatenate multiple short samples to a fixed length to avoid padding waste; documents must be separated by a document-level mask.
- **Curriculum/annealing**: raise the share of high-quality data in the mid-to-late phase and grow sequence length gradually (4k -> 32k), learning general knowledge before long documents and hard problems.

### 1.4 The "default profile" for training configs

| Hyperparameter | Typical value | Notes |
| --- | --- | --- |
| Global batch | 2M-16M tokens | larger is more stable but diminishing; needs warmup |
| Peak learning rate | 1e-4 to 3e-4 (7B class) | scales roughly with $\sqrt{B}$ |
| Warmup | 0.1%-1% of total steps | prevents early gradient explosions |
| Schedule | cosine to 10% of peak | WSD (warmup-stable-decay) is also common |
| Weight decay | 0.1 | applied to matrix weights only, not norm/bias |
| Gradient clipping | 1.0 | global-norm clipping |
| Precision | bf16 weights/activations + fp32 master weights | see 2.3 |
| Sequence length | 4k -> 8k -> 32k progressively | long-context continued pretraining |

## 2. Principles and derivations

### 2.1 Why deduplication is mandatory

If a document appears $r$ times in the corpus, it is effectively weighted $r$ times. Three consequences follow:

1. **Memorization**: the model tends to recite repeated content verbatim instead of learning patterns.
2. **Rapidly diminishing multi-epoch returns**: repeated-data experiments show gains up to about 4 epochs, after which returns approach zero or even turn negative; repetition also changes the Chinchilla-style data-parameter optimum.
3. **Evaluation distortion**: duplicated evaluation text leaking into the training set is data contamination (see 2.5).

In practice there are two levels: **exact hashing** (line/document hashes) is cheap; **MinHash + LSH** does approximate document-level dedup (Jaccard similarity) and catches duplicates that change templates or URLs.

### 2.2 Packing and document masks

The naive approach concatenates samples up to `max_len`, but then sample $i$ can "see" the beginning of sample $i+1$, causing cross-document attention leakage. There are two correct approaches:

- **block-diagonal mask**: each document attends only within its own block; simple, but kernel efficiency is poor;
- **varlen / document mask** (FlashAttention's `cu_seqlens`): batch multiple samples as variable-length sequences; most efficient and the current mainstream.

Also, during pretraining **all tokens contribute to the loss** (including EOS), and EOS teaches the model to stop; only SFT masks out the prompt part.

### 2.3 Mixed precision: bf16 vs fp16

| Format | Exponent bits | Mantissa bits | Needs loss scaling | Notes |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | no | stable but slow and memory-hungry |
| fp16 | 5 | 10 | **yes** | higher precision but narrow range, easy to overflow |
| bf16 | 8 | 7 | no | same range as fp32, fewer mantissa bits, the training default |

Large-model training usually uses "**bf16 compute + fp32 master weights/optimizer states**": forward and backward run in bf16 for speed and memory, while parameter updates happen in fp32 to avoid accumulated error. fp16 requires dynamic loss scaling (multiply the loss by a large factor before backprop to prevent gradient underflow).

### 2.4 Loss spikes and z-loss

Training occasionally shows the loss jumping by several points, then either recovering or collapsing. Common causes and fixes:

- **Causes**: attention logits too large (softmax saturation), growing embedding/residual norms, anomalous data samples, unstable second-moment estimates in the optimizer, learning rate too high.
- **z-loss**: add $\mathcal{L}_z=\lambda(\log\sum_j e^{z_j})^2$ to the softmax (PaLM uses $\lambda=10^{-4}$); it penalizes the log-partition function drifting away from 0 and suppresses global logit drift.
- **QK-norm**: normalize Q and K before attention, directly bounding logit magnitude.
- **Recovery strategy**: roll back to a checkpoint 100-200 steps before the spike, skip that data, lower the learning rate, increase gradient clipping; if that fails, restart with a smaller LR.

### 2.5 Data contamination

When evaluation sets leak into the training corpus, scores are inflated. Detection and mitigation:

- **n-gram overlap**: match training data against evaluation sets with 13-grams; drop any hit;
- **perplexity anomalies**: abnormally low PPL on an evaluation sample suggests the model has seen it;
- **canary strings**: plant special strings in evaluation data and check whether the model can reproduce them;
- **decontamination**: do it before training, not as an explanation after evaluation.

## 3. Implementation notes

```python
# 1) Rule-based quality filtering (an example; real systems have dozens of rules)
import re

def quality_ok(text, min_chars=200, max_symbol_ratio=0.1):
    if len(text) < min_chars:
        return False
    if text.count("\n") / max(len(text), 1) > 0.3:   # template pages with too many blank lines
        return False
    symbol = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))
    return symbol / len(text) < max_symbol_ratio

# 2) z-loss: suppress global drift of softmax logits
def z_loss(logits, coef=1e-4):
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return coef * (log_z ** 2).mean()

# 3) packing: hand cu_seqlens to flash-attn's varlen interface
# cu_seqlens = [0, len1, len1+len2, ...]; no need to build a big explicit mask
```

Two more engineering details: **sample-level dedup should run before tokenization** (to save cost); **mixing/sampling must be reproducible** (fixed seed + data version), otherwise experiments cannot be attributed.

## 4. Interview Q&A

### Q1. What is the difference between CLM and MLM? Why do LLMs use CLM?
**Answer**: CLM predicts the next token from the preceding context, supervises every position, and matches generation; MLM masks tokens at random and reconstructs them, which is a bidirectional understanding objective but mismatched with generation and supervises only 15% of positions. LLMs need generation and in-context ability, and CLM offers denser supervision and task consistency.
**Follow-up**: What is MLM still good for? Embedding, retrieval, and classification tasks, where BERT-style models remain efficient.

### Q2. What is T5's span corruption? Why is it no longer mainstream?
**Answer**: Random contiguous spans are deleted, marked with sentinel tokens, and the decoder reconstructs only the deleted content; this saves decoding steps compared with per-token MLM. It suits encoder-decoder seq2seq pretraining, but decoder-only CLM is simpler, more densely supervised, and scales better, so it fell out of fashion in the LLM era.
**Follow-up**: How is span length chosen? T5 uses spans of about 3 tokens on average, covering about 15% of the text.

### Q3. What is FIM? Why do code models require it?
**Answer**: FIM rearranges a document into "prefix + suffix + middle" so the model learns to fill in the middle (PSM/SPM are two permutations). In code completion the user cursor sits in the middle, and the model must generate the middle span from the surrounding context; ordinary CLM only continues the tail and cannot handle this.
**Follow-up**: What FIM ratio is appropriate? Commonly 50%-90%; too high hurts ordinary language modeling.

### Q4. Why deduplicate pretraining data? What happens if you repeat 4 epochs?
**Answer**: Repetition weights those samples more, causing memorization, wasted compute, and a distorted relationship between data volume and loss. Empirically the first few epochs still help, around 4 epochs the return approaches zero, and beyond that it can be harmful; repetition also corrupts "data volume" as a scaling-law variable.
**Follow-up**: What do exact and fuzzy dedup each catch? Exact catches identical documents; MinHash-LSH catches near-duplicates with template substitution or local edits.

### Q5. What methods exist for data quality filtering?
**Answer**: Two layers. The **rule layer** (length, symbol ratio, language confidence, repeated n-gram share, stopword distribution, template-heavy pages) and the **model layer** (fastText quality classifiers, small-LLM judges, educational-value classifiers such as FineWeb-Edu, DCLM's model filtering). The empirical conclusion is that **quality beats quantity**: less but cleaner data often trains a better model.
**Follow-up**: Does filtering remove useful data? Yes, so weigh recall against precision, keep thresholds relatively permissive, and run ablations.

### Q6. What is the difference between bf16 and fp16? Why do large models train in bf16?
**Answer**: bf16 has 8 exponent bits and 7 mantissa bits, giving the same dynamic range as fp32 but lower precision; fp16 has 5 exponent bits and 10 mantissa bits, giving higher precision but a narrower range that overflows/underflows easily and requires loss scaling. In large-model training, gradients and activations vary widely, so bf16 needs no loss scaling and is more stable, while the fp32 master weights restore mantissa precision.
**Follow-up**: fp16 or bf16 for inference? Either works, but bf16 is more stable; quantized inference is a separate story (see [09](09-quantization-and-compression.md)).

### Q7. How do you diagnose and handle loss spikes during training?
**Answer**: First localize - is it global or in a specific layer, does it coincide with a gradient-norm spike, is it correlated with a particular data batch? Then: 1) roll back to the checkpoint before the spike; 2) skip that data; 3) lower the peak LR or extend warmup; 4) add z-loss/QK-norm to bound logits; 5) increase gradient clipping. If spikes are frequent, check for anomalous samples and numerical overflow.
**Follow-up**: Why are large models more prone to spikes? More parameters, attention logits that diverge easily, and limited bf16 precision.

### Q8. What does z-loss do?
**Answer**: z-loss penalizes the square of $\log\sum_j e^{z_j}$, constraining the softmax normalization constant to stay near 0, which suppresses global numerical drift of logits and reduces the chance of loss spikes. It does not change the model's expressiveness; it acts as a soft constraint on the output layer.
**Follow-up**: What coefficient? PaLM uses $10^{-4}$; too large a value hurts normal learning.

### Q9. How should batch size and learning rate be paired?
**Answer**: Empirically LR grows with batch, approximately $\mathrm{LR}\propto\sqrt{B}$ (linear scaling is more aggressive and needs care); large batches need longer warmup. Beyond the critical batch size, further increases give diminishing returns in convergence steps and only improve throughput. For stable training also consider AdamW + gradient clipping + moderate weight decay.
**Follow-up**: What is μP? Maximal Update Parametrization, which lets hyperparameters tuned on small models transfer to large ones and cuts tuning cost.

### Q10. How do you decide the data mix?
**Answer**: Set initial ratios from capability goals (general/code/math/multilingual), then tune with small-scale ablations (fixed token budget, compare downstream benchmarks); a common practice is staged adjustment during training (raise high-quality/math/code shares late, i.e. "annealing"). There is no universal mix; it depends strongly on model size, vocabulary, and target scenario.
**Follow-up**: Why raise quality late? High-quality data has more influence on final capability late in training, similar to curriculum learning.

### Q11. How do you detect data contamination?
**Answer**: Match n-grams (e.g. 13-grams) between training and evaluation sets; check whether the model's PPL on evaluation samples is abnormally low; plant canary strings in the evaluation set and check whether the model can reproduce them; compare generated answers against evaluation references verbatim. The most reliable approach is to decontaminate before training and record versions.
**Follow-up**: Why is contamination hard to eliminate? Crawled data already contains benchmark pages, and paraphrased questions are hard to catch with n-grams.

### Q12. What do you do when pretraining data runs out?
**Answer**: Several directions: 1) multiple epochs (diminishing returns, but high-quality data is still worth repeating); 2) synthetic data/augmentation (math, code, textbook-style rewriting, guarding against model collapse and reduced diversity); 3) improve data quality and mixing efficiency; 4) shift budget from "more tokens" to "bigger model/longer training"; 5) use an annealing phase for high-quality reinforcement. Scaling under data constraints is covered in [05 Scaling laws](05-scaling-laws.md).
**Follow-up**: What are the risks of synthetic data? Narrow distribution, amplified errors, and evaluation contamination; filter strictly and mix in real data.

## 5. Pitfalls and follow-ups

- **"More data is always better" is wrong**: quality, deduplication, and mixing matter more than raw volume.
- **Packing without document separation lets the model peek at answers**: a document mask is mandatory.
- **bf16 is not "lower precision so worse"**: the range is sufficient and fp32 master weights restore precision.
- **z-loss does not regularize capability**: it only constrains the numerical range of the output layer.
- **Falling loss does not equal better capability**: also check downstream benchmarks and contamination.

## 6. Cheat sheet

| Topic | One-liner |
| --- | --- |
| CLM | supervises every token, consistent with generation |
| MLM | only 15% masked positions, bidirectional understanding |
| T5 | span corruption, encoder-decoder |
| FIM | prefix+suffix -> middle, required for code completion |
| Data pipeline | parse -> dedup -> quality -> safety -> mix -> packing |
| Dedup | exact hash + MinHash-LSH; prevents memorization and distortion |
| Quality filtering | rules + fastText/small-model scoring, quality beats quantity |
| Packing | varlen + cu_seqlens, documents must be separated |
| Precision | bf16 compute + fp32 master weights; fp16 needs loss scaling |
| Loss spike | rollback + skip data + lower LR + z-loss/QK-norm |
| Contamination | n-gram overlap + PPL + canary, decontaminate before training |

## 7. References

- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [PaLM: Scaling Language Modeling with Pathways (z-loss)](https://arxiv.org/abs/2204.02311)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [DCLM: DataComp-LM](https://arxiv.org/abs/2406.11794)
- [FineWeb / FineWeb-Edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [llm_interview_note · training datasets](https://github.com/wdndev/llm_interview_note)
