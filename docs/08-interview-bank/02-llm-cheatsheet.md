# LLM cheat sheet

> **One-line summary**: Scan this page in the 30 minutes before an interview - for architecture, know "why decoder-only + RoPE + GQA"; for training, the memory and Scaling Law formulas; for inference, KV cache plus throughput/latency; for applications, the RAG/Prompt defaults and failure modes.
>
> **Keywords**: `KV Cache` `GQA/MLA` `Chinchilla` `LoRA` `ZeRO` `quantization` `TTFT/TPOT` `RAG` `LLM-as-judge`

## 1. Architecture and training: conclusions you must say without thinking

| Topic | Conclusion |
| --- | --- |
| Language modeling | $p(x)=\prod_t p(x_t\mid x_{<t})$, trained with cross-entropy, $\text{PPL}=\exp(\text{CE})$ |
| Why decoder-only dominates | Unified objective, KV-cache friendly, strong zero-shot generalization; encoder-decoder still wins on translation/summarization |
| Attention variants | KV size $\propto$ number of KV heads: MHA > GQA > MQA; MLA caches a low-rank latent (DeepSeek-V2 reports another 93.3% reduction over GQA) |
| FlashAttention | Exact attention via tiling + online softmax, cutting HBM traffic below $O(N^2)$ to block granularity |
| Positional encoding | RoPE is mainstream; long-context extrapolation uses NTK/YaRN/PI; ALiBi uses a distance bias |
| Normalization/activation | Pre-LN + RMSNorm + SwiGLU is the Llama-family standard |
| Scaling Law | Training FLOPs $\approx 6ND$; Chinchilla compute-optimal is ~20 tokens/parameter; today everyone "overtrains" small models to cut inference cost |
| Full fine-tuning memory | AdamW in mixed precision is ~16-20 B/parameter (weights 2 + gradients 2 + fp32 master 4 + Adam 8) |
| LoRA | $\Delta W=BA$, parameter count $r(d+k)$, $\alpha/r$ scaling, mergeable with zero inference latency; QLoRA = 4-bit base + LoRA |
| ZeRO | Stage 1 shards optimizer states, 2 also gradients, 3 also parameters; FSDP is similar to ZeRO-3 |

## 2. Inference: formulas + three numbers to memorize

$$\text{KV per token}=2\times L\times n_{kv}\times d_{head}\times p\ \text{(bytes)},\qquad \text{KV total}=\text{above}\times S\times B$$

| Model | KV per token (FP16) | One 4K sequence |
| --- | --- | --- |
| Llama-2-7B (MHA, L=32, 32 heads x128) | 512 KiB | 2 GiB |
| Llama-3-8B (GQA-8) | 128 KiB | 512 MiB |
| Llama-2-70B (GQA-8, L=80) | 320 KiB | 1.25 GiB |

| Technique | One-liner | Gain |
| --- | --- | --- |
| Prefill / Decode | Prefill is compute-bound (GEMM), decode is bandwidth-bound (GEMV) | Sets TTFT / TPOT |
| PagedAttention | KV stored in non-contiguous blocks + block table, <4% waste | 2-4x throughput |
| Continuous batching | Rebuild the batch every decode step; finished sequences leave immediately | 2-3x+ throughput |
| Chunked prefill | Split long prompts into chunks and mix with decode | Stable TPOT |
| Prefix caching | Reuse KV of a shared prefix (vLLM / SGLang RadixAttention) | Lower TTFT, saves compute |
| Speculative decoding | Draft guesses $\gamma$ tokens, target verifies in one pass; expected output $(1-\alpha^{\gamma+1})/(1-\alpha)$ | 1.5-3x, **distribution-preserving** |
| PD disaggregation | Separate prefill/decode pools + KV transfer | Higher goodput at the same SLA |
| CUDA graph | Replay fixed shapes to skip kernel launch overhead | Effective for small-batch decode |

Memorize: 7B FP16 ~ 14 GB, 70B FP16 ~ 140 GB, 70B INT4 ~ 35 GB; A100 80G HBM ~ 2 TB/s, so the 7B single-request decode ceiling is ~140 tok/s (discount in practice).

## 3. Quantization and compression

| Method | Bits | Type | Key point |
| --- | --- | --- | --- |
| LLM.int8() | W8A8 | PTQ | Mixed-precision decomposition of outlier channels |
| SmoothQuant | W8A8 | PTQ | $Y=(X\text{diag}(s)^{-1})(\text{diag}(s)W)$, $\alpha\approx0.5$, zero inference overhead |
| GPTQ | 3/4-bit | W4A16 | Hessian per-column quantization + error compensation, group 128, act-order |
| AWQ | 4-bit | W4A16 | Protects ~1% salient channels (chosen by activation), cheap calibration |
| GGUF | 2-8-bit | PTQ | llama.cpp format, CPU/edge |
| FP8 | 8-bit | PTQ/training | E4M3 forward, E5M2 backward, dynamic scale needs no calibration |
| KV quantization | 8/4-bit | PTQ | K per-channel, V per-token; 4-bit degrades long context |

Principles: **weights quantize easily, activations do not (outliers), KV depends on length**; W4A16 saves capacity and bandwidth by about 1.5-2.5x, and only W8A8 truly raises compute; after quantization always check per-task scores + generation quality + long context, not just PPL.

## 4. Evaluation and hallucination

| Benchmark | What it measures | Metric |
| --- | --- | --- |
| MMLU / C-Eval | Subject knowledge (four-way) | accuracy |
| GSM8K / MATH | Math reasoning | exact match (CoT) |
| HumanEval / MBPP | Code | pass@k |
| MT-Bench / AlpacaEval 2.0 | Dialogue and instruction following | judge score / LC win rate |
| Chatbot Arena | Human preference | Elo / Bradley-Terry |
| LongBench / RULER | Long context | Task metrics + length sweeps |

- $\text{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}$; $\text{Elo}: E_A=1/(1+10^{(R_B-R_A)/400})$; judge-human agreement uses $\kappa=(p_o-p_e)/(1-p_e)$, usable only above 0.6.
- **Data contamination**: n-gram overlap, question variants, likelihood anomalies, time splits (LiveCodeBench).
- **Five LLM-as-judge biases**: position, length, self-preference, sycophancy, format -> swap order + multi-family judges + human alignment.
- **Three hallucination types**: factual (contradicts facts/unverifiable), faithfulness (violates context/instructions), self-contradiction; attribute root causes in three layers: data-training-inference.
- **Mitigation**: RAG for evidence, train "I don't know", lower decoding randomness/DoLa/CAD, mandatory citations and human handoff in the product. Hallucination is not the same as a wrong answer; evaluation must track both hallucination and refusal rates.
- **Launch system**: public leaderboard for positioning -> scenario golden set -> scaled judge -> online A/B + spot checks; all four layers are required.

## 5. RAG and Prompt: default choices

| Stage | Default | Key parameters/points |
| --- | --- | --- |
| Chunking | Recursive splitting | 256-1024 tokens, 10-20% overlap, tables/code as separate chunks |
| Index | HNSW (millions) / IVF-PQ (billions) | M, efSearch / nlist, nprobe |
| Recall | dense + BM25 hybrid | top-50, RRF with $k=60$ on ranks |
| Rerank | cross-encoder | top-50 -> top-5, plus a relevance threshold |
| Query rewriting | Coreference resolution / Multi-Query / HyDE | Enable only when semantics mismatch |
| Generation | Mandatory citations + allowed refusal | Low temperature, sentence-level faithfulness checks |
| Evaluation | Retrieval Recall@k + generation faithfulness | Golden set must point to chunk ids |

RAG failure-mode quick list: **not retrieved** (chunking/embedding/semantic gap) -> **retrieved but off-topic** (noise, k too large) -> **half right, half invented** (faithfulness) -> **edits/deletes not applied** (index sync) -> **privilege leak** (filtering must come first).

Prompt quick list: $T$=0 for extraction/classification/JSON; CoT buys serial compute, self-consistency votes over $N$ paths; tool calling uses ReAct + whitelists + feeding errors back; structured output relies on schema/constrained decoding + validation retry; injection defense relies on "isolation marking + least privilege + backend authorization + explicit confirmation", and **cannot be fully prevented**; context engineering means "put static content first to hit the prefix cache, restate key constraints at both ends".

## 6. Eight rapid-fire follow-up answers

- **Q**: Why does decoder-only dominate? **A**: Unified training objective + KV-cache friendly + strong zero-shot generalization; encoder-decoder still has an edge on translation/summarization.
- **Q**: Why does GQA save memory without losing much quality? **A**: Multiple Q heads share one set of KV heads, so the KV cache scales with the KV head count; a small amount of training uptrains an MHA model into it.
- **Q**: Chinchilla says 20 tokens/parameter, so why is everyone "overtraining" now? **A**: Inference cost dwarfs training cost, so training a small model longer pays off at deployment.
- **Q**: Why does LoRA save memory? **A**: The base is frozen, so you only store adapter gradients and optimizer states, with parameter count $r(d+k)$.
- **Q**: Why is 4-bit quantization not a 4x speedup? **A**: Weight-only quantization needs dequantization and prefill is compute-bound; only W8A8 both saves bandwidth and raises compute.
- **Q**: Can a "do not make things up" prompt eliminate hallucination? **A**: No; the model's "I don't know" signal is unreliable, so external evidence + verification + a refusal policy are required.
- **Q**: How do you choose between RAG and long context? **A**: Many documents/constant updates/needs permissions and traceability -> RAG; few documents + strong reasoning -> long context; production usually mixes both.
- **Q**: Why can't prompt injection be prevented? **A**: All input is tokens with no code-level "instruction/data" separation, so you can only fall back on isolation marking + least privilege + backend authorization.

## 7. One-page mnemonic

**Architecture**: unified objective means decoder-only, RoPE + RMSNorm + SwiGLU, GQA saves KV, MoE saves compute.
**Training**: $6ND$ for FLOPs, 20 tokens/parameter for data, 16-20 B/parameter for memory, ZeRO-3 shards everything.
**Inference**: prefill fights for FLOPs, decode fights for bandwidth, scheduling fights for utilization, speculation fights for acceptance rate.
**Quantization**: 4x capacity saving but only 1.5-2.5x speed, only W8A8 raises compute, re-measure the task after quantizing.
**Evaluation**: leaderboards for trends, eval sets for the product, judges for alignment, online for results.
**Hallucination**: evidence first, allow refusal, citations must be checkable, spot checks as a backstop.
**RAG**: the index sets the ceiling, retrieval sets the floor, rerank adds precision, the prompt keeps faithfulness.
**Prompt**: concrete instructions, examples define format, schema governs output, isolation stops injection, evaluation governs iteration.

## 8. Further reading

- Architecture and training: [01 Language modeling and tokenization](../04-llm/01-language-models-and-tokenization.md), [02 LLM architecture evolution](../04-llm/02-llm-architecture-evolution.md), [03 Attention and positional encoding](../04-llm/03-attention-and-position-encoding.md), [05 Scaling Law](../04-llm/05-scaling-laws.md), [07 Distributed training](../04-llm/07-distributed-training.md)
- Fine-tuning and alignment: [06 Fine-tuning and PEFT](../04-llm/06-fine-tuning-and-peft.md), [03 The full RLHF pipeline](../03-reinforcement-learning/05-rlhf-pipeline.md), [03 GRPO and reasoning RL](../03-reinforcement-learning/07-grpo-and-reasoning-rl.md)
- Applications and evaluation: [08 Inference optimization and deployment](../04-llm/08-inference-optimization-and-serving.md), [09 Quantization and compression](../04-llm/09-quantization-and-compression.md), [10 Evaluation and hallucination](../04-llm/10-evaluation-and-hallucination.md), [11 RAG](../04-llm/11-rag.md), [12 Prompt engineering](../04-llm/12-prompt-engineering.md)
- Agents: [05 Agent basics and architecture](../05-agents/01-agent-fundamentals-and-architecture.md), [05 Engineering evaluation and safety](../05-agents/07-evaluation-and-safety.md)
