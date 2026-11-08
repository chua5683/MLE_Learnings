# System design and infra cheat sheet

> **One-line summary**: Before a design round, memorize the "9-step skeleton + 5 estimation formulas + 1 component checklist"; in the interview, talk along the skeleton, back it with numbers, and close with trade-offs.
>
> **Keywords**: `9-step` `capacity` `funnel` `feature store` `serving` `SLO`

## 1. Nine-step skeleton (from memory)

| # | Step | One-line deliverable |
| --- | --- | --- |
| 1 | Clarify | Four questions on product/scale/latency/constraints + restate |
| 2 | Goals and metrics | North star + online proxy + offline proxy + guardrails |
| 3 | Data and labels | Sources, positive/negative samples, delay, bias, compliance |
| 4 | Features | user/item/context/cross/sequence + freshness |
| 5 | Model | baseline → advanced, explain why to upgrade |
| 6 | Offline evaluation | Metric + time split + leakage prevention |
| 7 | Online architecture | Funnel + cache + degradation + capacity |
| 8 | Experiments and monitoring | Canary, A/B, drift, rollback |
| 9 | Trade-offs | "If ..., then ..., at the cost of ..." |

## 2. Capacity estimation formula card

| Quantity | Formula | Example |
| --- | --- | --- |
| Average QPS | DAU × requests per user ÷ 86400 | 1e8 × 10 → 11.6k |
| Peak QPS | 2–5 × average | ×3 → 35k |
| Candidate funnel | 1e8→1e4→1e3→1e2→1e1 | One order of magnitude per layer |
| Scoring compute | peak QPS × candidates × FLOPs/sample | 35k×300×1e8 ≈ 1 PFLOP/s |
| GPU count | total compute ÷ (per-GPU compute × efficiency) | H100 500T, 30% efficiency → ~10 |
| Storage | impressions/day × sample size × days | 1e9 × 1KB × 90 ≈ 90 TB |
| Embeddings | N × d × 4 B | 1e9 × 128 ≈ 512 GB |
| Concurrency | QPS × latency (Little's law) | 35k × 0.1s = 3500 |
| KV cache | 2 × L × H_kv × d_head × bytes × seq | 70B ≈ 328 KB/token |
| A/B sample size | $n\approx15.7\sigma^2/\Delta^2$ | p=0.05, MDE 1% → 3M per arm |
| Model memory | params × bytes | 70B bf16 ≈ 140 GB |
| LLM cost | (in×price_in + out×price_out)/1e6 | 4k in + 400 out ≈ 0.018 USD |

## 3. Numbers at a glance

| Item | Typical magnitude |
| --- | --- |
| Recommendation p99 | 200 ms (retrieval 35 / coarse 40 / ranking 70 / re-ranking 15) |
| Search p99 | 200 ms (including query understanding) |
| Ads p99 | 100 ms |
| Suggestion p99 | 50 ms |
| Code completion p99 | 100 ms |
| RAG TTFT | 0.5–2 s; TPOT 20–50 ms |
| Single-machine QPS | light model 100–1000; heavy model 10–100 |
| Single-node LLM throughput | 8×H100 / 70B: 2–5k output tok/s |
| User embeddings | 1e8 × 128d × 4B ≈ 51 GB |
| Item embeddings | 1e9 × 128d × 4B ≈ 512 GB (must shard) |

## 4. Component checklist

| Layer | Components | Talking points |
| --- | --- | --- |
| Access | Gateway, rate limiting, auth, canary | Overload protection, degradation switches |
| Features | Feature platform, offline/real-time, KV, point-in-time join | Offline-online consistency, backfill, monitoring |
| Retrieval | Inverted index, vector index (HNSW/IVF-PQ), multi-channel fusion | Sharding, incremental updates, latency |
| Ranking | Coarse ranking, ranking, re-ranking | Distillation, multi-task, DPP/diversification |
| Training | Sample joining, distributed training, model registry | Data versioning, reproducibility, rollback |
| Serving | Online inference, batch inference, GPU/CPU, dynamic batching | Throughput vs latency, autoscaling |
| LLM | Inference engine (vLLM/SGLang), KV cache, quantization, routing | TTFT/TPOT, memory, caching |
| Data | Logs, Kafka, Flink, warehouse, vector store | Latency, cost, replay |
| Monitoring | Metrics, logs, traces, alerts | Feature drift, score drift, quality |
| Experimentation | A/B, stratification, CUPED, SRM checks | Sample size, significance, guardrails |

## 5. Metrics at a glance

| Type | Metrics |
| --- | --- |
| Classification | AUC, PR-AUC, logloss, F1, KS |
| Ranking | NDCG@k, MRR, Recall@k, HR@k, GAUC |
| Calibration | PCOC, ECE, reliability curve |
| Business | CTR, CVR, GMV, watch time, retention, approval rate, default rate |
| Stability | PSI, KS decay, feature missing rate, score distribution |
| LLM | faithfulness, answer relevance, citation accuracy, resolution rate, TTFT/TPOT, cost per request |
| Infra | QPS, p50/p99, error rate, GPU utilization, MFU, cache hit rate |

## 6. Top 12 follow-ups (one-line answers)

1. **Why retrieval + ranking?** Full-corpus ranking is infeasible in compute and latency; use a cheap model to screen and an expensive one to rank.
2. **Why is two-tower fast?** Item vectors are precomputed offline + ANN; online you only compute the user vector.
3. **What if negative sampling is biased?** Subtract $\log p_i$ from the logit, or correct probabilities with $p'/(p'+(1-p')/w)$.
4. **Position bias?** Train with position as a feature and set it to 0 at inference; or use IPW/PAL for unbiased estimation.
5. **Offline improves, online does not?** Check feature consistency, objective misalignment, sample bias, and assignment (SRM).
6. **Why must ads calibrate?** eCPM = pCTR × bid; wrong probabilities mean wrong revenue and ROI; track PCOC.
7. **What does ESMM solve?** CVR's sample selection bias and sparsity, via pCTCVR = pCTR × pCVR trained over the full space.
8. **How to evaluate class imbalance?** Use PR-AUC / Precision@k; do not rely on ROC-AUC alone.
9. **Label delay?** Mature samples + proxy labels + delayed-feedback modeling.
10. **RAG permissions?** ACL filtering at the retrieval layer, not via prompts; cache keys carry tenant/version.
11. **How to cut LLM cost?** Caching, context trimming, small-model routing, quantization, batching.
12. **Scale ×100?** Index/embedding sharding, model distillation, multi-level caching, rate limiting and degradation.

## 7. 45-minute whiteboard script

1. 0–5 min: ask the four clarifying questions, restate assumptions, write down the numbers.
2. 5–9 min: draw the metric pyramid (north star/online/offline/guardrails).
3. 9–17 min: data and features (how samples are built, feature groups, freshness).
4. 17–29 min: draw the funnel and cover retrieval/coarse/ranking/re-ranking models and constraints layer by layer.
5. 29–37 min: online architecture + latency budget + capacity estimation + caching/degradation.
6. 37–41 min: A/B, monitoring, drift, rollback.
7. 41–45 min: two trade-offs + one risk + one next step.

## 8. Common pitfalls checklist

- [ ] Started drawing models without clarifying
- [ ] Talked only about models, not samples/labels
- [ ] Only AUC as a metric, no business metric or guardrails
- [ ] Computed capacity but it changed no architectural decision
- [ ] No degradation or monitoring plan
- [ ] Ignored cold start/position bias/delayed feedback
- [ ] Could not explain real-time vs offline features
- [ ] Did not discuss offline-online consistency
- [ ] No trade-offs, or trade-offs not quantified
- [ ] LLM questions without cost or eval-set discussion

## 9. Related chapters

- Method and capacity: [Answer framework and methodology](../06-ml-system-design/01-framework-and-methodology.md)
- Recommendations: [Recommendation system design](../06-ml-system-design/02-recommender-system-design.md)
- Search and ads: [Search and ads design](../06-ml-system-design/03-search-and-ads-design.md)
- Risk and finance: [Risk and finance design](../06-ml-system-design/04-risk-and-finance-design.md)
- LLM applications: [LLM application system design](../06-ml-system-design/05-llm-application-design.md)
- Cases: [Case library](../06-ml-system-design/06-case-studies.md)
- Infra details: [ML Infra overview](../07-ml-infra/01-ml-infra-overview.md), [Feature platform](../07-ml-infra/02-feature-store-and-realtime-features.md), [Inference serving](../07-ml-infra/04-inference-serving-and-gpu-optimization.md)
- Evaluation metrics: [Evaluation metrics](../01-ml-fundamentals/05-evaluation-metrics.md)
