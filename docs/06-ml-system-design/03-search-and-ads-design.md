# Search and ads design

> **One-line summary**: Search is "a user arrives with an explicit intent and wants an answer"; ads are "finding the best impression opportunity for advertisers within the traffic". Both share the retrieval-ranking skeleton, but **search is about relevance while ads are about calibrated eCPM and ecosystem balance**.
>
> **Keywords**: `query understanding` `BM25` `dense retrieval` `LTR` `pCTR` `pCVR` `ESMM` `calibration` `eCPM` `GSP`

## 1. Clarifying requirements

This note ties together two prompts: **e-commerce search** (query → product ranking) and **search ads** (insert ads into search results and run an auction).

| Dimension | E-commerce search | Search ads |
| --- | --- | --- |
| User intent | Explicit; wants to "buy" | Passive; can be influenced by ads |
| Optimization goal | Relevance + conversion | Platform revenue + advertiser ROI + user experience |
| Key constraints | 200 ms, no-result rate, category coverage | 100 ms, budget/bid, ad density, compliance |
| Main difficulty | Query understanding, relevance, long tail | Prediction calibration, auction mechanics, cold start |

Clarification checklist: item count (~1e8), query volume (tens of millions per day), DAU/QPS, latency budget, whether there is ad inventory, billing model (CPC/CPM/oCPC), and whether personalization is needed.

## 2. Goals and metrics

| Layer | Search | Ads |
| --- | --- | --- |
| North star | GMV, conversion rate, search return visits | Platform revenue (or GMV), advertiser ROI |
| Online proxy | CTR, CVR, no-result rate, first-click position, post-rewrite CTR | CTR, CVR, eCPM, budget burn rate |
| Offline proxy | NDCG@10, MRR, Recall@k, relevance-grade accuracy | logloss, AUC, PCOC (calibration quality) |
| Guardrails | Latency, bounce rate, query coverage | Ad density, user complaints, share of impressions for small advertisers |

## 3. Data and labels

### 3.1 Search

- **Relevance labels**: human-annotated on 5 grades (perfect/excellent/fair/poor/irrelevant). Expensive, so use **distillation** to expand to millions of query-doc pairs; online behavior (clicks, dwell time, bounces, next query) serves as weak labels.
- **Samples**: one sample per query-impression; negatives are "impressed but not clicked", but watch for **position bias** and the truncation bias that "users only look at the top three".
- **Query-side data**: query logs, rewrite logs, category mappings, synonym tables.

### 3.2 Ads

- **Sample space**: impression → click → conversion, with conversion delays (hours to days) that require delayed-feedback modeling or a waiting window.
- **Class imbalance**: CTR is typically 1%–10%, CVR may be 0.1%–1%; focus on PR-AUC and calibration instead of only AUC.
- **Label trustworthiness**: a click is a weak signal of "interest"; conversion is closer to the business goal. Fraudulent clicks must be cleaned.
- **Compliance**: user profiling requires consent, and ad creatives require review.

## 4. Features

| Group | Search | Ads |
| --- | --- | --- |
| Query | Tokenization, term weights, intent, spelling correction, rewriting, synonym expansion | Keywords, match type, commercial value of the query |
| Doc | Title/attributes/category, sales, rating, price, text-image embeddings | Creative, landing page, category, historical CTR/CVR |
| Cross | Query-doc match (BM25, semantic similarity, category match) | Query-ad relevance, keyword match |
| User/context | Region, history, device, time | User profile, ad fatigue, frequency caps |
| Statistics | CTR, CVR, impression counts (smoothed) | Advertiser budget, bids, account history |

## 5. Model

### 5.1 The full search pipeline

```mermaid
flowchart LR
    Q["User query"] --> QU["Query understanding<br/>spelling·tokenization·term weights·intent·rewrite"]
    QU --> R1["Retrieval<br/>inverted index BM25 / dense / category / personalized"]
    R1 --> R2["Coarse ranking 1e3"]
    R2 --> R3["Ranking 1e2<br/>relevance + quality + personalization"]
    R3 --> ADS["Ad insertion and auction"]
    ADS --> RR["Re-ranking<br/>diversity·diversify·business rules"]
    RR --> OUT["Result page"]
```

**Query understanding** (sets the retrieval ceiling):
- Spelling correction (noisy channel/seq2seq), tokenization and POS tagging, term weights (learned offline from click logs), synonym expansion, intent classification (category/brand/model), entity recognition, term dropping, and query rewriting.
- Example: "famous hot springs in Beijing" → drop "in" → category intent → first retrieval returns nothing → drop the modifier "famous" → second retrieval succeeds.

**Retrieval**:
- Inverted index + BM25 guarantees exact matching:

$$
\text{BM25}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\cdot|d|/\text{avgdl})}
$$

- Dense retrieval (two-tower/embeddings) adds semantic generalization; category/attribute retrieval adds structural constraints; personalized retrieval adds user preference. Merge and deduplicate after running the channels in parallel.

**Ranking**:
- Objective = relevance × quality × personalization (combined via multi-task learning or a weighted sum).
- Three LTR paradigms: pointwise (regression/classification; simple and easy to calibrate), pairwise (AUC-style loss; suits clear positive/negative boundaries), and listwise (optimizes NDCG directly, e.g., LambdaRank).
- Models: GBDT+LR → DeepFM/DIN → two-tower + interaction layer; ranking usually uses early fusion.

### 5.2 Ads: prediction + calibration + auction

**Funnel**: targeting (audience/region/interest) → retrieval (keywords/dense/similar ads) → coarse ranking → ranking (pCTR × pCVR) → auction and pacing.

**eCPM ranking** (the billing model determines the formula):

$$
\text{eCPM}_{\text{CPC}}=1000\times pCTR\times bid_{CPC}
$$

$$
\text{eCPM}_{\text{oCPC}}=1000\times pCTR\times pCVR\times bid_{CPA}
$$

**Calibration is the lifeline of ads**: the auction ranks by probability times bid. If pCTR is uniformly 2x too high, the ad order may stay the same but **platform revenue and advertiser ROI are both wrong**. Measure with PCOC (Predicted CTR / Actual CTR, closer to 1 is better); use Platt/Isotonic, calibrated per ad slot and audience bucket.

**ESMM**: conversion samples are extremely sparse and exist only among clicked samples, so training CVR directly on clicked samples suffers from **sample selection bias (SSB)**. ESMM trains two tasks over the full impression space:

$$
pCTCVR=pCTR\times pCVR
$$

pCTR and pCVR share embeddings, and pCVR is learned implicitly through pCTCVR/pCTR — this mitigates SSB and uses all impression data to alleviate data sparsity (DS).

**Auction mechanics**:
- GSP (generalized second price): charge "the next advertiser's eCPM divided by your own pCTR"; VCG charges the marginal contribution to social welfare. VCG is theoretically more truthful but complex to implement and may reduce revenue.
- Budget constraints + pacing: control the burn rate so budgets do not exhaust early.
- Multiple objectives: GMV, ROI, new-customer acquisition, combined via constrained optimization or weights.

### 5.3 Search box suggestions

- Traditional: prefix trie + top-K by popularity + personal history + spelling correction, latency < 50 ms.
- Advanced: semantic retrieval + a lightweight generation model (query completion), with vulgar/sensitive-word filtering and safety checks.
- Key constraint: **every keystroke can trigger a request**, so QPS is 5–10x that of search; you need local caching and prefix aggregation.

## 6. Offline evaluation

- Search: NDCG@10, MRR, Recall@k, relevance-grade accuracy; slice by query frequency (head/long tail).
- Ads: logloss, AUC, PR-AUC, **PCOC calibration**, and error broken down by ad slot/industry.
- Evaluation traps: aggregate metrics hide the long tail; search must be sliced by query type (navigational/informational/transactional).

## 7. Online architecture and capacity

- Search path: gateway → query understanding → parallel multi-channel retrieval (40 ms) → coarse ranking (30 ms) → ranking (60 ms) → re-ranking (20 ms) → render; p99 200 ms.
- Ads path: request → targeting filter → retrieval → coarse ranking → ranking + auction → response; p99 100 ms (shares the budget with search).
- Capacity: e-commerce search with 1e8 DAU → 5e8 queries/day → 6k QPS average, 3x peak ≈ 18k QPS; each request retrieves 1e4 candidates and ranks 200.
- Index: inverted index of 1e8 products × ~100 terms ≈ 1e10 postings, requiring sharding + replicas; vector index of 1e8 × 128 dims ≈ 51 GB (float32).
- Caching: popular query results, suggestions, ad candidates, user profiles.

## 8. Experiments and monitoring

- Split A/B by user; ad experiments must split by **user or cookie** so budgets and frequency caps stay consistent.
- Monitoring: latency quantiles, no-result rate, rewrite rate, CTR/CVR, eCPM, PCOC, ad density, budget burn, complaint rate.
- Drift: query distributions shift sharply during major sales and holidays; backtest on similar historical days in advance.
- Degradation: ranking timeout → coarse-ranking results; ad service down → organic results only; index unavailable → cache fallback.

## 9. Trade-offs

| Choice | Benefit | Cost |
| --- | --- | --- |
| Large-model relevance ranking | Best relevance | High latency and cost; can only rank a few |
| Insert more ads | Higher short-term revenue | Worse user experience, long-term churn |
| Strict calibration | Fair auctions, trustworthy ROI | Calibration maintenance cost; too-fine buckets overfit |
| Higher share of dense retrieval | Better semantic generalization | Weaker exact match, lower explainability |

## 10. Interview Q&A

### Q1. What is the essential difference between search and recommendation?
**Answer**: Search is **question answering** — the user gives an explicit query and the system must return results that satisfy the information need; relevance is a hard constraint. Recommendation is **guessing** — the user did not say what they want, so the system infers interest from behavior and pursues long-term satisfaction. In engineering, search emphasizes query understanding and relevance (few but accurate), while recommendation emphasizes retrieval breadth, diversity, and exploration.
**Follow-up**: Does search need personalization? Yes, but personalization can only reorder within relevance; it must not violate relevance.

### Q2. Which module in query understanding matters most?
**Answer**: **Intent recognition + rewriting**. Intent decides which retrieval channel to use and how relevance is computed; rewriting sets the retrieval ceiling (there is a gap between how users express things and how products are described). Spelling correction and tokenization are infrastructure — errors there cascade downstream.
**Follow-up**: Can rewriting change user intent? Yes, so do "post-rewrite consistency checks + parallel retrieval over multiple rewrites" instead of trusting a single rewrite.

### Q3. Why does retrieval need both inverted index and dense vectors?
**Answer**: Inverted index + BM25 excels at **exact matching** (brands, model numbers, proper nouns) and is explainable and controllable; dense retrieval excels at **semantic generalization** ("noise-cancelling headphones" matches "active noise cancellation Bluetooth earbuds") but may retrieve irrelevant items. They complement each other; merge and deduplicate across channels. Pure dense retrieval fails easily in e-commerce because model numbers and specs must match exactly.
**Follow-up**: How do you fuse multiple retrieval channels? Normalize scores and take a weighted sum, or take each channel's top-K and let coarse ranking score them uniformly.

### Q4. Should ranking use pointwise, pairwise, or listwise losses?
**Answer**: Search usually uses pairwise/listwise because the "relevant/irrelevant" boundary is clear and rank position directly affects user experience. Ads usually use pointwise because ads are rarely presented as a continuous list, and pointwise is easier to calibrate (calibration matters more than ranking there). Listwise (LambdaRank) directly optimizes NDCG and often performs best, but is more complex to train.
**Follow-up**: Can pointwise optimize ranking? Yes, but the loss is misaligned with the ranking metric, so it needs strong features and enough samples.

### Q5. Why must ads calibrate pCTR? What happens without it?
**Answer**: eCPM = pCTR × bid determines both order and billing. If the model only ranks well (high AUC) but probabilities are uniformly too high: ① bids are overestimated and advertiser ROI suffers; ② platform revenue is distorted under second-price billing; ③ the relative order across ads can also be wrong because different ads have different bias. So ads track **PCOC ≈ 1** and calibrate per bucket with Isotonic/Platt.
**Follow-up**: If ranking is good but calibration is poor, can you apply only a monotonic transform? Yes — isotonic calibration does not change the order, it only fixes the values.

### Q6. How is eCPM computed? What is the difference between GSP and VCG?
**Answer**: CPC billing eCPM = 1000 × pCTR × bid; oCPC billing eCPM = 1000 × pCTR × pCVR × bid; CPM is simply the bid. GSP charges "the next advertiser's eCPM converted by your own pCTR" (second price): simple and high-revenue but not incentive-compatible. VCG charges the welfare loss your participation imposes on others: truthful bidding and optimal social welfare, but complex to compute and possibly lower platform revenue.
**Follow-up**: Which is used in practice? Most platforms use a GSP variant with a quality score and a reserve price, balancing revenue and implementability.

### Q7. What problem does ESMM solve, and why does it work?
**Answer**: It solves **sample selection bias** in CVR prediction (training only on clicked samples while inference must cover the full impression space) and **data sparsity** (very few conversions). ESMM trains pCTR and pCTCVR over the full impression space; with pCTCVR = pCTR × pCVR, pCVR is learned implicitly as a latent factor, which lets it use all impression samples.
**Follow-up**: What is wrong with ESMM's assumption? It assumes click and conversion are conditionally independent given features, which is not true in practice; later work such as ESMM2 and ESCMM addresses this.

### Q8. Why do ads need targeting and coarse ranking?
**Answer**: There can be millions of candidate ads, and the ranking model is heavy and high-latency, so scoring all of them is impossible. Targeting uses audience/region/keywords with rules and inverted indexes to cut candidates to tens of thousands; coarse ranking uses a lightweight model to cut to thousands; ranking cuts to hundreds. Every layer trades quality loss against latency cost, and the core metric for coarse ranking is the "coarse-to-ranking loss rate".
**Follow-up**: Can tight targeting starve delivery? Yes, so allow "relaxed targeting + ranking fallback" plus exploration traffic.

### Q9. How do you design search box suggestions? How do you guarantee latency?
**Answer**: Mine high-frequency queries and prefixes offline and build a trie or KV index; online, fetch top-K by prefix, layer personal history and time-decayed popularity, then add spelling correction and safety filtering. Latency < 50 ms comes from local/edge caching, prefix prefetching, limiting to top-10, and asynchronous logging. Advanced versions add semantic retrieval and a lightweight generation model for the long tail.
**Follow-up**: Why not use a pure generation model? Calling an LLM on every keystroke is unacceptable in both cost and latency, and controllability matters.

### Q10. How do you handle cold start for new ads or new products?
**Answer**: New products: use content features (title/image/category) for semantic retrieval, transfer embeddings from similar products, provide traffic support, and collect feedback quickly. New ads: predict from similar advertisers' profiles and creative features, give guaranteed delivery or exploration traffic during the cold-start period, and control trial cost with bandits; strictly control budget and risk at the same time.
**Follow-up**: Rules or models during cold start? Rules as a fallback plus bandit exploration, switching to models once enough samples are collected.

## 11. Pitfalls and follow-ups

- Equating relevance with "high CTR": a high-CTR clickbait title is not relevant.
- Talking only about models, not mechanics: eCPM, billing, budget, and pacing are the skeleton of an ad system.
- Forgetting calibration and failing "what happens if pCTR is too high?".
- Discussing only dense retrieval and ignoring inverted indexes and exact matching.
- Evaluating without slicing: head queries improve while long-tail queries degrade, and the aggregate may not move.
- Ignoring the mixing constraints between ads and organic results (ad density, relevance threshold).

## 12. Cheat sheet

| Module | Core | One-liner |
| --- | --- | --- |
| Query understanding | Intent + rewriting | Sets the retrieval ceiling |
| Retrieval | Inverted index + dense + category | Exact and semantic complement each other |
| Ranking | LTR + multi-task | Relevance × quality × personalization |
| Ad prediction | pCTR/pCVR + calibration | Calibration is more critical than ranking |
| Auction | eCPM + GSP | Balance revenue and experience |
| Suggestions | Trie + popularity + safety | Latency < 50 ms |

## 13. References

- [MLE Interview · search engine](https://github.com/longxingtan/MLE-interview/blob/main/03_system/03_ml/search_engine.md)
- [MLE Interview · ad click](https://github.com/longxingtan/MLE-interview/blob/main/03_system/03_ml/ad_click.md)
- [Embedding-based Retrieval in Facebook Search](https://arxiv.org/pdf/2006.11632)
- [Entire Space Multi-Task Model (ESMM)](https://arxiv.org/abs/1804.07931)
- [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
- [Ad Click Prediction: a View from the Trenches](https://dl.acm.org/doi/10.1145/2487575.2488200)
- [ML System Design 2.0 · Yuan Meng](https://www.yuan-meng.com/posts/mle_interviews_2.0/)
