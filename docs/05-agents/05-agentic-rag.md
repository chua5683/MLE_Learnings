# Agentic RAG and knowledge augmentation

> **One-line summary**: Agentic RAG turns retrieval from a fixed step in the pipeline into a capability the agent **can call, judge, and retry**: the model decides what to look up, how many times, whether the evidence is enough, and whether to switch tools. It complements the indexing, retrieval, and reranking in [RAG](../04-llm/11-rag.md); this note covers only **orchestration and control flow**.
>
> **Keywords**: `agentic RAG` `query planning` `multi-hop` `Self-RAG` `CRAG` `GraphRAG` `RRF` `citation`

## 1. Core concepts

### 1.1 Three generations of RAG

| Generation | Control flow | Typical approach | Limitation |
|---|---|---|---|
| Naive RAG | One retrieval, one generation | Vector Top-K plus prompt assembly | Wrong retrieval means a wrong answer with no second look |
| Advanced RAG | Fixed augmentation pipeline | Query rewriting, hybrid retrieval, reranking, context compression | Hard-coded pipeline; multi-hop questions still need manual design |
| Agentic RAG | The model decides the retrieval strategy | Plan sub-questions, iterate retrieval, self-assess evidence, switch tools | Higher cost and latency; needs termination conditions |

The key difference is not "more sophisticated retrieval" but that **the model owns the control flow**: retrieval becomes one tool in the action space, alongside SQL queries, web search, and code execution.

### 1.2 What it actually fixes

| Failure case | Plain RAG | Agentic RAG |
|---|---|---|
| Multi-hop question (which project does A's manager own) | One retrieval cannot collect it all, so it answers wrong | Look up A's manager first, then the project |
| Query wording differs from the documents | Retrieval misses | Rewrite, expand, or try HyDE several times |
| Retrieved results are irrelevant | Plows ahead and generates, so it hallucinates | Judges "insufficient evidence" and changes the query or admits ignorance |
| Information spread across systems | Can only query the vector store | Calls SQL, APIs, or the web as needed |
| Needs to compare several sources | Sees only one | Collects across rounds and cross-checks |

### 1.3 When agentic retrieval is worth it

**Worth it**: open-ended questions (research, analysis, cross-document comparison), multi-hop needs, heterogeneous sources, unstable recall with room to improve, and tolerance for cost.
**Not worth it**: single-hop factual QA (FAQs, return policies), latency-sensitive paths (each extra round adds 1-3 seconds), deterministic flows, or retrieval that is already solid.

> Rule of thumb: **"can one retrieval probably collect all the evidence?"** If yes, do not make it agentic.

## 2. Principles and derivations

### 2.1 Formalization

Agentic RAG is a decision loop carrying evidence state:

$$
s_t = (q,\ \mathcal{E}_t,\ b_t),\qquad a_t \sim \pi_\theta(\cdot \mid s_t),\qquad
\mathcal{E}_{t+1} = \mathcal{E}_t \cup \mathrm{retrieve}(a_t)
$$

where $q$ is the original question, $\mathcal{E}_t$ is the evidence collected so far, and $b_t$ is the remaining budget. The action space usually includes: `decompose` (split into sub-questions), `rewrite` (rewrite the query), `retrieve`, `call_tool` (structured query), `verify` (check the evidence), and `answer` / `abstain`.

Termination conditions must be explicit: evidence sufficient (every sub-question has supporting passages), budget exhausted, two consecutive rounds with no new useful evidence, or a verifier passes.

### 2.2 Query planning

| Technique | How it works | When it fits |
|---|---|---|
| Query decomposition | Split a compound question into sub-questions (can run in parallel) | Multi-hop, comparison |
| Query rewriting | Remove colloquialisms, resolve references, add domain terms | Sloppy user phrasing |
| HyDE | Have the model write a hypothetical answer and retrieve with its embedding | Short questions, terminology mismatch |
| Step-back | Ask a more abstract background question first, then the specific one | Needs conceptual background to answer |
| Multi-query expansion | Generate 3-5 variants of one question and fuse the results | Unstable recall |

Engineering reminder: **more decomposition is not better**. Doubling the sub-questions doubles retrieval calls and tokens, and the sub-questions often overlap; in practice cap it at 3-5 and deduplicate the results.

### 2.3 Iterative retrieval and judging evidence sufficiency

Iterative retrieval (IRCoT / Self-Ask style): each round derives the next question from the evidence collected so far until the question can be answered.

The hard part is **deciding when to stop**. Usable signals:

1. **Coverage**: does every sub-question have a supporting passage (NLI or an LLM can judge entailment)?
2. **Marginal gain**: does the new round bring new entities or facts (all duplicates means stop)?
3. **Consistency**: do sources contradict each other (if so, look for authoritative sources instead of piling up more)?
4. **Budget**: a hard cap as the backstop.

### 2.4 Self-RAG and CRAG: letting the model judge whether to retrieve and whether retrieval was right

| Method | Mechanism | Key point |
|---|---|---|
| Self-RAG | Train the model to emit **reflection tokens**: whether to retrieve, whether a passage is relevant (ISREL), whether it is supported by the passage (ISSUP), whether the answer is useful (ISUSE) | Internalizes the judging into the model and retrieves on demand; needs dedicated training data |
| CRAG | Bolt on a **retrieval evaluator** that classifies results as correct / incorrect / ambiguous | Correct goes to close reading, incorrect triggers web search or rewriting, ambiguous goes to fusion; easier to ship |
| Common practice | Use an LLM judge with a threshold and retries | Works without training, but control judge cost and bias |

Interview framing: **Self-RAG internalizes through training, CRAG externalizes as a pipeline step**; products usually ship the latter first and consider the former later.

### 2.5 Multi-source retrieval and fusion

Scores from different sources are not directly comparable (vector similarity, BM25 scores, and SQL results have no common scale), so **RRF (Reciprocal Rank Fusion)** fuses ranks only:

$$
\mathrm{RRF}(d) = \sum_{i} \frac{1}{k + \mathrm{rank}_i(d)}, \qquad k \approx 60
$$

Strengths: no normalization needed and robust to outlier scores. Weaknesses: it discards the score gap (the distance between first and second place is flattened).

Design point for tool-based retrieval: **every tool returns structured "evidence units"** (content + source + timestamp + trust level + permission tags). Only a uniform format can be fused and cited.

### 2.6 GraphRAG: using a graph for global questions

- **Build**: extract entities and relations → build the graph → cluster hierarchically with community detection (such as Leiden) → summarize each community.
- **Query**: **local search** expands the neighborhood from an entity (good for "details about this entity"); **global search** answers theme-level questions from community summaries (map-reduce style).
- **Value**: it answers questions vector retrieval cannot, such as "what are these documents mainly about", and the relations are explainable.
- **Cost**: LLM cost for graph construction and summaries is high, incremental updates are awkward, and entity alignment is error-prone. **Do not add GraphRAG just to show off.**

### 2.7 Enterprise knowledge base checklist

| Dimension | Key point |
|---|---|
| Permissions | ACL must be a hard filter at the **retrieval layer** (not post-generation trimming), otherwise the model has already "seen" content it may not access |
| Incremental | A document update must be able to rebuild its index chunks (content hash plus version), so answers never come from the old version |
| Multi-tenant | Partition indexes or namespaces; never return results across tenants |
| Citations | Answers carry a document ID plus a paragraph anchor, clickable back to the source |
| Caching | The query-to-answer cache key must include permissions and version, so one user never hits another's cache |
| Fallback | When retrieval is empty, say "the knowledge base has no such information" instead of relying on model memory |
| Evaluation | Recall, citation accuracy, abstention correctness, end-to-end task success rate |

### 2.8 Long context vs RAG

| Dimension | Long context | RAG |
|---|---|---|
| Cost | Full tokens every turn, growing linearly with material | Only Top-K injected, cheap |
| Recall | Everything is inside, but lost in the middle applies | Depends on retrieval quality, can miss |
| Freshness | Static (you must reload it) | Incrementally updatable |
| Permissions | Hard to apply ACLs per fragment | Can filter at the retrieval layer |
| Best for | Analyzing one long document in a single pass | Large, dynamic, multi-tenant knowledge bases |

Conclusion: **they are not substitutes**. A common combination is "retrieval narrows the scope, long context holds the full chapter".

## 3. Implementation notes

```python
def agentic_rag(question, retriever, tools, llm, max_rounds=4):
    evidence, queries = [], [question]
    for _ in range(max_rounds):
        new = []
        for q in queries:                      # sub-questions can be retrieved in parallel
            hits = retriever.search(q, k=6, filters={"acl": user_acl()})
            new += [h for h in hits if h.id not in {e.id for e in evidence}]
        evidence += new
        if not new:                            # no new evidence -> stop or change course
            break
        verdict = llm.judge_sufficiency(question, evidence)   # coverage + consistency
        if verdict.enough:
            break
        queries = llm.next_queries(question, evidence)[:3]    # 3 to 5 sub-questions
    if not evidence:
        return {"answer": "no supporting evidence found in the knowledge base", "citations": []}
    return llm.answer_with_citations(question, evidence)      # citations are mandatory
```

Three guardrails you must add: **deduplicate retrieved results** (or the same passage enters the context repeatedly), **count the judge against the budget**, and **require citations at answer time with abstention when there are none**.

## 4. Interview Q&A

### Q1. What is the essential difference between Agentic RAG and Advanced RAG?
**Answer**: Who owns the control flow. Advanced RAG is a **fixed pipeline** (rewrite → retrieve → rerank → generate) with the steps and their counts hard-coded. Agentic RAG makes retrieval a **tool** and lets the model decide from the current evidence whether to retrieve, what to retrieve, how many times, and whether to switch tools. That handles multi-hop and multi-source questions, at the cost of higher cost, latency, and variance.
**Follow-up**: When is it unnecessary? Single-hop FAQ where one retrieval suffices; adding an agent just multiplies latency and cost.

### Q2. How do you decide whether a scenario deserves agentic retrieval?
**Answer**: Three criteria: (1) **can one retrieval collect the evidence** (only if not do you need multiple rounds); (2) **is the question open-ended** (multi-hop, comparison, research); (3) **failure cost versus cost** (worth it only when wrong answers are expensive and the budget allows). Also check the data foundation: if recall is already poor, fix indexing and embeddings before adding an agent loop.
**Follow-up**: Is there a quantitative standard? Compare plain RAG with Agentic on an offline set: success-rate gain versus token increase, that is, the cost of each point of improvement.

### Q3. How do you handle multi-hop questions?
**Answer**: The essence is a dependency chain of intermediate entities or answers. Approach: (1) decompose the query and identify sub-questions plus their dependency order; (2) iterate serially, using the previous hop's result to build the next query (A's manager → the project that manager owns); (3) ground every hop in evidence with citations, so the model does not invent intermediate answers from memory; (4) cap the number of hops (usually 2-4) and beyond that give a partial answer plus what is missing.
**Follow-up**: Can hops run in parallel? Independent sub-questions can; dependent ones must be serial, or you fetch the wrong intermediate entity.

### Q4. Self-RAG versus CRAG, and which do you pick in a product?
**Answer**: Self-RAG trains the model to output reflection tokens for whether to retrieve, whether passages are relevant, whether they support the answer, and whether the answer is useful; at inference the tokens drive behavior. It is end-to-end and retrieves on demand but needs dedicated training. CRAG bolts on a retrieval evaluator that sorts results into correct / incorrect / ambiguous and triggers different actions (close reading, re-query, fusion); it ships easily and is observable, but the evaluator itself errs. Products usually start with the CRAG-style external layer and internalize later.
**Follow-up**: How do you control evaluator error? Calibrate the thresholds on a labeled set and keep an exit for "low confidence → abstain or hand off".

### Q5. What if retrieval returns nothing or the wrong thing?
**Answer**: Layered fallbacks: (1) empty retrieval → change the query (rewrite, relax filters, switch index) and retry; (2) results but low relevance → rerank and threshold, treating anything below threshold as no evidence; (3) contradictory evidence → prefer authoritative and latest sources and flag the disagreement in the answer; (4) still insufficient → **abstain explicitly and say what is missing**, never fabricate from model memory. Saying "I do not know" beats inventing an answer.
**Follow-up**: How do you measure fallback quality? Correct abstention rate (it refused when it should) and false-refusal rate (it refused though it could answer).

### Q6. Is GraphRAG worth it?
**Answer**: Only for two needs: (1) answering **global or thematic** questions (what are these documents about, what are the main risks), which vector retrieval cannot do; (2) **multi-hop relational reasoning** over a stable structure (org charts, supply chains, legal references). The cost is high (graph construction and community summaries both need LLM calls) and incremental updates are painful. Most enterprise knowledge bases do fine with hybrid retrieval plus reranking plus agentic follow-up queries.
**Follow-up**: What order of magnitude? At build time every chunk takes at least one LLM extraction, plus a summary per community, which is usually the most expensive part.

### Q7. How do you guarantee citations and reduce hallucination?
**Answer**: Three things: (1) **force citations during generation** (every conclusion carries a chunk ID, and sentences without a citation are not allowed); (2) **verify after generation** (entailment check per sentence: is it supported by the cited passage? If not, delete or rewrite it); (3) **an abstention mechanism** (say plainly that the evidence is insufficient). Add clickable links back to the source so users can verify.
**Follow-up**: How do you test citation alignment? Build an answer-passage annotation set and measure citation precision and coverage.

### Q8. Which layer should permission filtering live in?
**Answer**: The **retrieval layer** (attach the ACL filter to the query), not post-generation trimming. If you trim after generation, the model has already seen unauthorized content and may leak it into the answer, which is a data breach. Implementation-wise, make permissions a hard metadata filter in the vector store, or partition indexes by tenant or department; the cache key must include permissions so user A never hits user B's cached answer.
**Follow-up**: What about fine-grained (paragraph-level) permissions? Attach permission tags to chunks, filter at retrieval, and trim paragraph-level content for partially visible documents.

### Q9. How do you evaluate Agentic RAG?
**Answer**: Layer by layer: (1) **retrieval**: recall and hit rate on constructed question-evidence pairs, plus multi-hop chain hit rate; (2) **generation**: faithfulness (citation support rate), citation accuracy, abstention correctness; (3) **end to end**: task success rate, average retrieval rounds, tokens and latency per question. Also run an **ablation**: how much do you lose without the agent loop? That is how you justify the complexity.
**Follow-up**: What if there are no labels? Use LLM-as-judge for faithfulness (calibrated) and hand-label a batch of online failures.

### Q10. As long contexts improve, will Agentic RAG disappear?
**Answer**: No, but its shape will change. Long context solves "how much can I see at once"; RAG solves "which parts should I look at, how do I keep them fresh, and how do I enforce permissions". Enterprise data dwarfs any window, changes constantly, and has complex permissions, so retrieval stays necessary. The trend combines **retrieval to narrow the scope, long context for deep reading, and the agent choosing the strategy**.
**Follow-up**: Where is long context better? Deep analysis of a single long document (a contract, a paper); loading it whole loses less context than chunk-and-retrieve.

## 5. Pitfalls and follow-ups

- **Agentic for its own sake**: multi-round retrieval on single-hop QA doubles cost and latency for nothing.
- **No stop condition**: "let me check once more" until the budget runs out and the answer is still missing.
- **Sub-questions without grounding**: intermediate answers come from model memory, so everything downstream is wrong.
- **Permission filtering after generation**: the data has already leaked.
- **No deduplication**: the same passage keeps entering the context and squeezes out other evidence.
- **Treating GraphRAG as a silver bullet**: build cost is high, maintenance is hard, and many scenarios gain little.
- **Follow-up trap**: "What is your RAG accuracy?" Break it into recall, faithfulness, abstention correctness, and end-to-end success rate.

## 6. Cheat sheet

| Question | One-line answer |
|---|---|
| Essence of Agentic RAG | The model owns the control flow: what to look up, how many times |
| When it is worth it | Multi-hop, multi-source, open-ended questions, expensive mistakes |
| When it is not | Single-hop FAQ, latency sensitive, retrieval already solid |
| Query planning | Decompose, rewrite, HyDE, step-back; cap at 3-5 sub-questions |
| Stop criteria | Coverage plus marginal gain plus consistency plus budget |
| Self-RAG vs CRAG | Internalized reflection tokens vs an external evaluator with three buckets |
| Fusion | RRF, ranks only, no scores ($k\approx 60$) |
| GraphRAG | Handles global and multi-hop questions, expensive, do not overuse |
| Four enterprise musts | Push permissions down, incremental indexing, forced citations, abstention fallback |

**Mnemonic**: "**Fix retrieval before orchestrating; if one pass collects everything, do not loop; answers must carry evidence, and say so when evidence is thin.**"

## 7. References

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- [Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT)](https://arxiv.org/abs/2212.10509)
- [RAG (Chapter 4)](../04-llm/11-rag.md)
- [AI Agent Interview Guide · RAG techniques](https://github.com/bcefghj/ai-agent-interview-guide)
