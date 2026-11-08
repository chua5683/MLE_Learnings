# Memory systems

> **One-line summary**: An agent's memory is the whole set of mechanisms for "putting the information most useful for the current decision into the context within a token budget". **The context window is RAM and external storage is disk**; the hard part is what to write, when to retrieve, and when to forget.
>
> **Keywords**: `working memory` `long-term memory` `episodic` `semantic` `context compression` `MemGPT` `memory decay` `privacy`

## 1. Core concepts

### 1.1 Why memory is needed

An agent without memory is a "stateless function": every request feels like the first meeting. It breaks in four situations:

| Situation | Symptom without memory |
|---|---|
| Multi-turn dialogue | Repeated questions, contradictions, forgetting options the user already rejected |
| Long tasks | Forgetting completed subgoals, redoing a step, losing intermediate artifacts |
| Personalization | Re-explaining preferences, terminology, and taboos every time |
| Cost | Stuffing the full history into the prompt every turn, so tokens explode |

### 1.2 Layered memory

```mermaid
flowchart TB
    subgraph ctx [Context window (RAM)]
        W[Working memory<br/>current task instruction + recent turns + tool results]
        S[Session summary<br/>key points of compressed history]
    end
    subgraph ext [External storage (disk)]
        L1[(Long-term memory store<br/>vector / relational / graph)]
        L2[(Documents and knowledge base<br/>RAG index)]
        L3[(State and artifacts<br/>files / DB / checkpoint)]
    end
    W -->|compress when over budget| S
    S -->|write key points| L1
    L1 -->|retrieve and inject on demand| W
    L2 -->|retrieve and inject| W
    L3 <-->|read and write progress| W
```

| Layer | Lifetime | What it stores | Typical implementation |
|---|---|---|---|
| Working memory | One task / session | Instructions, recent turns, tool results | Context window |
| Short-term memory | Within a session | Compressed history, key facts | Summary plus sliding window |
| Long-term memory | Across sessions | User preferences, domain knowledge, lessons learned | Vector store / KV / graph DB |
| External memory | Persistent | Documents, codebase, task artifacts | RAG index, filesystem |

### 1.3 By content type (a cognitive-science analogy)

| Type | Meaning | Example inside an agent |
|---|---|---|
| Episodic | "What happened, when, and where" | The story of last time's failure, a user complaint record |
| Semantic | Abstract facts and concepts | "This customer is in financial services", "the API requires OAuth" |
| Procedural | How to do something | Tool-use patterns, SOPs, templates of successful plans |

Engineering implication: **episodic memory fits "retrieve then review", semantic memory fits "assemble into a profile", and procedural memory fits "inject directly as a system-prompt skill"**. The read/write paths differ, so do not cram all three into one table.

## 2. Principles and derivations

### 2.1 Allocating the context budget

Treat the context window as a fixed budget $B$ (tokens) split four ways:

$$
B = B_{\text{sys}} + B_{\text{mem}} + B_{\text{hist}} + B_{\text{out}}
$$

- $B_{\text{sys}}$: system prompt and tool schemas (often underestimated; many tools can eat several thousand tokens);
- $B_{\text{mem}}$: retrieved memories and knowledge;
- $B_{\text{hist}}$: dialogue history and tool results;
- $B_{\text{out}}$: reserved for the output (including the reasoning chain).

In practice: **fix $B_{\text{sys}}$ and $B_{\text{out}}$ first, then allocate the remainder dynamically by task type**; cap each individual tool result, and organize history as "the last N turns plus a summary".

### 2.2 Summaries and compression

| Strategy | How it works | Strengths | Weaknesses |
|---|---|---|---|
| Incremental summary | Every k turns, compress the earliest turns into a summary and roll it forward | Cheap, sustainable | Repeated compression distorts; early detail is lost |
| Full rewrite | Rewrite the summary from the whole history each time | More complete, clearer themes | Cost grows with conversation length and the summary can drift |

A compromise is **layered summaries**: keep bullet points per turn, re-summarize the bullets every k turns to form a pyramid, and persist the raw conversation so you can go back (summaries for retrieval, raw text for evidence).

Higher compression is not automatically better: the goal is "does the follow-up task still need this detail". Judge by **answerability** (can the summary answer later questions), not by the compression ratio.

### 2.3 Scoring retrieval over vector memory

Following Generative Agents, the retrieval score weights **relevance, recency, and importance**:

$$
\mathrm{score}(m \mid q) = w_{\text{rel}}\,\widehat{\mathrm{rel}}(m,q) + w_{\text{rec}}\,\widehat{\mathrm{recency}}(m) + w_{\text{imp}}\,\widehat{\mathrm{importance}}(m)
$$

where:

- $\widehat{\mathrm{rel}}$: cosine similarity between query and memory, mapped to $[0,1]$;
- $\widehat{\mathrm{recency}}$: exponential decay $\exp(-\lambda \Delta t)$, where $\Delta t$ is the time since last access;
- $\widehat{\mathrm{importance}}$: a model- or rule-assigned importance score (for example a normalized 1-10);
- the weights must be tuned per business and normalized, otherwise different scales suppress each other.

**Common problems with pure similarity retrieval**: it recalls memories that "look alike but are irrelevant", while exact identifiers (order numbers, account IDs) are actually better served by keywords or a relational store. So production usually does **hybrid retrieval (BM25 plus vectors) + metadata filtering (user, time, type) + reranking**.

### 2.4 Write policy: when to remember

Not everything deserves storage. Common criteria:

1. **Information gain**: not a duplicate of existing memories and useful in the future (preferences, stable facts, lessons learned);
2. **Importance threshold**: write only when the model's score exceeds a threshold;
3. **Explicit instruction**: "remember that I am allergic to peanuts" must be written with high priority;
4. **Post-task retrospective**: distill success or failure experience into reusable entries.

Writes **must carry metadata**: `user_id`, timestamp, source, confidence, type. Otherwise you cannot isolate, forget, or trace anything later.

### 2.5 Forgetting and consolidation

- **Time decay**: the longer a memory goes unused, the lower its weight (archive rather than delete);
- **Access reinforcement**: memories that are successfully recalled and used gain weight (like cache hotness);
- **Conflict merging**: when a new memory contradicts an old one (the user changed a preference), overwrite by time or keep multiple versions with confidence scores;
- **Offline consolidation**: a periodic job merges scattered episodic memories into higher-level semantic memory ("this user likes concise answers").

> In regulated domains (healthcare, finance), "forgetting" is mostly **archiving plus an audit trail**, together with the ability for users to export or delete their data (the right to be forgotten).

### 2.6 MemGPT: managing the context like RAM

Core idea: the main context is limited, external storage acts as disk, and content is **paged in and out through function calls**:

| Analogy | MemGPT mechanism |
|---|---|
| RAM | Main context (system prompt plus recent messages) |
| Disk | External memory (archival store, vector DB) |
| Page fault | When the context nears capacity, an "eviction" writes less important content out and removes it |
| Page in | A retrieval function loads content back into the context when needed |

Unlike "passively running RAG once per turn", MemGPT emphasizes **active management of the memory substrate**: the model decides when to page and when to recall, which requires function-calling ability and an explicit control protocol.

### 2.7 Memory middleware and knowledge graphs

- **Mem0-style components**: package "extract reusable memories from dialogue → update → retrieve" as a reusable layer, often combined with a vector store, a graph structure, and a user profile; the focus is engineering the **write pipeline and conflict handling**.
- **Knowledge-graph memory**: entity-relation-fact triples, good for multi-hop queries and explainable reasoning ("A reports to B, and B owns project X"); the price is build cost and entity alignment, so it rarely pays off for unstructured chit-chat.
- **Selection rule of thumb**: **preferences and facts → KV or a profile table; experience and documents → vector store; multi-hop relations → knowledge graph; task progress → files or a database.**

### 2.8 Privacy and isolation

| Risk | Measure |
|---|---|
| Cross-user leakage | Force `user_id` / `tenant_id` filters on every read and write; no global retrieval |
| Sensitive data entering memory | Run PII detection and redaction before writing; never store secrets |
| Memory poisoning | Separate trust levels for "user input" and "tool or web content"; low-trust content is never auto-written to long-term memory |
| No deletion capability | Provide per-user and per-item deletion plus export, and keep an audit log |

## 3. Implementation notes

A minimal skeleton for memory writes and retrieval scoring:

```python
import math, time

def add_memory(store, user_id, text, importance, kind="semantic"):
    # 1) importance threshold + dedup (similarity > 0.95 counts as a duplicate)
    if importance < 0.4 or store.has_similar(user_id, text, thr=0.95):
        return None
    return store.insert(user_id=user_id, text=text, kind=kind,
                        importance=importance, created_at=time.time(),
                        source="conversation")

def recall(store, user_id, query, k=6, lam=2e-6, w=(0.6, 0.2, 0.2)):
    w_rel, w_rec, w_imp = w
    now, scored = time.time(), []
    for m in store.candidates(user_id=user_id, since_days=180):   # metadata prefilter
        rel = store.cosine(query, m.text)                        # relevance
        rec = math.exp(-lam * (now - m.last_accessed))           # recency
        score = w_rel * rel + w_rec * rec + w_imp * m.importance
        scored.append((score, m))
    scored.sort(key=lambda x: -x[0])
    hits = scored[:k]
    store.touch([m.id for _, m in hits])                         # access reinforcement
    return hits
```

Three things easy to miss: **prefilter by metadata before computing similarity** (faster and guarantees isolation), **touch the timestamp after recall** (implements access reinforcement), and **return source and timestamp** (so answers can cite and judge freshness).

## 4. Interview Q&A

### Q1. With long contexts (128k / 1M), do we still need external memory?
**Answer**: Yes, for three reasons. (1) **Cost and latency**: stuffing the full history in every turn makes token cost grow linearly or even quadratically with turns. (2) **Attention dilution**: long contexts suffer from lost in the middle, so key facts in the middle get ignored. (3) **Cross-session, cross-task continuity**: a context window only lives for one request, while preferences and experience must persist. Long context makes "retrieve less" feasible but does not replace **writing, decay, and isolation**.
**Follow-up**: Then what is the low-effort use of long context? Put all material for the current task in at once, and leave cross-session information to external memory.

### Q2. Does a vector store count as a memory system?
**Answer**: It is **one implementation of long-term memory**, but incomplete. A full memory system also needs a write policy (what is worth remembering), importance and time weighting for ranking, conflict handling and updates (what if the user changes a preference), forgetting and archiving, and user isolation plus compliant deletion. With only "chunk the text and search by similarity" you hit two traps: recalling superficially similar but irrelevant items, and failing to match exact IDs.
**Follow-up**: What should be added? Hybrid retrieval (BM25 plus vectors), metadata filtering, and reranking.

### Q3. Do summaries lose critical information? How do you trade it off?
**Answer**: They always lose something; the question is whether it is detail the follow-up task still needs. Approach: (1) layered summaries (bullets then higher-level summaries) with the raw conversation persisted for backtracking; (2) extract **structured key facts** (order numbers, dates, preferences) into dedicated fields that are never compressed away; (3) evaluate compression by answerability (can the summary answer later questions). Compression ratio is not the KPI; task success rate is.
**Follow-up**: How do you prevent summary drift? A fixed template, incremental appends only, and periodic rebuilds from the raw text.

### Q4. How should memories be updated when a user changes a preference?
**Answer**: Give memory items a primary key and a version. On write, first search for an existing memory on the same topic; if found, **update** it (new value, updated_at, higher confidence) instead of appending forever. Three conflict policies: latest wins, keep multiple versions ordered by time, or ask the user. For preferences, "latest wins plus keep the history" is best: correct behavior with an audit trail.
**Follow-up**: How do you detect conflicts? Attach a topic tag during extraction (for example diet); the same tag with different values is a conflict.

### Q5. What factors should memory retrieval ranking consider?
**Answer**: Three factors: **relevance** (vector similarity), **recency** (exponential decay), and **importance** (model score or rules). Then add business constraints: metadata filters (user, time range, type), diversity (avoid a pile of near-duplicates from one event), and bin packing under the token budget. Weights need A/B testing, not guesswork.
**Follow-up**: Why not use similarity alone? It keeps recalling synonymous duplicates, squeezes out other key information, and cannot express "recent events matter more".

### Q6. How is MemGPT different from ordinary RAG?
**Answer**: RAG is **passive retrieval**: every query goes to the knowledge base once. MemGPT is **active memory management**: the model decides through function calls when to write content out of the main context and when to page it back, treating the context as limited RAM and external storage as disk, so paging breaks the window limit. RAG asks "which documents are relevant"; MemGPT asks "what should be in the context".
**Follow-up**: What is the cost? More complex control flow (the model must learn to call management functions), and failures can look like lost memories or thrashing in and out.

### Q7. How do you keep multi-user memory from leaking across users?
**Answer**: Three layers. (1) **Data layer**: every item carries tenant_id/user_id and retrieval uses them as **hard filters** (not scoring features); no global top-k. (2) **Index layer**: partition by user (namespace or collection) so approximate search never crosses users. (3) **Application layer**: inject only the current user's memory into the session and verify identity on writes. Add audit logs and overreach detection.
**Follow-up**: What about shared memory (team knowledge)? Put it in a separate organization-level namespace with explicit permissions and filter by ACL at read time.

### Q8. How do you evaluate a memory system?
**Answer**: Three parts. (1) **Retrieval quality**: hit rate and recall on a "did we store what should be stored" test set. (2) **Effect on task**: success rate, repeated-question rate, and satisfaction with versus without memory. (3) **Cost and timeliness**: tokens injected per turn, retrieval latency, and memory-store growth. Also test **forgetting correctness**: things that should be forgotten are, and things that should stay remain.
**Follow-up**: How do you build the data? Replay historical sessions and have humans label the "key facts" that should be recallable.

### Q9. What should not be written to long-term memory?
**Answer**: (1) One-off context (temporary variables of the current task). (2) Low-trust sources (unverified web content, the model's own guesses). (3) Sensitive data (secrets, full ID numbers, health data) unless there is explicit authorization and an encryption scheme. (4) High-churn information with no long-term value (current inventory counts). Judge "will this be needed later" before writing, or the store becomes a garbage dump quickly.
**Follow-up**: Can the model's own distilled "lessons" be written? Yes, but tag them with source and confidence and let later evidence overturn them.

## 5. Pitfalls and follow-ups

- **Equating memory with a vector store**: without write policies, decay, and isolation, the system degrades quickly.
- **Retrieval without user filters**: a soft filter (a scoring feature) is no isolation at all.
- **Treating the summary as the original**: summaries serve retrieval, raw text serves evidence; keep both.
- **Append-only memory**: the store grows without bound and both quality and cost get worse.
- **Ignoring the context budget**: tool schemas, retrieved content, and history stack up and squeeze out the output.
- **Follow-up trap**: "What is your memory stack?" You must cover layering, write criteria, retrieval scoring, isolation, and evaluation instead of naming one vector DB.

## 6. Cheat sheet

| Question | One-line answer |
|---|---|
| Essence of memory | Put the most useful information into the context within a token budget |
| Layers | Working / short-term summary / long-term / external knowledge |
| Three content types | Episodic (experience), semantic (facts), procedural (how-to) |
| Retrieval scoring | Relevance plus recency (exponential decay) plus importance, normalized and weighted |
| Write criteria | Information gain, importance threshold, explicit instruction, post-task retrospective |
| Forgetting | Decay, archiving, conflict merging, compliant deletion |
| MemGPT | Context as RAM, function calls for paging in and out |
| Isolation | `user_id` hard filters, partitioned indexes, audit |
| Evaluation | Recall, task success rate, injected tokens, forgetting correctness |

**Mnemonic**: "**Store the right things (criteria plus metadata), fetch the right things (three factors plus filters), forget the right things (decay plus compliance).**"

## 7. References

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [AI Agent Interview Guide · memory systems](https://github.com/bcefghj/ai-agent-interview-guide)
- [LLM-Agent-Interview-Guide · complete agent guide](https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide)
