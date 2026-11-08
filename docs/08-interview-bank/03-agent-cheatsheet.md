# Agent cheat sheet

> **Purpose**: a one-page review of agent concepts, paradigms, protocols, and metrics. Full detail is in [Chapter 5 · AI Agents](../05-agents/README.md).
>
> **Keywords**: `agent loop` `ReAct` `MCP` `memory` `multi-agent` `task success rate` `pass@k`

## 1. Definition and the four components

**Agent = LLM (policy) + Planning + Tools + Memory**, an observe → think → act loop controller at heart.

| Component | One-line responsibility | Typical failure |
|---|---|---|
| LLM | Pick the next action, write arguments, decide when to stop | Loops, hallucinated arguments, wrong tool |
| Planning | Decompose, order, replan | Unexecutable plan |
| Tools | Interact with the outside world | Timeouts, side effects, overreach |
| Memory | Persist state across steps and sessions | Forgetting, leakage, context overflow |

**Selection rule**: if a workflow can do it, skip the agent (enumerable steps → workflow; not enumerable → agent; partially enumerable → an agent with workflows embedded).

## 2. Paradigm comparison (memorize this one table)

| Paradigm | Structure | External feedback | Cost | When to use |
|---|---|---|---|---|
| CoT | Reason first, then answer | None | 1x | Multi-step tasks whose knowledge is inside the model |
| Self-Consistency | Sample k chains and vote | None | kx | Comparable answers, scattered errors |
| ToT | Expand thoughts into a tree, evaluate and prune | Self-evaluation | 10x+ | Multiple paths, scorable states |
| ReAct | Thought-Action-Observation alternation | Yes | 3-10x | Needs external information, uncertain paths |
| Plan-and-Execute | Plan first, execute, then replan | Yes | 2-5x | Long, parallelizable tasks |
| Reflexion | Act → evaluate → reflect → retry | Yes (must be trustworthy) | kx | Has unit tests or a reward signal |
| LATS | MCTS plus UCB tree search | Yes | Tens of x | High-value tasks that tolerate retries |

## 3. Tool use and MCP

- **Three call steps**: the model emits tool_calls → the host executes (auth / validation / rate limits / audit) → role=tool plus call_id feeds the result back.
- **Five schema elements**: name, description (when to use and not use), types, enum or range, required fields.
- **Too many tools**: rules or intent → vector recall of Top-K → model decision plus fallback (three layers).
- **Error feedback**: error code plus plain-language cause plus retryable flag plus a fix hint.
- **MCP**: Host → Client → Server; three primitives **Tools / Resources / Prompts**; transport stdio (local) or Streamable HTTP (remote); solves access and discovery, complementary to function calling rather than a replacement.
- **Four security measures**: least privilege, allowlist, sandbox, human confirmation.

## 4. Memory

| Layer | Content | Implementation |
|---|---|---|
| Working | Instructions, recent turns, tool results | Context window |
| Short-term | Key points of the session history | Summary plus sliding window |
| Long-term | Preferences, facts, experience | Vector store / KV / graph |
| External | Documents, code, artifacts | RAG index, filesystem |

- **Retrieval scoring**: relevance plus recency (exponential decay) plus importance, normalized and weighted, then metadata hard filters.
- **MemGPT**: the context is RAM, external storage is disk, and function calls page content in and out.
- **Isolation**: `user_id` hard filters, partitioned indexes, audit; **forgetting**: decay, archiving, conflict merging, compliant deletion.

## 5. Multi-agent

- **Gains**: context isolation, specialization, parallelism, cross-checking; **costs**: communication $O(n^2)$, unpredictability, hard debugging.
- **Topologies**: supervisor (most common), pipeline, debate, blackboard, handoff.
- **Termination**: max rounds plus budget plus convergence criteria plus loop detection, with an explicit final owner.
- **Protocol split**: MCP covers agent-to-tool, A2A covers agent-to-agent.

## 6. Agentic RAG

- **Essence**: make retrieval a tool and let the model own the control flow (what to look up, how many times, whether it suffices).
- **Trigger conditions**: multi-hop, heterogeneous sources, open-ended questions, expensive mistakes; single-hop FAQ does not justify it.
- **Stop criteria**: evidence coverage plus marginal gain plus consistency plus budget.
- **Self-RAG**: internalized reflection tokens; **CRAG**: an external evaluator sorts results into three buckets (correct / incorrect / ambiguous).
- **Four enterprise musts**: push permissions down to retrieval, incremental indexing, forced citations, abstain when evidence is thin.

## 7. Key numbers and metrics

| Metric | Notes |
|---|---|
| Steps vs success | Independent approximation $p^T$; with $p=0.95$ and $T=20$, about 36% |
| pass@k | $1-\binom{n-c}{k}/\binom{n}{k}$; in practice also track the mean over k repeats and the worst 10% |
| Three metric layers | Component (tool accuracy) / trajectory (steps, repeat rate) / end to end (success, cost, P95 latency) |
| Cost levers | Fewer steps > compress context > cache > model routing > parallelism > early stop |
| Multi-agent cost | Full connectivity is $O(n^2)$; a multi-agent research system uses about 15x chat tokens (Anthropic) |
| RRF fusion | $\sum_i 1/(k+\mathrm{rank}_i)$, $k\approx 60$ |
| Judge calibration | Align against 200-500 human labels and report agreement ($\kappa$) |

## 8. Security and observability (bonus points in interviews)

- **Minimum trace fields**: trace/span/parent, type, input/output (redacted), token/cost, latency, versions.
- **Defense in depth**: input labeling → instruction/data separation → least privilege → sandbox → HITL → output filtering → audit → rollback.
- **Why injection is worse**: agents actively read untrusted content and hold execution permissions.
- **Canary**: split by session or user, 1% → 5% → 25% → full traffic, with automatic rollback.

## 9. Ten conclusions

1. If a workflow can do it, do not use an agent.
2. Stop conditions = budget plus verification plus loop detection.
3. Fewer steps is more stable ($p^T$).
4. A tool description is interface documentation.
5. MCP handles access, function calling handles expression.
6. Three planning paths: think while acting, think before acting, reflect after acting.
7. Memory is not a vector store; it also needs write criteria, decay, and isolation.
8. Multi-agent is a cost multiplier; if subtasks cannot stand alone, do not split.
9. The essence of Agentic RAG is that the model owns the control flow; fix retrieval before orchestrating.
10. No traces, no optimization; no regression set, no iteration.

## 10. Related chapters

- [Agent basics and architecture](../05-agents/01-agent-fundamentals-and-architecture.md)
- [Tool use and function calling](../05-agents/02-tool-use-and-function-calling.md)
- [Planning and reasoning](../05-agents/03-planning-and-reasoning.md)
- [Memory systems](../05-agents/04-memory-systems.md)
- [Multi-agent systems](../05-agents/05-multi-agent-systems.md)
- [Agentic RAG and knowledge augmentation](../05-agents/06-agentic-rag.md)
- [Production evaluation and safety](../05-agents/07-evaluation-and-safety.md)
- [Agent interview Q&A](../05-agents/08-agent-interview-qa.md)
- [RAG](../04-llm/11-rag.md)
