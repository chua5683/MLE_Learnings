# Agent interview Q&A

> **How to use**: a 30-minute review before the interview; memorize only the "conclusion plus one reason" per question. For full derivations and code, follow the chapter link on each section.
>
> **Keywords**: `agent loop` `MCP` `ReAct` `memory` `multi-agent` `agentic RAG` `trace` `injection`

## A. Concepts and architecture ([01](01-agent-fundamentals-and-architecture.md))

### Q1. What is an LLM agent?
**Answer**: A system with an LLM as its decision core that can **plan, call tools, remember, and use feedback**, deciding the next action at runtime and iterating until the task completes. In one line: **the LLM is the brain, tools are the hands, memory is the notebook**.

### Q2. How is an agent different from a workflow?
**Answer**: It comes down to **who owns the control flow**. Workflow steps are hard-coded and the LLM only fills in local generations; an agent lets the model choose the action sequence from the observations. If the task paths are enumerable, use a workflow: more stable, cheaper, testable.

### Q3. How is an agent different from a "chatbot with tools"?
**Answer**: A chatbot is "one question, one tool call, one answer", so the user owns the control flow. An agent decides how many calls to make, which tool to use, and which path to take after failure, so it needs stop conditions, budgets, permissions, and observability.

### Q4. What are the four core components of an agent?
**Answer**: LLM (policy), Planning (decomposition and replanning), Tools (interaction with the outside world), and Memory (state across steps and sessions). Shipping also needs a guardrail layer: budgets, permissions, tracing, HITL.

### Q5. How do you design stop conditions?
**Answer**: Layer four: the model reports finish, a hard cap (steps / budget / time), an external verifier passing (unit tests / schema / state assertions), and loop detection (same action and arguments repeated). Production relies on **budget plus verification**; model self-awareness is only a hint.

### Q6. Why does "more steps mean lower success rate"?
**Answer**: Under independence, a $T$-step task succeeds with probability about $p^T$; with $p=0.95$ and $T=20$ only about 36% remains. So the goal is **fewer necessary steps plus more reliable steps** (better tools and schemas, earlier verification), not more rounds.

### Q7. What are the typical failure modes of an agent?
**Answer**: Wrong tool or hallucinated arguments, loops, unexecutable plans, context overflow, injection, and irreversible side effects. Countermeasures: schema plus validation, loop detection, capability inventories, context management, permissions plus confirmation, sandboxes plus rollback.

### Q8. How do you choose a framework?
**Answer**: Single agent: LangGraph or the OpenAI Agents SDK (explicit state machine plus checkpoints). Retrieval-heavy: LlamaIndex. Many roles: AutoGen or CrewAI. Business teams editing it themselves: a low-code platform. The core is always "state plus routing plus termination".

### Q9. What are the three trends for 2025-2026?
**Answer**: (1) **MCP standardization** of tool access; (2) **computer use** operating graphical interfaces; (3) **engineering-grade evaluation** (traces, regression sets, cost governance, sandboxes). A2A-style protocols are also exploring agent-to-agent collaboration.

### Q10. Why is context the bottleneck for agents?
**Answer**: Every round appends tool results to the context, so long tasks inflate it: cost and latency rise, and key information is diluted or truncated. So the engineering focus is context engineering: summaries, structured returns, retrieval-based memory, output truncation.

## B. Tool use and protocols ([02](02-tool-use-and-function-calling.md))

### Q11. How does function calling work?
**Answer**: Put tool schemas into the context → the model emits structured tool_calls (name plus JSON arguments) → the host executes them → results are fed back as role=tool with call_id → the model continues or answers. The model only expresses intent; execution always happens on the host side.

### Q12. What is the essential difference between function calling and "asking for JSON"?
**Answer**: The source of format reliability differs: function calling relies on **training plus decoding constraints plus a dedicated field**, is stable, and supports parallel calls; JSON prompting relies on the prompt and easily adds prose or drops fields. Both can be **semantically wrong**, so business validation is mandatory.

### Q13. Why does a tool description matter more than its name?
**Answer**: The model matches tools by natural language. The description must state **when to use it, when not to, what the arguments mean, what it returns, and the limits**; it is basically interface documentation written for the model.

### Q14. How do you design a schema that reduces bad arguments?
**Answer**: Use enums where possible and numbers where possible, provide defaults and examples, restrict the required set, set additionalProperties: false, and on validation failure feed the **specific field error** back to the model.

### Q15. What do you do with too many tools (50+)?
**Answer**: Three routing layers: rules or intent classification for high-frequency cases → vector recall of tool descriptions for Top-K → the model decides inside the candidate set; plus grouping and a fallback that explicitly refuses when no tool fits.

### Q16. What should you watch out for with parallel tool calls?
**Answer**: Dependencies (dependent calls must be serialized into separate rounds), side-effect conflicts (writes must be idempotent or serialized), rate limits and timeouts, strict call_id pairing, and an explicitly defined partial-failure policy in the application layer.

### Q17. How do you help an agent recover from tool failures?
**Answer**: Feed back **structured, actionable** errors (error code plus plain-language cause plus whether it is retryable plus a fix hint); retryable errors use exponential backoff, non-retryable ones tell the model to change arguments or path, and repeated failures degrade or hand off to a human.

### Q18. What is MCP?
**Answer**: An open protocol proposed by Anthropic that standardizes discovery and invocation of tools, resources, and prompt templates between hosts, clients, and servers; stdio locally and Streamable HTTP remotely. Think of it as USB-C for the AI side.

### Q19. Is MCP a replacement for function calling?
**Answer**: No. MCP covers **where tools come from, how they are discovered, and how they are authenticated**; function calling covers **how the model expresses a call**. The host converts MCP tools into function schemas, so they are complementary.

### Q20. What are the hard parts of computer use?
**Answer**: No DOM semantics, so elements must be located from screenshots (coordinate drift), errors accumulate over long sequences, dynamic pages are unreproducible, and a misclick can be irreversible. Mitigations: accessibility-tree numbering, verifying the screenshot after each action, human confirmation for risky steps, and an application allowlist.

### Q21. How do you defend against tool poisoning?
**Answer**: Allowlist and review third-party tools and MCP servers, treat tool descriptions and returns as **data** rather than instructions, scope credentials to least privilege, require human confirmation for risky actions, and monitor and audit tool calls for anomalies.

### Q22. How do you keep code execution safe?
**Answer**: Container-level isolation (gVisor / Firecracker / nsjail), no network by default, a read-only root filesystem, resource quotas, a fresh environment per task, output truncation, and audit logging; secrets never enter the sandbox.

## C. Planning and reasoning ([03](03-planning-and-reasoning.md))

### Q23. What is ReAct?
**Answer**: Thought → Action → Observation alternation, so reasoning guides action and action results correct the reasoning. It is flexible and interpretable, but myopic, step-heavy, and errors propagate through the observations.

### Q24. ReAct or Plan-and-Execute?
**Answer**: Short tasks with strong feedback and uncertain paths → ReAct; long, decomposable, parallelizable tasks → plan first, then execute with replanning. A common structure is **Plan-and-Execute outside, ReAct inside**.

### Q25. Why does CoT work?
**Answer**: It adds serial computation, externalizes implicit computation into intermediate results that later steps can rely on, and matches the training distribution of "hard problems come with worked steps". Limitation: no external information enters, the chain can fabricate, and factual tasks still need retrieval or tools.

### Q26. What does Self-Consistency require?
**Answer**: Errors must be **scattered** (different chains fail in different places) and answers must be **comparable**. Hence it helps on math or multiple choice, does almost nothing for open-ended generation, and its cost grows linearly with the sample count.

### Q27. Is ToT worth it?
**Answer**: Only when there are many paths, intermediate states can be scored, and failure is expensive. It often costs more than ten times CoT (multiple generations plus evaluations per level). Production usually enables search only for failing samples.

### Q28. Why does Reflexion sometimes do nothing?
**Answer**: It depends on a **trustworthy evaluation signal**. If the evaluator is the same model with the same blind spots as the actor, the reflection approves itself; vague reflections cannot guide action either. It needs an external signal such as unit tests or environment reward, and reflections concrete to "which step, what cause, what to change".

### Q29. How do you design a verifier?
**Answer**: Prefer external objective signals (unit tests, compilation, schema, numeric assertions, comparing against retrieved source text), then multi-model cross-validation and splitting the problem into decidable sub-questions. Asking the model to "check again" is internal verification and helps little with factual or logical errors.

### Q30. How do you keep long-horizon tasks on track?
**Answer**: Externalize state (files / DB / checkpoint), use layered budgets, checkpoint per subtask for rollback, treat the plan as a board, detect loops, require HITL before key decisions, and hand off to a human after N consecutive failures.

### Q31. How do you evaluate planning ability?
**Answer**: Three layers: subtask level (covers the necessary steps, executable, correct dependencies), trajectory level (step count, wasted-step share, repeat rate), and end to end (success rate, pass^k, cost). Reporting only end to end cannot distinguish a planning error from an execution error.

### Q32. When should the LLM not be doing the planning?
**Answer**: When the path is fixed (use a workflow), when a mature solver exists (OR-Tools for scheduling and routing), when latency matters, or when strict auditability is required. LLMs fit **high-level goal decomposition and exception handling**, not precise scheduling.

## D. Memory systems ([04](04-memory-systems.md))

### Q33. Is long context enough, or do we still need memory?
**Answer**: Not enough. Cost grows with turns, long contexts suffer from lost in the middle, and cross-session information does not persist. Long context makes "retrieve less" feasible but does not replace write policies, decay, and isolation.

### Q34. How should memory be layered?
**Answer**: Working memory (context window) → short-term summaries (compressed within a session) → long-term memory (vector / KV / graph, across sessions) → external knowledge (RAG index); by content, episodic, semantic, and procedural. Each layer has a different read/write path.

### Q35. Does a vector store count as a memory system?
**Answer**: It is one implementation of long-term memory but incomplete. It still lacks write criteria, time and importance weighting, conflict updates, forgetting and archiving, and user isolation plus compliant deletion. Pure similarity also recalls "superficially similar but irrelevant" items and cannot match exact IDs.

### Q36. How should memory retrieval be scored?
**Answer**: Weight three factors: relevance (cosine similarity) plus recency (exponential decay) plus importance (model or rule score), normalized and weighted; then add metadata hard filters (user / time / type), diversity, and token packing.

### Q37. How is MemGPT different from RAG?
**Answer**: RAG is passive retrieval (query the store each time); MemGPT is active memory management, treating the context as RAM and external storage as disk, with the model deciding through function calls when to page out and page in, breaking the window limit. The price is a more complex control flow.

### Q38. How do you keep multi-user memory from leaking?
**Answer**: Use `user_id` / `tenant_id` as **hard filters** (not scoring features), partition indexes per user, inject only the current user's memory into the session, verify identity on writes, and keep a full audit trail.

### Q39. How do you design a forgetting policy?
**Answer**: Time decay lowers weight, access reinforcement raises it, conflicts merge (latest wins for preferences), hot and cold tiers archive, and compliance requires hard deletion with export and audit. Decay must be slowed for important memories so they are not deleted by mistake.

## E. Multi-agent systems ([05](05-multi-agent-systems.md))

### Q40. Are multi-agent systems always better than a single agent?
**Answer**: No. The gains hold only when the task can be split with few information dependencies, needs different tools and roles, or needs independent verification; costs grow roughly quadratically (Anthropic reports about 15x tokens for its multi-agent research system) and debugging gets harder.

### Q41. How do you estimate multi-agent communication cost?
**Answer**: With full connectivity the message count is $O(n^2)$ and each message carries context. Savings: star or hierarchical topologies ($O(n)$), structured summaries, capped rounds and length, and early stopping when there is no progress.

### Q42. How do you prevent multi-agent loops?
**Answer**: Max rounds plus budget cap plus convergence criteria (stable judge, consistent votes, no new information) plus loop detection (repeated delegation), and appoint an explicit final owner.

### Q43. When does debate help?
**Answer**: When errors can be exposed by argument and participants have different information sources (different models or evidence). Debating with the same model, prompt, and retrieval source only fails together.

### Q44. Single agent with many tools, or multiple agents?
**Answer**: Three tests: does context contaminate itself, do you need parallelism, do you need independent verification. If none applies, stay single-agent; the middle ground is "one controller plus a few dedicated sub-agents (retrieval, code execution)".

### Q45. What is the difference between A2A and MCP?
**Answer**: MCP is the vertical access protocol for "agent to tools and data"; A2A is the horizontal protocol for "agent to agent" discovery, delegation, and state synchronization. The former is mature, the latter early, and they are complementary.

## F. Agentic RAG ([06](06-agentic-rag.md))

### Q46. How is Agentic RAG different from Advanced RAG?
**Answer**: Advanced RAG is a fixed pipeline (rewrite → retrieve → rerank → generate); Agentic RAG makes retrieval a tool and lets the model decide what to look up, how many times, whether the evidence suffices, and whether to switch tools. Control flow moves to the model, and cost and latency rise with it.

### Q47. When is agentic retrieval worth it?
**Answer**: Multi-hop, heterogeneous sources, open-ended questions, expensive mistakes, and unstable recall with room to improve. It is not worth it for single-hop FAQ, latency-sensitive paths, or already-solid retrieval. The test: can one retrieval probably collect all the evidence.

### Q48. How do you handle multi-hop questions?
**Answer**: Decompose the query → iterate serially (use the previous hop's result to build the next query) → ground every hop with citations → cap at 2-4 hops → if still short, give a partial answer and say what is missing. Independent sub-questions can run in parallel.

### Q49. Self-RAG versus CRAG?
**Answer**: Self-RAG trains the model to emit reflection tokens for whether to retrieve and whether passages are relevant or supporting (internalized, needs training); CRAG bolts on a retrieval evaluator that sorts results into correct / incorrect / ambiguous and triggers different actions (easy to ship, observable). Products start with the CRAG-style external layer.

### Q50. Is GraphRAG worth it?
**Answer**: Only for global thematic questions or multi-hop relations over a stable structure. Graph construction plus community summaries is LLM-expensive and incremental updates are painful; most enterprise knowledge bases do fine with hybrid retrieval plus reranking plus agentic follow-up queries.

### Q51. How do you guarantee citations and reduce hallucination?
**Answer**: Force citations during generation (sentences without a citation are not emitted) → verify entailment sentence by sentence after generation (delete or rewrite unsupported claims) → abstain explicitly when evidence is insufficient → make citations clickable back to the source.

### Q52. Which layer should permission filtering live in?
**Answer**: The **retrieval layer** (attach the ACL filter to the query), not post-generation trimming; otherwise the model has already seen unauthorized content and may leak it. The cache key must include permissions and version too.

## G. Production, evaluation, and safety ([07](07-evaluation-and-safety.md))

### Q53. How do you evaluate an agent?
**Answer**: Three layers: end to end (task success rate, pass@1 / pass^k, cost, P95 latency), trajectory (steps, wasted steps, repeat rate, tool accuracy), component (retrieval hit rate, judge agreement), plus an online A/B.

### Q54. What are the biases of LLM-as-judge and how do you use it?
**Answer**: Position, length, self-preference, and sycophancy bias. Mitigate with a rubric split into dimensions, pairwise comparison with swapped order, reference answers, and calibration against human labels with reported agreement; before calibration use it only as a soft gate.

### Q55. How do you localize an online failure?
**Answer**: Walk the span tree of the trace, find the **first anomalous span**, check that step's input (missing information?), tool arguments (schema), and return (truncated or timed out), then reproduce offline with fixed model, prompt, and tool versions, and add it to the regression set.

### Q56. How do you optimize cost?
**Answer**: Fewer steps > compress context > cache > model routing > parallelism (saves time, not money) > early stopping; account cost per **successful task** and amortize failed retries.

### Q57. Why is prompt injection more dangerous for agents?
**Answer**: Agents actively read untrusted content and hold execution permissions, so injection can directly cause exfiltration or writes. Assume "some injection will succeed" and bound the blast radius with least privilege, sandboxes, HITL, and audit.

### Q58. How do you roll out safely?
**Answer**: Split by session or user (not per request, so one session never spans versions), monitor success rate, cost, latency, human takeover rate, and security events, set automatic rollback thresholds, ramp 1% → 5% → 25% → full traffic, and use shadow mode first for risky changes.

## Topic quick reference (memorize this table if 58 questions is too many)

| Topic | One-line conclusion | Details |
|---|---|---|
| Agent definition | An LLM policy plus planning, tools, and memory in a loop | [01](01-agent-fundamentals-and-architecture.md) |
| Selection | If a workflow can do it, skip the agent | [01](01-agent-fundamentals-and-architecture.md) |
| Tool use | The model expresses intent; the host always executes | [02](02-tool-use-and-function-calling.md) |
| MCP | Handles tool access and discovery, complements function calling | [02](02-tool-use-and-function-calling.md) |
| Planning | Decompose, search, verify; trade test-time compute for accuracy | [03](03-planning-and-reasoning.md) |
| Memory | Put the most useful information into the token budget; it is not "having a vector store" | [04](04-memory-systems.md) |
| Multi-agent | A cost multiplier; first ask whether the task splits independently | [05](05-multi-agent-systems.md) |
| Agentic RAG | The model owns the control flow: what to look up, how often, whether it suffices | [06](06-agentic-rag.md) |
| Shipping | Observable, measurable, controllable; all three are required | [07](07-evaluation-and-safety.md) |

## Ten conclusions

1. **If a workflow can do it, do not use an agent**; freedom must be bought back with engineering.
2. **Stop conditions** are always: budget plus verification plus loop detection.
3. **Fewer steps is more stable**; $p^T$ is a merciless fact.
4. **A tool description is interface documentation**; wrong tools are usually a description problem.
5. **MCP handles access, function calling handles expression**; they complement each other.
6. **Three planning paths**: think while acting (ReAct), think before acting (P&E), reflect after acting (Reflexion).
7. **Memory is not a vector store**; it also needs write criteria, decay, and isolation.
8. **Multi-agent is a cost multiplier**; if subtasks cannot stand alone, do not split.
9. **The essence of Agentic RAG is that the model owns the control flow**; fix retrieval before orchestration.
10. **No traces, no optimization**; no regression set, no iteration.

> Full derivations, code, and follow-ups live in the other seven notes of this chapter; the cross-topic cheat sheet is [Agent cheat sheet](../08-interview-bank/03-agent-cheatsheet.md).
