# Planning and reasoning

> **One-line summary**: Planning replaces "one-shot generation" with "decomposition + search + verification", trading more test-time compute for accuracy: ReAct is **myopic greedy control**, Plan-and-Execute is **a global plan plus replanning**, Reflexion is **distilling lessons from failure**, and LATS is **treating reasoning as tree search**.
>
> **Keywords**: `CoT` `self-consistency` `ToT` `ReAct` `Plan-and-Execute` `Reflexion` `LATS` `verifier`

## 1. Core concepts

### 1.1 The four problems planning must solve

| Problem | Meaning | Cost of getting it wrong |
|---|---|---|
| Decomposition | Split the goal into executable subtasks | Too coarse → cannot execute; too fine → step explosion |
| Ordering | Establish dependencies and parallelism | Wrong order → arguments unavailable, results invalid |
| Allocation | Choose the tool / model / agent per subtask | Capability mismatch → repeated failure |
| Replanning | Adjust the plan from intermediate results | Clinging to a stale plan → total failure |

### 1.2 The paradigm family

| Paradigm | In one line | External feedback | Relative cost | Typical use |
|---|---|---|---|---|
| CoT | Write the reasoning, then answer | None | 1x | Math, logic, multi-step QA |
| Self-Consistency | Sample several chains and vote | None | kx | Reasoning tasks with enumerable answers |
| ToT | Expand thoughts into a tree, search and evaluate | Partial (self-evaluation) | 10x+ | Tasks that need exploration and mid-state scoring |
| ReAct | Alternate thinking and tool calls | Yes (tools) | 3-10x | Retrieval QA, tasks needing external information |
| Plan-and-Execute | Plan first, execute, replan on failure | Yes | 2-5x | Long, parallelizable multi-step tasks |
| Reflexion | Act, evaluate, reflect, retry | Yes (evaluator) | kx | Code generation, tasks with a clear success signal |
| LATS | MCTS-style tree search with reflection backprop | Yes (environment + self-eval) | Tens of x | High-value problems that tolerate repeated attempts |

### 1.3 Two kinds of "thinking"

- **Implicit thinking** (reasoning models such as o1 / R1): the long CoT happens inside the model and only the final answer is visible. Pros: end-to-end optimization, no orchestration. Cons: not steerable, not auditable, cannot call tools.
- **Explicit thinking** (an agent's thought step): the reasoning is written into the context, so it can be inspected, verified, and interleaved with tool calls. Cons: depends on prompt quality and consumes context.

Interview framing: **the two are complementary**. Reasoning models are responsible for "thinking deeply"; agent orchestration is responsible for "acting correctly". A common pattern is a reasoning model as the brain inside a tool loop.

## 2. Principles and derivations

### 2.1 Why CoT works

Three reasons: (1) **more serial computation** (one forward pass becomes multi-step token generation, effectively deeper computation); (2) **explicit intermediate results** become conditions for later steps, lowering the difficulty of each step; (3) **alignment with the training distribution** (hard problems in the training data usually come with worked solutions).

Limitation: no external information enters, so the reasoning chain can fabricate confidently, and the longer the chain the more errors accumulate. CoT fits tasks whose **knowledge already lives in the model**, not tasks that need factual retrieval.

### 2.2 Self-Consistency

Sample $k$ reasoning chains and take a majority vote over final answers:

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^{k} \mathbb{1}\!\left[a_i = a\right]
$$

It works when **errors are scattered** (different chains fail in different places) so the correct path is hit more often. It therefore helps on tasks with a unique, comparable answer (math problems) and gives almost nothing on open-ended generation. Cost grows linearly with $k$ while gains diminish; in practice $k=5\sim 20$.

### 2.3 Tree-of-Thoughts (ToT)

Turn reasoning into explicit search: **state** = partial solution, **action** = generate candidate next steps, **evaluation** = have the model score or vote on candidates, **search** = BFS/DFS with pruning and backtracking.

$$
\text{complexity} \approx O(b^{d}) \quad (b=\text{branching factor},\ d=\text{depth})
$$

So pruning is mandatory: keep only top-$k$ per level, cap the depth, and use a cheaper model to evaluate. The gain comes from **being able to back out**: once CoT goes wrong, it can only stay wrong.

### 2.4 ReAct: interleaving reasoning and acting

The trajectory has the form:

$$
\tau = (t_1, a_1, o_1, t_2, a_2, o_2, \dots, t_T)
$$

where $t_i$ is a thought (natural-language reasoning), $a_i$ is an action (tool call), and $o_i$ is an observation (tool return). At each round the model generates the next step from the full trajectory.

**Three hidden problems in ReAct** (good material for deep-dive questions):
1. **Myopia**: every step optimizes locally with no global plan.
2. **Context growth**: each round's observation stays in the context.
3. **Error propagation**: one wrong tool call produces a misleading observation that skews all later reasoning.

### 2.5 Plan-and-Execute

Two phases: a **planner** emits a subtask list (optionally with a dependency DAG) → an **executor** runs them one by one (each subtask can use ReAct internally) → if a subtask fails or its result contradicts an assumption, **replanning** fires.

Key design points:
- The planner prompt must list the **real tools and their limits**, otherwise the plan is unexecutable.
- Each subtask needs **a verifiable completion criterion**.
- Replan too often and it degenerates into ReAct; replan too slowly and it clings to a broken plan.
- If subtasks can run in parallel, execute them by DAG layer to cut wall-clock time.

Cost comparison: ReAct feeds the whole trajectory to the model at every step ($O(T)$ steps times $O(T)$ context); Plan-and-Execute generates the plan once and each execution step only sees "the current subtask plus relevant intermediate results", so total tokens are usually lower, but plan quality sets the ceiling.

### 2.6 Reflexion: learning from failure

Three roles: **Actor** (runs the task) → **Evaluator** (judges success, using unit tests, environment reward, or an LLM judge) → **Self-Reflection** (writes a natural-language critique such as "I forgot the empty-list case") → the reflection is stored in memory and the next Actor attempt retries with it.

$$
\text{improvement} = f(\text{failed trajectory},\ \text{evaluation signal}) \;\Rightarrow\; \text{verbal memory}
$$

**The precondition is a trustworthy evaluation signal.** It works well on code tasks with unit tests, a compiler, or environment reward (the Reflexion paper reports large gains on HumanEval). If the evaluator is the same model "feeling whether it is right", it congratulates itself and the gains evaporate.

### 2.7 LATS: MCTS plus a language model

Treat each node as a partial solution and run Monte Carlo tree search:

$$
\mathrm{UCB}(s,a) = Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}
$$

where $Q(s,a)$ is a value estimate (an LLM score or environment reward), $N$ counts visits, and $c$ controls exploration. After the search, values are backpropagated and reflections from failed nodes are written to their parents. LATS fits tasks that **tolerate repeated trial and error and have a clear reward**, at tens of times the cost of plain ReAct.

### 2.8 Verifiers: why "generate then verify" beats "generate once"

Verification is usually easier than generation (judging a proof is easier than writing one). So you can do:
- **Outcome verification**: unit tests, compilation, schema, numeric assertions, retrieval comparison.
- **Process verification** (PRM): score intermediate steps and use that for search pruning or RL reward.
- **Cross-validation**: accept only when multiple models or samples agree.

The most valuable engineering move is **finding a trustworthy verifier**: if you can write tests, have the model write tests; if you can query a database, do not rely on memory; if you can run it, do not just talk about it.

### 2.9 Long-horizon tasks: state, checkpoints, recovery

| Problem | Approach |
|---|---|
| Context cannot hold it | Externalize state (files / DB / checkpoint) and keep only the current slice in the prompt |
| Runaway execution / loops | Layered budgets (total plus per-subtask) and loop detection |
| Mid-run failure | Retry at the subtask level → change strategy → roll back to the latest checkpoint → ask a human |
| Invisible progress | The plan is the board: every subtask has a status (todo / doing / done / failed) |
| Irreproducible results | Record per-step inputs, outputs, and tool versions to enable replay |

## 3. Implementation notes

A Plan-and-Execute plus Reflexion skeleton (LLM call details omitted):

```python
def plan_execute(task, tools, llm, max_rounds=3):
    memory = []                                   # reflection memory
    for rnd in range(max_rounds):
        plan = llm.plan(task, tools=describe(tools), lessons=memory)   # 1) plan
        done, logs = [], []
        for step in plan:                          # 2) execute (DAG layers if parallelizable)
            try:
                out = llm.react(step, tools=tools, context=done[-3:])
                done.append({"step": step, "out": out})
            except Exception as e:
                logs.append(f"{step} failed: {e}")
                plan = llm.replan(task, done, logs)   # 3) trigger replanning
                break
        verdict = llm.evaluate(task, done)          # 4) evaluate (prefer unit tests)
        if verdict.ok:
            return done
        memory.append(llm.reflect(task, done, verdict))   # 5) reflection into memory
    return done
```

Key engineering details: **record a span per step** (see [production evaluation](07-evaluation-and-safety.md)), **truncate subtask output before it enters the context**, **cap the number of replans**, and **prefer external signals over model self-evaluation**.

## 4. Interview Q&A

### Q1. What is the essential difference between ReAct and Plan-and-Execute?
**Answer**: **The timing of plan generation versus execution**. ReAct thinks and acts alternately, so each step depends only on the current observation; it is a greedy strategy. Plan-and-Execute generates a global plan first and then executes it. The former is flexible and suits short tasks with uncertain paths; the latter gives a global view, allows parallelism, and uses less context, but its plan can go stale and may need replanning.
**Follow-up**: Can you combine them? Yes. A common structure is Plan-and-Execute outside with a ReAct loop inside each subtask.

### Q2. Why does Reflexion sometimes do nothing at all?
**Answer**: It depends on **a trustworthy evaluation signal**. If the evaluator is the same model with the same blind spots as the actor, the reflection validates the error, and the next round repeats it. Reflections that are too vague ("I should be more careful") cannot guide action either. It requires external signals such as unit tests, environment reward, or human feedback, and reflections must be concrete about "which step, what cause, what to change next time".
**Follow-up**: How do you fix it? Use execution feedback (error messages, test results) as the reflection input, and require the reflection to produce an executable checklist.

### Q3. Is ToT worth using?
**Answer**: Only when **exploration gain exceeds cost**: the task has several viable paths, intermediate states can be scored, and a single failure is expensive. The cost is multiple generations plus multiple evaluations per level, often ten times CoT or more, with high latency. In production the more common approach is to take a CoT or ReAct baseline and enable search only for failing samples.
**Follow-up**: How do you prune? Keep top-k candidates per level, cap depth, evaluate with a small model, and cache repeated states.

### Q4. What are the common failure modes of planning?
**Answer**: (1) unexecutable plans (assuming tools or permissions that do not exist); (2) wrong granularity (subtasks too coarse to execute or too fine, causing step explosion); (3) dependency errors (running dependent steps in parallel); (4) plan clinging (no replanning); (5) missing evaluation (no idea whether a subtask finished); (6) ignoring the budget (the plan itself exceeds the cost ceiling).
**Follow-up**: How do you catch this early? Have the planner emit a structured plan (tools, inputs, completion criteria, estimated steps) and run a static check before execution.

### Q5. How do you evaluate "planning ability"?
**Answer**: Three layers: (1) **subtask level**: does the plan cover every necessary condition, are the steps executable, are dependencies correct (checked by humans or rules); (2) **trajectory level**: step count, fraction of wasted steps, repeat rate, tool-call accuracy; (3) **end to end**: task success rate, pass^k, cost and latency. Reporting only end-to-end success is not enough because you cannot tell a planning error from an execution error.
**Follow-up**: Can it be automated? Build a verifiable task set (unit tests or state assertions) and infer plan quality from execution outcomes.

### Q6. When should you not let the model plan?
**Answer**: (1) The path is fixed and known (a workflow is more stable and faster). (2) A mature solver exists (use OR-Tools or PDDL for scheduling and routing instead of making an LLM think hard). (3) Latency is critical (multi-round planning is slow). (4) Results must be strictly explainable or auditable (use rules or structured plans). The right place for LLM planning is **high-level goal decomposition and exception handling**, not precise scheduling.
**Follow-up**: Why? LLMs are unstable on combinatorial optimization, while solvers guarantee optimality and reproducibility.

### Q7. How do you keep long-horizon tasks from running away?
**Answer**: Four things: **externalize state** (plan and progress in files or a DB, not in context memory), **layered budgets** (total plus per-step plus per-tool caps), **checkpoints** (persist after each subtask so you can roll back), and **a board plus human intervention points** (HITL before key decisions). Add loop detection (same action and arguments repeated) and "N consecutive failures → human handoff".
**Follow-up**: What rollback granularity? The latest subtask checkpoint with an explicit completion criterion.

### Q8. How do you design a verifier, and why is it better than asking the model to "check again"?
**Answer**: Prefer **external objective signals**: unit tests, compilation, schema validation, numeric assertions, database queries, the retrieved source text. An objective verifier cannot be contaminated by the model's self-preference. Asking the model to re-check is internal verification: it helps with format issues but is weak on factual and logical errors because the model does not know what it does not know.
**Follow-up**: What if no objective signal exists? Use multi-model cross-validation, check citations against retrieved text, and split verification into smaller decidable questions.

### Q9. Will reasoning models (long CoT) replace ReAct?
**Answer**: No; they solve different problems. Reasoning models strengthen **internal multi-step reasoning** (math, code, logic), but they still do not know real-time information, cannot take actions, and cannot access private data. ReAct-style loops solve **interaction with the outside world**. The trend is fusion: a reasoning model for deep single-step thinking, wrapped in a tool loop with environment feedback.
**Follow-up**: How do you weigh cost? Reasoning models have high per-token prices and long latency; for simple tasks a small model plus tools is cheaper.

### Q10. How do you cut the cost of multi-round reasoning?
**Answer**: (1) Fewer steps (better tools, stricter schemas, gathering more information per call). (2) Compress context (keep only structured observations, summarize old steps). (3) Route (single call for simple tasks, loop only for complex ones). (4) Cache (reuse identical retrieval or tool results). (5) Tiered models (strong model for planning, cheap model for execution). (6) Early stopping (stop when the verifier passes; do not "think again").
**Follow-up**: How do you measure the improvement? Watch tokens per task, tool-call count, and end-to-end success rate move together.

## 5. Pitfalls and follow-ups

- **Assuming "thinking longer equals thinking better"**: without external feedback, a longer chain may just be a longer chain of errors.
- **Substituting self-evaluation for verification**: LLM self-review barely helps on factual errors; use unit tests when you can.
- **Unexecutable plans**: if the planner does not know the tool limits, the whole plan is castles in the air.
- **Reflections that stay abstract**: a reflection must name the step, the cause, and the next action, or it cannot be executed.
- **Ignoring the cost ceiling**: ToT and LATS search trees blow budgets on real tasks very easily.
- **Follow-up trap**: "Have you used Reflexion, and how did it go?" You must be able to say under which signal it works and where it fails.

## 6. Cheat sheet

| Paradigm | Memory hook |
|---|---|
| CoT | Externalizes implicit computation; fits multi-step tasks whose knowledge is already inside |
| Self-Consistency | Vote over chains; requires scattered errors and comparable answers |
| ToT | Reasoning as search; control $O(b^d)$ with evaluation and pruning |
| ReAct | Thought-Action-Observation alternation; myopic, errors accumulate |
| Plan-and-Execute | Plan then execute plus replan; saves context, enables parallelism |
| Reflexion | Fail → reflect → write memory → retry; needs a trustworthy evaluator |
| LATS | MCTS plus UCB; highest cost, fits high-value tasks |
| Verifier | Verification is easier than generation; prefer external objective signals |

**Mnemonic**: "**Decompose when you can, verify when you can, go external instead of self-evaluating, use a workflow instead of search.**"

## 7. References

- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Language Agent Tree Search Unifies Reasoning Acting and Planning](https://arxiv.org/abs/2310.04406)
- [AI Agent Interview Guide · core frameworks](https://github.com/bcefghj/ai-agent-interview-guide)
