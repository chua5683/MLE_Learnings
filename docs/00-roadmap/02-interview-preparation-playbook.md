# Interview Preparation Playbook

> **One-line summary**: An interview is not a knowledge contest; it is a bounded-time demonstration of trustworthy engineering judgment. Break preparation down by round and practice delivery — that beats memorizing another 100 questions.
>
> **Keywords**: `interview rounds` `ML coding` `model design` `project deep dive` `behavior` `retrospective`

## 1. Typical rounds

Companies differ, but MLE / LLM on-sites usually combine the following (using Yuan Meng's MLE Interview 2.0 taxonomy):

| Round | Duration | What it tests | Common format |
| --- | --- | --- | --- |
| Recruiter screen | 30 min | Background, motivation, compensation | Conversation |
| Coding (LeetCode-style) | 45–60 min | Data structures and algorithms, code quality | Two medium problems |
| ML coding | 45–60 min | Implement ML algorithms, tensor ops, numerical stability | KMeans, logistic regression, attention, BN |
| OOP / engineering | 45–60 min | API design, testable code | Design a sampler or feature transform |
| ML fundamentals | 45–60 min | Statistics, optimization, model details, evaluation | Follow-up-driven Q&A |
| ML model design | 45–60 min | Business to model to production | Recsys / search / risk / LLM app |
| ML infra design | 45–60 min | Training / inference / feature infrastructure | Pipeline walk-through or component design |
| AI-assisted coding | 30–45 min | Solving an engineering task with AI tools | Complete a small project with Copilot |
| Behavior | 45 min | Collaboration, conflict, ownership, impact | STAR stories |
| Project deep dive | 45–60 min | Whether you truly understand your work | Five layers of follow-ups |
| Team match | 30 min | Mutual fit | Two-way conversation |

**Strategy**: confirm the round mix first (ask the recruiter or read interview reports), then spend 70% of your time on the two or three rounds with the highest weight.

## 2. How to prepare for each round

### 2.1 Coding (LeetCode-style)

- **Learn patterns, not counts**: two pointers, sliding window, binary search, heap, graph traversal, DP, prefix sums, topological sort. 10–15 problems per pattern covers most mediums.
- **Connect to real systems**: for any problem, be able to say what it maps to in a recommender (e.g., top-k as recall truncation). Interviewers like that transfer.
- **Talk while coding**: state the approach and complexity, write, then run an example. Silent coding is a common deduction.
- **Code quality**: naming, edge cases, empty input, duplicates.

### 2.2 ML coding

Frequent problems:

| Problem | What it tests |
| --- | --- |
| KMeans | Distance, iteration, empty clusters, convergence |
| Logistic regression | Gradient derivation, numerical stability (log-sum-exp) |
| Batch norm / layer norm | Train vs inference, dimensions |
| Attention / multi-head | Matrix shapes, mask, scaling |
| Sampling (top-k / top-p / temperature) | Probability handling, edge cases |
| AUC | Sorting, ties, O(n log n) |
| Cross-entropy / softmax | Numerical stability, label smoothing |
| Convolution output size | Formula and padding |

Preparation: hand-write each algorithm once in PyTorch or NumPy, paying attention to dtype, device, numerical stability, and empty input.

### 2.3 ML fundamentals

- Self-quiz with the Q&A in [chapter 1](../01-ml-fundamentals/README.md) and [chapter 2](../02-dl-fundamentals/README.md).
- **Follow-up chains** matter: "Why AUC?" → "What does AUC mean probabilistically?" → "Is AUC reliable with class imbalance?" → "What would you use instead?".
- Never bluff: state the boundary of what you know, then reason about the rest.

### 2.4 ML model design

**Universal framework (45-minute version)**:

1. **Clarify (5 min)**: who the users are, scale, latency/cost constraints, success metric, cold start.
2. **Objectives and metrics (5 min)**: business metric → model objective → offline metric → online metric, and why they map that way.
3. **Data and labels (8 min)**: sample source, positive/negative definition, label delay, sampling bias, leakage.
4. **Features (8 min)**: user/item/context/cross features, real-time vs offline, feature-platform reuse.
5. **Model (8 min)**: baseline → advanced; multi-objective/multi-task; why this choice.
6. **Evaluation and launch (6 min)**: offline eval, A/B, rollout, monitoring, rollback.
7. **Wrap-up (5 min)**: bottlenecks, trade-offs, next steps.

See the [chapter 6 framework](../06-ml-system-design/01-framework-and-methodology.md).

### 2.5 ML infra design

Two question types:

- **Pipeline walk-through**: given a product (e.g., a feed), explain from request arrival to log return and retraining. Focus on component boundaries, data flow, latency/throughput, consistency.
- **Component design**: feature store, real-time features, distributed training, GPU serving. Focus on abstractions, interfaces, and failure modes.

See [chapter 7](../07-ml-infra/README.md).

### 2.6 Behavior

Prepare 6–8 reusable stories, each with a 2-minute and a 5-minute version, covering:

| Theme | Story essentials |
| --- | --- |
| Conflict | Disagreement with PM or peers; how you used data to persuade or compromise |
| Failure | A failed project, your responsibility, what you learned |
| Initiative | Something nobody asked for that produced impact |
| Ambiguity | How you decomposed a vague goal |
| Leadership | Cross-team push, mentoring |
| Prioritization | Trade-offs across multiple tasks |
| Highest impact | Quantified results |
| Technical depth | The hardest technical problem you solved |

**STAR is only the skeleton**; the credit comes from your judgment and "what I would do differently".

### 2.7 Project deep dive

The interviewer follows "why → how → impact → what if" five layers deep. Template:

1. **Context and goal**: business problem, baseline, your role.
2. **Key decisions**: 2–3 trade-offs (data/model/engineering) and why you chose A over B.
3. **Results**: offline and online metrics; whether attribution is credible.
4. **Failures and retrospective**: what did not work and why.
5. **Boundaries**: what you would change if data were 10× larger, latency halved, or labels arrived a week late.

**Red lines**: do not claim team work as your own; do not invent numbers; do not be unable to explain your own module.

## 3. Timeline (T-4 weeks to interview)

| Time | Task |
| --- | --- |
| T-4 weeks | Confirm roles and rounds; scan the cheat sheets for 01/02/04 and identify weak spots |
| T-3 weeks | Fix weak chapters; two ML coding problems a day; draft three project stories |
| T-2 weeks | System design focus: one design a day, 45 minutes timed; finalize behavior stories |
| T-1 week | Two or three mock interviews (with a peer or recorded); review only cheat sheets and mistakes |
| Interview week | Learn nothing new; sleep; spend 30 minutes after each interview on a retrospective |

## 4. Retrospective table (fill after every interview)

| Round | Questions asked | My performance (1–5) | Where I got stuck | Improvement |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Be specific — "the sentence where I failed to explain X", not "review more".

## 5. Questions to ask

Prepare 3–5 questions that show you care about the team and growth:

- What is the team's most important goal right now, and which part does this role own?
- How long does it usually take a model to go from experiment to production? Where is the bottleneck?
- How does the team do evaluation and retrospectives? Is there an on-call rotation?
- What would you expect me to deliver in the first six months?

## 6. References

- [MLE Interview 2.0 · Yuan Meng](https://www.yuan-meng.com/posts/mle_interviews_2.0/) — round breakdown, preparation strategy, emergency-prep appendix
- [Preparing for ML Infra Interviews · Yuan Meng](https://www.yuan-meng.com/posts/ml_infra_interviews/)
- [MLE Interview · longxingtan](https://longxingtan.gitbook.io/mle-interview)
