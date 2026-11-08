# MLE Learning Notes

> Personal study notes for ML / DL / RL / LLM / Agent interviews and long-term review.
> Every topic follows the same structure — **core concepts → derivations → implementation notes → interview Q&A → pitfalls → cheat sheet** — so the notes are easy to read, derive from, and explain out loud.

## Where these notes come from

The notes reorganize and condense the following public resources (credit to their authors):

| Area | Source |
| --- | --- |
| ML / DL interview fundamentals | [MachineLearningFAQ · bagu.md](https://github.com/nxpeng9235/MachineLearningFAQ/blob/main/bagu.md) |
| General MLE interviews | [MLE Interview (longxingtan)](https://longxingtan.gitbook.io/mle-interview) |
| LLM fundamentals | [llm_interview_note](https://github.com/wdndev/llm_interview_note), [LLM-Agent-Interview-Guide](https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide) |
| Agent interviews | [AI Agent Interview Guide](https://github.com/bcefghj/ai-agent-interview-guide) |
| ML system design | [MLE Interview 2.0 · Yuan Meng](https://www.yuan-meng.com/posts/mle_interviews_2.0/) |
| ML infra | [Preparing for ML Infra Interviews · Yuan Meng](https://www.yuan-meng.com/posts/ml_infra_interviews/) |
| Chinese interview collections | [Zhihu · algorithm interview questions](https://zhuanlan.zhihu.com/p/429901476), [CSDN · machine learning interview questions](https://blog.csdn.net/qq_18822147/article/details/120243772) |

All chapters are **rewritten and condensed notes** (not copied verbatim) for personal study.

## Structure

| Chapter | Content | When to use it |
| --- | --- | --- |
| [00 · Learning Map](00-roadmap/README.md) | Roadmap, interview methodology, knowledge map | Read this first |
| [01 · ML Fundamentals](01-ml-fundamentals/README.md) | Bias-variance, classical models, trees, evaluation, sampling, EM | First round for every ML role |
| [02 · DL Fundamentals](02-dl-fundamentals/README.md) | Backprop, initialization, normalization, CNN/RNN, Transformer basics, optimizers | Derivations and detail questions |
| [03 · Reinforcement Learning](03-reinforcement-learning/README.md) | MDP, value methods, policy gradients, PPO, RLHF, DPO, GRPO | Alignment / reasoning roles |
| [04 · Large Language Models](04-llm/README.md) | Architecture, pretraining, scaling laws, fine-tuning, distributed training, inference, quantization, evaluation, RAG, prompting | Core LLM track |
| [05 · AI Agents](05-agents/README.md) | Agent architecture, tool use, planning, memory, multi-agent systems, engineering | Agent / application roles |
| [06 · ML System Design](06-ml-system-design/README.md) | Answering framework plus recommender / search / ads / risk / LLM application designs | System design rounds |
| [07 · ML Infra](07-ml-infra/README.md) | Feature stores, real-time features, training and serving infrastructure | Infra rounds |
| [08 · Interview Bank](08-interview-bank/README.md) | Cheat sheets and top 100 questions | The day before the interview |

## How to use these notes

1. **Build the frame first**: read the [knowledge map](00-roadmap/03-knowledge-map.md) and each chapter README to see what problem every block of knowledge solves.
2. **Read a chapter, then answer its Q&A without looking**: if you cannot, go back to the derivation.
3. **Write your own one-liner**: the "Cheat sheet" at the end of each note is the smallest unit you should be able to say out loud in an interview; rewrite it in your own words.
4. **Three days before an interview**: review only the [cheat sheets](08-interview-bank/README.md) and the [top 100 questions](08-interview-bank/04-top-100-questions.md).
5. **After an interview**: add every question you fumbled to the relevant chapter, with the date and company type.

## Conventions

- English terms are given in full on first use; formulas are LaTeX; architecture diagrams use Mermaid.
- Every Q&A is organized as "conclusion → reason → follow-up" so you can answer directly.
- Chapters cross-reference each other with relative links to avoid duplication: each concept is expanded in the one chapter where it belongs.

## Disclaimer

These notes are a personal re-organization of public material and may contain errors or outdated information. For specific numbers (model sizes, memory usage, benchmark results), always check the paper or official documentation.
