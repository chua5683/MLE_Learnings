# Learning Roadmap

> **One-line summary**: Pick the minimum reading list for your target role first, then cover it in eight weeks, using "I can explain it" as the weekly acceptance test.
>
> **Keywords**: `role profiles` `8-week plan` `minimum reading` `study rhythm`

## 1. Start from the target role

The same body of knowledge carries very different weight across roles. Below are four profiles distilled from common 2025–2026 job descriptions:

| Role | Typical work | High-weight chapters | Lower-weight chapters |
| --- | --- | --- | --- |
| **Traditional MLE** (recsys / ads / search / risk) | Features, model iteration, A/B tests, offline-online consistency | 01, 02, 06, the pipeline part of 07 | 03; inference details in 04 |
| **LLM application engineer** | RAG, agents, prompting, evaluation, serving | 04 (08–12), 05, 06-05 | Tree-model derivations in 01; 07 |
| **LLM training / inference engineer** | Pretraining, fine-tuning, distributed training, inference optimization | 02, 03, 04 (01–09), 07 | Business design in 06 |
| **Agent / application architect** | Tool use, multi-agent systems, engineering | 04, 05, 06-05 | Formula-heavy parts of 01 and 02 |

**Advice**: apply mainly to one profile and secondarily to one more. The primary profile drives your 8-week plan; the secondary only needs its minimum reading list in the final two weeks.

## 2. Eight-week plan (2–3 hours/day while employed)

| Week | Topic | Required reading | Deliverable (acceptance) |
| --- | --- | --- | --- |
| 1 | ML fundamentals | All of 01 | Derive bias-variance; explain XGBoost vs GBDT; write 10 Q&A |
| 2 | DL fundamentals | All of 02 | Derive backprop and BN; write self-attention; compare optimizers |
| 3 | LLM architecture and pretraining | 04-01 to 04-05 | Draw Llama; compute KV cache; explain Chinchilla |
| 4 | Fine-tuning and distributed training | 04-06, 04-07 | Explain LoRA/QLoRA; compute ZeRO memory; explain TP/PP |
| 5 | Inference and evaluation | 04-08 to 04-10 | Explain PagedAttention and speculative decoding; design an eval plan |
| 6 | RAG, prompting, agents | 04-11, 04-12, all of 05 | Design a RAG system; write a tool schema |
| 7 | System design | All of 06 plus 07-01/02/04 | Two 45-minute designs; draw a feature store |
| 8 | Final sprint | 03 (role-dependent), chapter 08 | Three mock interviews; finalize project stories; review mistakes |

**Full-time study** can compress this to 4–5 weeks: six hours a day — read in the morning, write Q&A in the afternoon, do mock explanations in the evening.

## 3. Weekly execution template

1. **Read (40%)**: go in chapter order and mark anything you cannot follow or derive.
2. **Write (30%)**: close the Q&A and answer in your own words; if you cannot, go back.
3. **Explain (20%)**: 15 minutes at a whiteboard or recorded, focusing on derivations and trade-offs.
4. **Test (10%)**: self-quiz with the [top 100 questions](../08-interview-bank/04-top-100-questions.md); log missed questions in the relevant chapter.

## 4. Minimum reading list (when time is short)

### Traditional MLE
- 01-01 bias-variance, 01-03 tree models, 01-04 regularization, 01-05 evaluation metrics
- 02-01 backprop, 02-03 normalization, 02-06 optimizers
- 06-01 answering framework, 06-02 recommender systems, 07-01 overview, 07-02 feature platform

### LLM applications
- 04-01 tokenization, 04-02 architecture, 04-05 scaling laws, 04-06 fine-tuning
- 04-08 inference, 04-10 evaluation, 04-11 RAG, 04-12 prompting
- 05-01 agent fundamentals, 05-02 tool use, 05-04 memory, 05-07 engineering
- 06-05 LLM application system design

### LLM training / inference
- 02-05 Transformer, 04-02 architecture, 04-03 attention, 04-04 pretraining
- 04-05 scaling laws, 04-06 fine-tuning, 04-07 distributed training
- 04-08 inference, 04-09 quantization, 07-03 training infrastructure, 07-04 GPU serving

## 5. Common pitfalls

- **Reading without writing**: understanding is not derivation. Write the key formulas by hand at least once a week.
- **Following paper order**: papers are written chronologically, which is not how mental models are built. Learn the textbook mainline first, then read papers.
- **Optimizing details too early**: studying the CUDA implementation of FlashAttention before understanding attention's IO complexity is a poor return.
- **Neglecting delivery**: interviews are oral exams. Practice a 30-second version and a 2-minute version of every answer.

## 6. References

- [MLE Interview 2.0 · Yuan Meng](https://www.yuan-meng.com/posts/mle_interviews_2.0/) — interview rounds and preparation strategy (highly recommended)
- [MLE Interview · longxingtan](https://longxingtan.gitbook.io/mle-interview) — question bank and system design cases
- [llm_interview_note](https://github.com/wdndev/llm_interview_note) — LLM topic index
- [AI Agent Interview Guide](https://github.com/bcefghj/ai-agent-interview-guide) — agent learning path
