# Language models and tokenization

> **One-line summary**: A language model factorizes the joint probability autoregressively, $p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})$, and is trained with cross-entropy; tokenization decides the granularity of "language" the model sees and directly affects capability, sequence length, and inference cost.
>
> **Keywords**: `autoregressive` `cross-entropy` `perplexity` `BPE` `WordPiece` `Unigram` `SentencePiece` `chat template`

## 1. Core concepts

### 1.1 What a language model computes

A language model (LM) scores a token sequence $x_{1:T}$; at heart it models the joint distribution:

$$
p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1})
$$

This is the **exact** chain rule - no approximation is made. The approximation happens in "which function represents the conditional probability":

| Modeling approach | Conditional form | Limitation |
| --- | --- | --- |
| n-gram | counts of the previous $n-1$ tokens only | parameters explode with $n$, sparse, cannot generalize |
| Neural LM (NNLM/RNN) | compresses history into a hidden state $h_t$ | fixed window or serial, weak long-range dependencies |
| Transformer LM | attention looks at the whole history directly | $O(T^2)$ compute; inference relies on KV cache |

### 1.2 From n-gram to neural language models

- **n-gram**: estimate $p(x_t\mid x_{t-n+1:t-1})$ from counts with smoothing (Kneser-Ney and friends). The problem is that long-tail combinations are almost never observed, and sparsity explodes as $n$ grows.
- **NNLM (Bengio 2003)**: word embeddings + a fixed-window concatenation + an MLP to predict the next word. The first use of **distributed representations** to generalize; one of the intellectual roots of "pretrain + fine-tune".
- **RNN/LSTM**: hidden-state recursion; in theory it can see an unbounded history, but serial training is slow and long-range gradients decay.
- **Transformer**: all positions are processed in parallel during training, and attention sees the full history in one step; the price is $O(T^2)$, and inference must cache past K/V.

Interviewers often compress this line into one sentence: **"n-grams solve counting, NNLM solves generalization, RNNs solve variable length, Transformers solve parallelism."**

### 1.3 What tokenization is and why it matters

A tokenizer maps strings to integer id sequences. It has no learnable parameters of its own (except Unigram, which carries a probability table). It decides three things:

1. **Sequence length**: for the same sentence, finer splits mean more tokens, which makes both training and inference more expensive (attention is $O(T^2)$).
2. **Vocabulary size $V$**: the embedding has $V\times d$ parameters, and so does the output layer (often sharing weights with the embedding).
3. **Semantic granularity**: whether a token is a character, a word, or a subword affects the compositionality and multilingual ability the model learns.

### 1.4 Four subword algorithms

- **BPE (Byte Pair Encoding)**: start from characters and repeatedly merge the most frequent adjacent pair. GPT, Llama, and Qwen all use the BPE family.
- **WordPiece**: the merge criterion is not frequency but "likelihood gain"; used by BERT, with subwords prefixed by `##`.
- **Unigram LM**: start from a very large candidate vocabulary, estimate each token's probability with EM, then prune gradually. Used by T5 and ALBERT.
- **SentencePiece**: not an algorithm but a **tool**; it treats spaces as characters (`▁`), supports both BPE and Unigram modes, takes raw text directly, and is friendly to Chinese/Japanese.

## 2. Principles and derivations

### 2.1 Cross-entropy, negative log-likelihood, and perplexity

Training maximizes the log-likelihood, which is equivalent to minimizing cross-entropy. For a sequence of length $T$:

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
=\frac{1}{T}\sum_{t=1}^{T}\mathrm{CE}\big(\text{one-hot}(x_t)\,\|\,p_\theta(\cdot\mid x_{<t})\big)
$$

**Perplexity (PPL)** is defined as $\mathrm{PPL}=\exp(\mathcal{L})$. It reads as "how many equally likely candidates the model hesitates among at each step": if the model is uniform over the next token, then $\mathrm{PPL}=V$; lower is better.

Converting between the two: when $\mathcal{L}$ uses natural logs, $\mathrm{PPL}=e^{\mathcal{L}}$; in bits, $\text{bits/token}=\mathcal{L}/\ln 2$ and $\mathrm{PPL}=2^{\text{bits/token}}$.

### 2.2 The BPE algorithm

Training phase (given a target vocabulary size $V$):

```python
# Minimal BPE training: split words into symbols, repeatedly merge the most frequent pair
from collections import Counter

def train_bpe(words, vocab_size, base_vocab=256):
    # words: list[list[str]], each word already split into characters; base_vocab is the initial char/byte vocabulary
    merges = []
    while len(merges) + base_vocab < vocab_size:
        pairs = Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        pair, freq = pairs.most_common(1)[0]
        if freq < 2:            # no repeated pair left to merge, stop
            break
        # Merge the pair and record the rule (merges are ordered; encoding applies them by rank)
        new_words = []
        for w in words:
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            new_words.append(out)
        words = new_words
        merges.append(pair)
    return merges
```

Encoding phase: split the input into characters, then repeatedly merge the adjacent pair with the **highest merge priority (learned earliest)** until nothing can be merged. Production implementations store merge rules as a rank dictionary, making encoding roughly linear.

### 2.3 WordPiece vs Unigram

- **WordPiece**: for a candidate pair $(a,b)$ compute $\text{score}=\dfrac{\text{freq}(ab)}{\text{freq}(a)\cdot\text{freq}(b)}$ and merge the pair with the largest gain. The intuition is that merging is only worthwhile when both symbols appear often on their own but tend to appear together. Compared with BPE's raw frequency, this makes it harder to glue frequent stopwords into oversized tokens.
- **Unigram**: assume the sequence is generated from tokens independently, $p(x)=\prod_i p(t_i)$, and use EM to estimate $p(t)$ over "all segmentations under the current vocabulary"; the E step finds the best segmentation with Viterbi, and the M step re-estimates the probabilities; finally prune to the target vocabulary by "how much likelihood is lost if this token is removed". At inference Viterbi returns the $\arg\max$ segmentation and naturally supports multiple candidates.

### 2.4 Byte-level and SentencePiece

- **byte-level BPE**: first map UTF-8 bytes to 256 base symbols, then run BPE. The benefit is **no OOV ever** (any string can be represented); the cost is that a CJK character may be split into 3 byte tokens. GPT-2 used it to handle emoji and rare characters in web text.
- **SentencePiece**: trains directly on raw text, encodes spaces as `▁` and restores them at decode time; it supports byte fallback (unknown characters fall back to bytes) and NFKC normalization. For Chinese, SentencePiece avoids error propagation from word segmentation before tokenization.

### 2.5 Trade-offs in vocabulary size

| Vocabulary size | Pros | Costs |
| --- | --- | --- |
| Small (about 30k) | small embedding/output layers, long-tail tokens trained well | long sequences, slow inference, poor compression for multilingual/code |
| Large (100k-250k) | short sequences, high throughput, good multilingual coverage | large embedding, heavy softmax, undertrained low-frequency tokens |

Concrete numbers: GPT-2/3 about 50k; Llama-2 32k; Llama-3 128k; Qwen2.5 about 150k; DeepSeek-V3 about 100k; Gemma-2 256k. The trend is **larger vocabularies with better multilingual coverage**, because a large model can afford the embedding share, while shorter sequences pay off directly in training and inference.

### 2.6 Chinese text and embedding details

- Chinese has no spaces, so vocabulary quality directly determines the compression ratio. Llama-2's 32k vocabulary is English-heavy, and a Chinese character is often split into 1-2 tokens; vocabularies expanded to 128k+ (Qwen, Llama-3) include common CJK characters and words, so the same Chinese text needs noticeably fewer tokens.
- **Weight tying**: the input embedding $E\in\mathbb{R}^{V\times d}$ is reused as the output projection, saving parameters and usually helping small models; large models often do it too (GPT-2, T5), while the Llama family does not.
- **Special tokens**: BOS/EOS/PAD, `<|im_start|>`, `<|endoftext|>`. During training EOS teaches the model to stop, and the chat template packs multi-turn dialogue into a fixed format, see 2.7.

### 2.7 Chat template: training and inference must match

A dialogue model is not fed raw natural language; it is assembled with a template:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hello, how can I help you?<|im_end|>
```

During training the loss is computed only on the **assistant spans** (user/system labels are set to -100); at inference the same template is used and the prompt ends with the assistant start token. A template mismatch is the classic cause of "the model got worse after fine-tuning": the delimiters seen during training are absent at inference.

## 3. Implementation notes

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
msgs = [{"role": "user", "content": "Explain BPE"}]

# Recommended: use the official chat template so training and inference agree
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids

# Measure compression: characters covered per token on Chinese text
sample = "The vocabulary size of an LLM affects inference cost."
print(len(sample) / len(tok(sample).input_ids))  # higher is better
```

When **expanding a tokenizer's vocabulary**, call `resize_token_embeddings` as well; initializing new token embeddings with the mean of the old embeddings is more stable than random initialization, and you must keep training afterwards, otherwise the new tokens are just random vectors.

## 4. Interview Q&A

### Q1. Why does perplexity equal exp(cross-entropy)? What is the intuition?
**Answer**: Because $\mathrm{PPL}=\exp\left(-\frac{1}{T}\sum_t \log p(x_t)\right)$, and the term in parentheses is exactly the average cross-entropy. Intuitively, PPL is the model's "effective number of candidates" at each step: the more certain the model, the closer PPL is to 1; when the model is guessing uniformly, PPL equals the vocabulary size $V$. That is why PPL is only comparable under the same tokenizer - comparing PPL across vocabularies is meaningless.
**Follow-up**: Why can't PPL be compared across vocabularies? Different granularity means different information per token; for cross-vocabulary comparison use bits/byte or bits/char.

### Q2. What is the difference between BPE, WordPiece, and Unigram? Why do LLMs mostly use BPE?
**Answer**: All three are subword segmenters; they differ in the merge criterion. BPE merges greedily by frequency; WordPiece merges by likelihood gain and adds the `##` marker; Unigram starts from a large vocabulary and uses EM + pruning for probabilistic segmentation. LLMs prefer BPE (especially byte-level BPE) because it is simple to implement, deterministic at encoding time, and can cover any byte sequence, which suits large multilingual corpora.
**Follow-up**: What are Unigram's advantages? It gives segmentation probabilities, enables sampling-based tokenization, and is more robust to noise, but training and encoding are heavier.

### Q3. Is a larger vocabulary always better? How do you choose?
**Answer**: No. A larger vocabulary shortens sequences, improves throughput and multilingual coverage, but the embedding/output parameters grow linearly, low-frequency tokens are undertrained, and softmax gets more expensive. The practical approach is to train or reuse a large vocabulary from the target corpora (including code and multiple languages) and evaluate by "compression ratio (tokens/character) x parameter share"; mainstream LLMs land in the 100k-250k range.
**Follow-up**: Why shouldn't small models use huge vocabularies? Their parameter budget is small, and an oversized embedding would eat capacity from the Transformer layers.

### Q4. Why not character-level or word-level tokenization?
**Answer**: Character-level sequences are far too long (attention is $O(T^2)$ and blows up), and a single character carries little semantics; word-level vocabularies would be enormous, suffer badly from OOV, and fail on new words and spelling variants. Subwords are the compromise: common words become one token, rare words are split into subwords, balancing length and generalization.
**Follow-up**: What about pure byte level? It never has OOV, but Chinese/emoji sequences become extremely long and expensive to train and serve, so the common choice is byte-level BPE (a byte base plus learned high-frequency merges).

### Q5. How does byte-level BPE avoid OOV?
**Answer**: It puts all 256 byte values in the base vocabulary, so any UTF-8 string can be represented and there are no unknown characters; BPE only learns high-frequency byte combinations on top of that base. GPT-2 used it to handle arbitrary web text, emoji, and rare symbols.
**Follow-up**: What is the cost? A CJK character usually takes 3 bytes, and without learned merges it is split very finely, so Chinese vocabularies need enough CJK merges.

### Q6. What are the pitfalls of Chinese tokenizers? How do you evaluate a tokenizer?
**Answer**: Pitfalls include: an English-heavy vocabulary that inflates Chinese token counts (poor inference economics); inconsistent number/date splitting (hurts arithmetic); inconsistent handling of spaces and full/half-width characters; and token competition between languages. Evaluation metrics: compression (chars/token, bytes/token), vocabulary utilization, downstream task quality, and the stability of splitting for critical characters such as digits and code.
**Follow-up**: Why does digit splitting matter? If "1234" and "12" are split inconsistently, the model struggles to learn arithmetic; tokenizers like GPT-4's force digits into fixed-length groups.

### Q7. Why does the chat template matter? What happens if training and inference disagree?
**Answer**: The template defines role delimiters, the system position, and the assistant start marker. During training the model learns "start answering after the assistant marker, stop at the end marker"; a different template at inference is out-of-distribution input, causing failure to stop, role confusion, and repetition. In production the template must be pinned in the tokenizer config and covered by consistency tests.
**Follow-up**: Why do many models use special tokens instead of natural-language delimiters? A special token is a single atom in the vocabulary, never split, so the boundary is more reliable.

### Q8. What must you do after adding new tokens to a tokenizer?
**Answer**: 1) Call `resize_token_embeddings` to grow the embedding and lm_head; 2) initialize the new tokens (mean or small-variance random) to avoid breaking the original distribution; 3) continue pretraining/fine-tuning so the new token representations converge; 4) check that the special-token ids are consistent with the attention-mask and labels logic. Otherwise the new tokens are just random noise.
**Follow-up**: Why can't you add tokens without training? Embeddings are learned; a new id has no semantics, so the model sees noise.

### Q9. What is teacher forcing? How should we understand exposure bias?
**Answer**: Training uses the ground-truth prefix $x_{<t}$ to predict $x_t$ (parallel and stable), which is teacher forcing; at inference the model can only use its own predictions as the prefix, and early errors accumulate - this mismatch is exposure bias. Mitigations: scheduled sampling (mix in model predictions during training), data augmentation, and RL/preference optimization that directly optimizes a sequence-level objective.
**Follow-up**: Why do today's LLMs rarely discuss exposure bias? Scale plus instruction tuning makes per-step errors small, and beam search/sampling can correct them, but it remains the root cause of error accumulation in long generations.

### Q10. Should the training loss average over tokens or over sequences?
**Answer**: The standard is to **average over tokens** (over all valid label tokens), so long sequences are not over-weighted; for multi-task mixtures, if you want each dataset to count equally, normalize per dataset and then weight. Averaging over sequences over-weights short sequences and is generally not recommended.
**Follow-up**: How is padding handled? Set labels to -100; cross-entropy ignores them and they do not enter the denominator.

### Q11. Why do many models tie the embedding and lm_head?
**Answer**: Tying saves the $V\times d$ parameters (substantial with a large vocabulary) and aligns the input and output semantic spaces, which usually gives a regularization benefit on small models. Whether large models tie is an engineering trade-off: the Llama family does not, T5/GPT-2 do, and the difference is small.
**Follow-up**: How does tying affect quantization? When lm_head and embedding share weights, quantization error affects both input and output, so they must be evaluated together.

## 5. Pitfalls and follow-ups

- **PPL cannot be compared across tokenizers**: information per token differs, so switch to bits/byte.
- **BPE merge order is the encoding priority**: it is not "merge every occurrence", but trying learned ranks from low to high.
- **`##` is WordPiece's "non-word-initial" marker**, not BPE's; BPE commonly uses `Ġ` (a space).
- **Special tokens are not ordinary tokens**: they have dedicated semantics in attention/loss, and adding them carelessly breaks the format.
- **Higher Chinese compression is not always better**: over-merging common phrases hurts compositional generalization, so the vocabulary should cover both characters and words.

## 6. Cheat sheet

| Concept | One-liner |
| --- | --- |
| Autoregressive factorization | $p(x)=\prod_t p(x_t\mid x_{<t})$, chain rule, no approximation |
| Training objective | Minimize token-averaged cross-entropy = maximize log-likelihood |
| PPL | $\exp(\text{CE})$, comparable within a vocabulary, use bits/byte across vocabularies |
| BPE | Greedy frequency merges; merge rank determines encoding |
| WordPiece | Likelihood-gain merges; `##` marks non-word-initial |
| Unigram | EM probabilities + Viterbi segmentation + pruning |
| SentencePiece | A tool; `▁` encodes a space; supports byte fallback |
| byte-level BPE | 256-byte base, never OOV |
| chat template | Training and inference must match, otherwise out-of-distribution |
| Vocabulary trade-off | Large vocabulary shortens sequences but costs parameters; mainstream 100k-250k |

## 7. References

- [Neural Probabilistic Language Models (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909) - the classic paper that brought BPE to NMT
- [Google's Neural Machine Translation System (WordPiece)](https://arxiv.org/abs/1609.08144)
- [Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tiktoken](https://github.com/openai/tiktoken)
- [llm_interview_note · tokenization and word vectors](https://github.com/wdndev/llm_interview_note)
