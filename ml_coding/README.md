# ML Coding Practice Suite

Zero-dependency pure-Python implementations and unit tests for the canonical **ML Coding** interview problems listed in [docs/00-roadmap/02-interview-preparation-playbook.md](../docs/00-roadmap/02-interview-preparation-playbook.md) and [docs/08-interview-bank/06-ml-coding-cheatsheet.md](../docs/08-interview-bank/06-ml-coding-cheatsheet.md).

## Covered algorithms

- Numerically stable `log_softmax`, `softmax`, and `cross_entropy`
- Binary `fit_logistic_regression` (piecewise-stable `sigmoid` + L2 without bias penalty)
- `kmeans` with empty-cluster recovery
- $O(N \log N)$ `roc_auc` with exact average-rank tie handling
- `layer_norm` and `rms_norm`
- `scaled_dot_product_attention` with causal masking
- `sample_next_token` (temperature, top-$k$, top-$p$ nucleus filtering)
- Minimal `train_bpe` merge trainer
- `conv2d_output_shape` calculator

## Run locally

```bash
python3 -m unittest discover -s ml_coding -v
```
