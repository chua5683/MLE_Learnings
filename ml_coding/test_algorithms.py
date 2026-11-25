import math
import random
import unittest

from ml_coding.algorithms import (
    conv2d_output_shape,
    cross_entropy,
    fit_logistic_regression,
    kmeans,
    layer_norm,
    log_softmax,
    rms_norm,
    roc_auc,
    sample_next_token,
    scaled_dot_product_attention,
    sigmoid,
    softmax,
    train_bpe,
)


class TestMLCodingAlgorithms(unittest.TestCase):
    def test_softmax_and_cross_entropy_numerical_stability(self):
        probs = softmax([1000.0, 1001.0, 999.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertGreater(probs[1], probs[0])
        ce = cross_entropy([[1000.0, 1000.0], [10.0, 0.0]], [0, 0])
        self.assertAlmostEqual(ce, 0.5 * (math.log(2.0) + math.log(1.0 + math.exp(-10.0))), places=6)
        self.assertEqual(len(log_softmax([0.0, 0.0])), 2)

    def test_logistic_regression_linearly_separable(self):
        X = [[-2.0, -1.0], [-1.5, -2.0], [2.0, 1.0], [1.5, 2.0]]
        y = [0, 0, 1, 1]
        w, b = fit_logistic_regression(X, y, lr=0.5, l2=1e-4, steps=200)
        preds = [1 if sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b) >= 0.5 else 0 for x in X]
        self.assertEqual(preds, y)

    def test_kmeans_clusters_and_convergence(self):
        X = [[0.0, 0.0], [0.1, -0.1], [-0.1, 0.2], [10.0, 10.0], [10.2, 9.9], [9.8, 10.1]]
        centers, labels = kmeans(X, k=2, seed=7)
        self.assertEqual(len(centers), 2)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_roc_auc_exact_and_ties(self):
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertAlmostEqual(roc_auc([0, 1, 1, 0], [0.2, 0.5, 0.5, 0.9]), 0.5)
        with self.assertRaises(ValueError):
            roc_auc([1, 1, 1], [0.2, 0.5, 0.8])

    def test_layer_norm_and_rms_norm(self):
        ln = layer_norm([1.0, 2.0, 3.0, 4.0], [1.0] * 4, [0.0] * 4, eps=0.0)
        self.assertAlmostEqual(sum(ln) / 4.0, 0.0, places=9)
        self.assertAlmostEqual(sum(v * v for v in ln) / 4.0, 1.0, places=9)
        rn = rms_norm([3.0, 4.0], [1.0, 1.0], eps=0.0)
        expected_rms = math.sqrt((9.0 + 16.0) / 2.0)
        self.assertAlmostEqual(rn[0], 3.0 / expected_rms, places=9)

    def test_causal_attention_blocks_future_tokens(self):
        Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        V_a = [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]
        V_b = [[1.0, 0.0], [0.0, 1.0], [-99.0, -99.0]]
        out_a = scaled_dot_product_attention(Q, K, V_a, causal=True)
        out_b = scaled_dot_product_attention(Q, K, V_b, causal=True)
        # Token 0 and token 1 must be strictly invariant to changes at future token 2
        for pos in (0, 1):
            for c in (0, 1):
                self.assertAlmostEqual(out_a[pos][c], out_b[pos][c], places=9)

    def test_sampling_temperature_topk_topp(self):
        self.assertEqual(sample_next_token([0.1, 2.5, 1.0], temperature=0.0), 1)
        rng = random.Random(0)
        picks = {sample_next_token([10.0, 9.0, -10.0, -10.0], top_k=2, rng=rng) for _ in range(30)}
        self.assertTrue(picks.issubset({0, 1}))
        pick_p = sample_next_token([10.0, 0.0, 0.0], top_p=0.5, rng=random.Random(1))
        self.assertEqual(pick_p, 0)

    def test_bpe_merges_and_conv_shape(self):
        merges = train_bpe(["low", "low", "low", "lower", "lo", "newest", "widest"], num_merges=3)
        self.assertEqual(len(merges), 3)
        self.assertEqual(merges[0], ("l", "o"))
        self.assertEqual(conv2d_output_shape(32, 32, (3, 3), stride=(2, 2), padding=(1, 1)), (16, 16))


if __name__ == "__main__":
    unittest.main()
