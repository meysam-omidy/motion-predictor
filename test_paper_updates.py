"""Contracts for the paper-update options (U1 innovation loss, U2 velocity Q,
detection context, label-free tracklet data). Run: python -m pytest -q"""
import argparse
import os
import tempfile
import unittest

import numpy as np
import torch

from adaptive_kalman_dataset import (
    CONTEXT_FEATURE_DIM, FEATURE_DIM, build_adaptive_kalman_features, compute_detection_context,
)
from adaptive_kalman_dataset_tracklets import AdaptiveKalmanTrackletDataset
from adaptive_kalman_motion import (
    AdaptiveKalmanLoss, AdaptiveKalmanLSTM, AdaptiveKalmanTransformer,
    build_cv_innovations, build_prediction_gaps, innovation_likelihood_kf_loss,
)
from train_adaptive_kalman_real import apply_objective


def reference_features(bboxes, scores, observed, max_gap_norm=30.0):
    """The original loop implementation, kept to pin the vectorized builder."""
    bboxes = np.asarray(bboxes, dtype=float).reshape(-1, 4)
    n = len(bboxes)
    feat = np.zeros((n, 15), dtype=np.float32)
    for i in range(n):
        if not observed[i]:
            continue
        feat[i, :4] = bboxes[i]
        feat[i, 12] = scores[i]
        if i >= 1 and observed[i - 1]:
            feat[i, 4:8] = bboxes[i] - bboxes[i - 1]
            if i >= 2 and observed[i - 2]:
                feat[i, 8:12] = (bboxes[i] - bboxes[i - 1]) - (bboxes[i - 1] - bboxes[i - 2])
    gap = 0.0
    for i in range(n):
        gap = 0.0 if observed[i] else gap + 1.0
        feat[i, 13] = min(gap / max_gap_norm, 1.0)
        feat[i, 14] = 1.0 if observed[i] else 0.0
    return feat


def constant_velocity_windows(batch=64, src_len=10, trg_len=20, noise=0.002, seed=0):
    """Synthetic tracks moving at constant velocity, observed with Gaussian noise."""
    rng = np.random.default_rng(seed)
    total = src_len + trg_len
    sources, targets = [], []
    for _ in range(batch):
        start = rng.uniform(0.2, 0.6, size=2)
        velocity = rng.uniform(-0.004, 0.004, size=2)
        size = rng.uniform(0.05, 0.15, size=2)
        t = np.arange(total)[:, None]
        boxes = np.concatenate([start + velocity * t, np.repeat(size[None], total, 0)], axis=1)
        boxes = boxes + rng.normal(0, noise, size=boxes.shape)
        observed = np.ones(total, dtype=bool)
        feats = build_adaptive_kalman_features(boxes, np.full(total, 0.9), observed)
        sources.append(feats[:src_len]); targets.append(feats[src_len:])
    return (torch.tensor(np.array(sources)), torch.tensor(np.array(targets)))


class FeatureBuilderTest(unittest.TestCase):
    def test_vectorized_builder_matches_original_loop(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            n = int(rng.integers(1, 40))
            boxes = rng.random((n, 4))
            scores = rng.random(n)
            observed = rng.random(n) < rng.uniform(0.2, 1.0)
            np.testing.assert_array_equal(
                build_adaptive_kalman_features(boxes, scores, observed),
                reference_features(boxes, scores, observed),
            )

    def test_context_columns_are_appended_on_observed_rows_only(self):
        boxes = np.random.default_rng(2).random((5, 4))
        observed = np.array([True, False, True, True, False])
        context = np.full((5, 3), 0.5)
        feats = build_adaptive_kalman_features(boxes, np.ones(5), observed, context=context)
        self.assertEqual(feats.shape, (5, CONTEXT_FEATURE_DIM))
        np.testing.assert_array_equal(feats[:, :FEATURE_DIM],
                                      build_adaptive_kalman_features(boxes, np.ones(5), observed))
        np.testing.assert_array_equal(feats[observed, FEATURE_DIM:], 0.5)
        np.testing.assert_array_equal(feats[~observed, FEATURE_DIM:], 0.0)

    def test_detection_context_values(self):
        boxes = np.array([[100, 100, 200, 300], [150, 100, 250, 300], [800, 500, 900, 700],
                          [0, 400, 50, 500]], dtype=float)
        ctx = compute_detection_context(boxes, 1920, 1080)
        self.assertAlmostEqual(ctx[0, 0], ctx[1, 0])
        self.assertAlmostEqual(ctx[0, 0], 50 * 200 / (2 * 100 * 200 - 50 * 200))
        np.testing.assert_array_equal(ctx[:, 1], [0.1, 0.1, 0.0, 0.0])
        np.testing.assert_array_equal(ctx[:, 2], [0.0, 0.0, 0.0, 1.0])


class VelocityQTest(unittest.TestCase):
    def _models(self):
        return (
            AdaptiveKalmanTransformer(d_model=16, nhead=4, num_layers=1, dim_ff=32,
                                      dropout=0.0, q_dim=7).eval(),
            AdaptiveKalmanLSTM(d_model=16, hidden_dim=16, num_layers=1, dropout=0.0, q_dim=7).eval(),
        )

    def test_seven_dim_q_keeps_the_causal_two_stage_contract(self):
        torch.manual_seed(0)
        src, trg = torch.rand(2, 6, 15), torch.rand(2, 4, 15)
        src[..., 13] = trg[..., 13] = 0.0
        src[..., 14] = trg[..., 14] = 1.0
        gap = build_prediction_gaps(src, trg, 30.0)[:, 0, 0]
        for model in self._models():
            with self.subTest(model=type(model).__name__), torch.no_grad():
                log_q, log_r = model(src, trg)
                self.assertEqual(log_q.shape, (2, 4, 7))
                self.assertEqual(log_r.shape, (2, 4, 4))
                torch.testing.assert_close(log_q[:, 0], model.predict_q(src, gap), atol=2e-6, rtol=1e-6)
                changed = trg.clone(); changed[..., :13] += 50.0
                torch.testing.assert_close(model(src, changed)[0][:, 0], log_q[:, 0], atol=2e-6, rtol=1e-6)

    def test_velocity_noise_starts_near_the_fixed_tracker_floor(self):
        model = AdaptiveKalmanTransformer(d_model=16, nhead=4, num_layers=1, dim_ff=32, q_dim=7)
        bias = model.head.q_head[-1].bias
        torch.testing.assert_close(bias[:4], torch.full((4,), -12.0))
        torch.testing.assert_close(bias[4:], torch.full((3,), -18.0))


class InnovationLikelihoodTest(unittest.TestCase):
    def test_loss_is_finite_differentiable_and_ignores_ground_truth(self):
        src, trg = constant_velocity_windows(batch=8)
        model = AdaptiveKalmanTransformer(d_model=16, nhead=4, num_layers=1, dim_ff=32,
                                          dropout=0.0, q_dim=7)
        log_q, log_r = model(src, trg)
        criterion = AdaptiveKalmanLoss(innovation_coeff=0, r_supervise_coeff=0, q_gap_coeff=0,
                                       q_gap_trend_coeff=0, q_easy_coeff=0, innovation_kf_coeff=1.0)
        self.assertFalse(criterion.uses_ground_truth)
        nan_gt_src = torch.full_like(src, float('nan'))
        nan_gt_trg = torch.full_like(trg, float('nan'))
        innovations = torch.zeros(trg.shape[0], trg.shape[1], 4)
        loss, metrics = criterion(log_q, log_r, innovations, trg, nan_gt_trg, nan_gt_src, src=src)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["innov_kf_steps"], 0)
        loss.backward()
        q_grad = model.head.q_head[-1].weight.grad
        r_grad = model.head.r_residual_head[-1].weight.grad
        self.assertTrue(torch.isfinite(q_grad).all() and q_grad.abs().sum() > 0)
        self.assertTrue(torch.isfinite(r_grad).all() and r_grad.abs().sum() > 0)

    def test_true_measurement_noise_has_the_lowest_loss(self):
        noise = 0.002
        src, trg = constant_velocity_windows(batch=64, noise=noise)
        B, T = trg.shape[:2]
        log_q = torch.full((B, T, 4), float(np.log(1e-8)))
        losses = {}
        for scale in (0.01, 1.0, 100.0):
            log_r = torch.full((B, T, 4), float(np.log(scale * noise ** 2)))
            losses[scale], _, _ = innovation_likelihood_kf_loss(log_q, log_r, src, trg, burn_in=3)
        self.assertLess(losses[1.0], losses[0.01])
        self.assertLess(losses[1.0], losses[100.0])

    def test_float32_default_matches_float64(self):
        src, trg = constant_velocity_windows(batch=32)
        B, T = trg.shape[:2]
        log_q = torch.full((B, T, 7), -16.0)
        log_r = torch.full((B, T, 4), float(np.log(0.002 ** 2)))
        loss32, _, n32 = innovation_likelihood_kf_loss(log_q, log_r, src, trg)
        loss64, _, n64 = innovation_likelihood_kf_loss(
            log_q, log_r, src, trg, compute_dtype=torch.float64)
        self.assertEqual(n32, n64)
        self.assertTrue(torch.isfinite(loss32))
        torch.testing.assert_close(loss32, loss64.to(loss32.dtype), atol=1e-3, rtol=1e-4)
        self.assertEqual(AdaptiveKalmanLoss().innovation_kf_dtype, torch.float32)
        self.assertEqual(AdaptiveKalmanLoss(innovation_kf_fp32=False).innovation_kf_dtype,
                         torch.float64)

    def test_gap_steps_are_not_scored_but_still_propagate(self):
        src, trg = constant_velocity_windows(batch=4)
        log_q = torch.full((4, trg.shape[1], 4), -12.0)
        log_r = torch.full((4, trg.shape[1], 4), -12.0)
        _, _, all_steps = innovation_likelihood_kf_loss(log_q, log_r, src, trg, burn_in=0)
        gapped = trg.clone(); gapped[:, 5:8, 14] = 0.0
        _, _, gap_steps = innovation_likelihood_kf_loss(log_q, log_r, src, gapped, burn_in=0)
        self.assertEqual(all_steps - gap_steps, 4 * 3)

    def test_thesis_loss_is_unchanged_when_new_terms_are_off(self):
        torch.manual_seed(3)
        src, trg = constant_velocity_windows(batch=8)
        log_q, log_r = torch.randn(8, 20, 4) - 10, torch.randn(8, 20, 4) - 9
        innovations = build_cv_innovations(src, trg, observed=trg[..., 14])
        base = AdaptiveKalmanLoss()(log_q, log_r, innovations, trg, trg, src)[0]
        with_src = AdaptiveKalmanLoss()(log_q, log_r, innovations, trg, trg, src, src=src)[0]
        torch.testing.assert_close(base, with_src)


class ObjectiveAndTrackletTest(unittest.TestCase):
    def _args(self, **kw):
        base = dict(objective="proxy", innovation_coeff=1.0, r_supervise_coeff=2.0, q_gap_coeff=0.5,
                    q_gap_trend_coeff=1.0, kf_track_coeff=0.0, innovation_kf_coeff=0.0,
                    tracklet_train=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_innovation_objective_removes_every_ground_truth_term(self):
        args = self._args(objective="innovation")
        apply_objective(args)
        self.assertEqual(args.innovation_kf_coeff, 1.0)
        for name in ("innovation_coeff", "r_supervise_coeff", "q_gap_coeff",
                     "q_gap_trend_coeff", "kf_track_coeff"):
            self.assertEqual(getattr(args, name), 0.0)

    def test_tracklets_refuse_ground_truth_terms(self):
        with self.assertRaises(ValueError):
            apply_objective(self._args(tracklet_train=["a|b|c"]))

    def test_tracklet_windows_recover_exact_detections_and_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "seqs", "S1"); os.makedirs(seq)
            with open(os.path.join(seq, "seqinfo.ini"), "w") as f:
                f.write("[Sequence]\nimWidth=1000\nimHeight=500\nseqLength=60\n")
            dets, outs = [], []
            for frame in range(1, 61):
                x1 = 100 + 2 * frame
                dets.append(f"{frame},{x1 + 0.123},{50.456},{x1 + 40.789},{150.0},0.87654")
                if frame not in (35, 36):  # a two-frame gap inside the tracklet
                    outs.append(f"{frame},7,{x1 + 0.1},{50.5},{40.7},{99.5},0.9,-1,-1,-1")
            os.makedirs(os.path.join(root, "out")); os.makedirs(os.path.join(root, "det"))
            open(os.path.join(root, "out", "S1.txt"), "w").write("\n".join(outs))
            open(os.path.join(root, "det", "S1.txt"), "w").write("\n".join(dets))
            runs = AdaptiveKalmanTrackletDataset.discover_runs(
                os.path.join(root, "seqs"), os.path.join(root, "out"), os.path.join(root, "det"))
            ds = AdaptiveKalmanTrackletDataset.from_runs(runs, seq_in_len=30, seq_out_len=20, steps=5)
            self.assertGreater(len(ds), 0)
            src, trg, gt_src, gt_trg = ds[0]
            self.assertTrue(torch.equal(src, gt_src) and torch.equal(trg, gt_trg))
            # exact score recovered from the detection file, not the rounded output
            self.assertAlmostEqual(float(src[0, 12]), 0.87654, places=5)
            # frames 35/36 are gaps: rows 34 and 35 of the first window
            self.assertEqual(float(src[29, 14]), 1.0)
            self.assertEqual(float(trg[4, 14]), 0.0)
            self.assertEqual(float(trg[5, 14]), 0.0)


if __name__ == "__main__":
    unittest.main()
