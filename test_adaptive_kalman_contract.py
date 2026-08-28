import unittest

import torch

from adaptive_kalman_motion import (
    AdaptiveKalmanLSTM,
    AdaptiveKalmanTransformer,
    build_prediction_gaps,
)


class AdaptiveKalmanContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.src = torch.rand(2, 6, 15)
        self.trg = torch.rand(2, 4, 15)
        self.src[..., 13] = 0.0
        self.trg[..., 13] = 0.0
        self.src[..., 14] = 1.0
        self.trg[..., 14] = 1.0

    def _models(self):
        return (
            AdaptiveKalmanTransformer(
                d_model=16, nhead=4, num_layers=1, dim_ff=32,
                dropout=0.0, max_gap_norm=30.0,
            ).eval(),
            AdaptiveKalmanLSTM(
                d_model=16, hidden_dim=16, num_layers=1,
                dropout=0.0, max_gap_norm=30.0,
            ).eval(),
        )

    def test_first_training_step_equals_two_stage_inference(self):
        prediction_gap = build_prediction_gaps(self.src, self.trg, 30.0)[:, 0, 0]
        for model in self._models():
            with self.subTest(model=type(model).__name__), torch.no_grad():
                log_q, log_r = model(self.src, self.trg)
                q_one = model.predict_q(self.src, prediction_gap)
                r_one = model.predict_r(self.src, self.trg[:, :1])
                torch.testing.assert_close(log_q[:, 0], q_one, atol=2e-6, rtol=1e-6)
                torch.testing.assert_close(log_r[:, 0], r_one, atol=2e-6, rtol=1e-6)

    def test_q_cannot_see_current_or_future_measurements(self):
        changed = self.trg.clone()
        changed[:, 0:, :13] = torch.randn_like(changed[:, 0:, :13]) * 100.0
        for model in self._models():
            with self.subTest(model=type(model).__name__), torch.no_grad():
                q_original, _ = model(self.src, self.trg)
                q_changed, _ = model(self.src, changed)
                torch.testing.assert_close(
                    q_original[:, 0], q_changed[:, 0], atol=2e-6, rtol=1e-6
                )

    def test_prediction_gap_is_one_step_ahead_of_history(self):
        src = self.src[:1].clone()
        trg = self.trg[:1].clone()
        src[:, -1, 13] = 0.0
        trg[:, 0, 13] = 1.0 / 30.0
        trg[:, 1, 13] = 2.0 / 30.0
        gaps = build_prediction_gaps(src, trg, 30.0)[0, :, 0]
        torch.testing.assert_close(
            gaps[:3], torch.tensor([1.0, 2.0, 3.0]) / 30.0
        )

    def test_kalman_head_depth_and_terminal_initialisation(self):
        models = (
            AdaptiveKalmanTransformer(
                d_model=16, nhead=4, num_layers=1, dim_ff=32,
                dropout=0.0, kalman_head_layers=4,
            ),
            AdaptiveKalmanLSTM(
                d_model=16, hidden_dim=16, num_layers=1,
                dropout=0.0, kalman_head_layers=1,
            ),
        )
        for model, expected_depth in zip(models, (4, 1)):
            with self.subTest(model=type(model).__name__):
                q_linears = [layer for layer in model.head.q_head if isinstance(layer, torch.nn.Linear)]
                r_linears = [layer for layer in model.head.r_residual_head if isinstance(layer, torch.nn.Linear)]
                self.assertEqual(len(q_linears), expected_depth)
                self.assertEqual(len(r_linears), expected_depth)
                torch.testing.assert_close(q_linears[-1].bias, torch.full((4,), -12.0))
                torch.testing.assert_close(r_linears[-1].weight, torch.zeros_like(r_linears[-1].weight))
                torch.testing.assert_close(r_linears[-1].bias, torch.zeros_like(r_linears[-1].bias))


if __name__ == '__main__':
    unittest.main()
