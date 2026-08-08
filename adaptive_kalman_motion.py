"""
Context-aware adaptive Kalman noise predictor (Q/R only).

Thesis-aligned design (see analysis/THESIS_DIAGNOSTICS_AND_TRACKER_GUIDANCE.md):
- Predict diagonal log-variance for process noise Q and measurement noise R.
- No bbox regression head (kalman_fusion_blend = 0 at tracker integration time).
- R = confidence prior + learned log-space delta (can go above or below prior).
- Training loss: gap-weighted innovation NLL + R supervision + Q gap/trend terms.

Use with ``AdaptiveKalmanDataset`` (15-D features) and ``train_adaptive_kalman.py``.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Variance floor. Normalized-xywh process noise on heavily-interpolated GT
# (e.g. MOT17) is tiny — gap innovation² can sit ~6e-7, i.e. *below* a 1e-6
# floor. A 1e-6 floor forces the model to over-predict variance on low-motion
# data and fails the eval calibration check. 1e-8 gives room for both MOT17
# (slow) and DanceTrack/SportsMOT (fast) scales without underflow.
VAR_FLOOR = 1e-8


def softplus_var(log_v: torch.Tensor, floor: float = VAR_FLOOR) -> torch.Tensor:
    return F.softplus(log_v) + floor


def exp_var(log_v: torch.Tensor, floor: float = VAR_FLOOR) -> torch.Tensor:
    """Recover a positive variance from a log-variance (non-saturating)."""
    return torch.exp(log_v.clamp(min=-20.0, max=10.0)).clamp(min=floor)


def confidence_log_r_prior(
    score: torch.Tensor,
    alpha: float = 2.0,
    base_log_var: float = -9.0,
) -> torch.Tensor:
    """
    Log-variance prior for normalized xywh (typical det. error var ~ 1e-4).

    Matches OC-SORT conf-R shape: low score -> larger R.
    Prior is **additive in log-space** (not exp(2*(1-s)) which is ~O(1) and
    far too large for [0,1]-normalized boxes).
    """
    s = score.clamp(0.0, 1.0)
    return base_log_var + alpha * (1.0 - s)


def constant_velocity_predict(
    prev: torch.Tensor, prev_prev: torch.Tensor
) -> torch.Tensor:
    """One-step CV prediction from two consecutive xywh states. Shapes (..., 4)."""
    return prev + (prev - prev_prev)


def build_cv_innovations(
    gt_src: torch.Tensor,
    gt_trg: torch.Tensor,
    observed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-step innovations y - μ_CV for each target timestep.

    gt_src: (B, S, D), gt_trg: (B, T, D)
    observed: optional (B, T) mask (1 = real detection, 0 = dropped / gap).

    The CV base advances the way a constant-velocity tracker actually behaves:
    on an **observed** step the state is corrected to the (true) detection, but
    during a **gap** there is no measurement, so the filter keeps extrapolating
    its own prediction (velocity frozen). Error therefore *accumulates* over the
    length of the gap — which is exactly the process-noise signal Q must learn.

    When ``observed`` is None the old behaviour (reset to truth every step) is
    kept for backward compatibility, but training/eval should always pass it so
    gap innovations grow with gap length.

    Returns: (B, T, 4)
    """
    b, t_len, _ = gt_trg.shape
    # "true" base = last two ground-truth positions. Used to score the honest
    # one-step process noise on observed frames (no reappearance spike).
    t_prev = gt_src[:, -1, :4]
    t_prev_prev = gt_src[:, -2, :4]
    # "drift" base = the tracker's own prediction. During a gap it keeps
    # extrapolating (velocity frozen), so its error accumulates with gap length.
    d_prev = t_prev.clone()
    d_prev_prev = t_prev_prev.clone()

    innovations = []
    for t in range(t_len):
        cur = gt_trg[:, t, :4]
        if observed is None:
            mu = constant_velocity_predict(t_prev, t_prev_prev)
            innovations.append(cur - mu)
            t_prev_prev, t_prev = t_prev, cur
            continue

        obs_t = (observed[:, t] > 0.5).unsqueeze(-1)
        mu_true = constant_velocity_predict(t_prev, t_prev_prev)
        mu_drift = constant_velocity_predict(d_prev, d_prev_prev)
        # Observed -> honest one-step CV error from the true state.
        # Gap      -> error of the (accumulating) drifted prediction.
        mu = torch.where(obs_t, mu_true, mu_drift)
        innovations.append(cur - mu)

        # Drift base: reset to truth on observed, propagate prediction on gaps.
        d_prev_prev = torch.where(obs_t, t_prev, d_prev)
        d_prev = torch.where(obs_t, cur, mu_drift)
        # True base always advances with ground truth.
        t_prev_prev, t_prev = t_prev, cur
    return torch.stack(innovations, dim=1)


def _log_var_match(pred_var: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return F.smooth_l1_loss(
        torch.log(pred_var.clamp(min=eps)),
        torch.log(target_var.clamp(min=eps)),
    )


def _gaussian_nll(
    innovations: torch.Tensor, var: torch.Tensor, log_2pi: float
) -> torch.Tensor:
    return 0.5 * (
        innovations.pow(2) / var + torch.log(var.clamp(min=1e-8)) + log_2pi
    )


def _gaussian_nll_logvar(
    innovations: torch.Tensor, log_var: torch.Tensor, log_2pi: float
) -> torch.Tensor:
    """
    Gaussian NLL using log-variance directly.

    Gradient w.r.t. log_var is 0.5*(1 - innov²*exp(-log_var)), which is always
    well-conditioned and never vanishes — unlike the softplus path, which produces
    a sigmoid factor that approaches zero when the logit becomes very negative.

    The quadratic term is capped so a single huge innovation at a tiny predicted
    variance cannot overflow float32 and poison the run.
    """
    # exp(20) ≈ 4.9e8; keep inv-var in a safe float32 range. The quadratic term
    # is capped so a huge innovation at a tiny predicted variance cannot overflow.
    log_var = log_var.clamp(-20.0, 8.0)
    quad = (innovations.pow(2) * torch.exp(-log_var)).clamp(max=1.0e6)
    return 0.5 * (quad + log_var + log_2pi)


def _grads_finite(module: nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _optimizer_step_finite(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    max_norm: float = 1.0,
) -> bool:
    """Backward + clipped step. Returns False if loss/grads were non-finite (step skipped). """
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return False
    loss.backward()
    if not _grads_finite(module):
        optimizer.zero_grad(set_to_none=True)
        return False
    nn.utils.clip_grad_norm_(module.parameters(), max_norm)
    optimizer.step()
    return True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1,L,D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



def _kf_matrices(device, dtype):
    Fm = torch.tensor(
        [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1]], device=device, dtype=dtype)
    Hm = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], device=device, dtype=dtype)
    return Fm, Hm


def _xywh_to_z(b, eps=1e-6):
    x, y = b[..., 0], b[..., 1]
    w = b[..., 2].clamp(min=eps); h = b[..., 3].clamp(min=eps)
    return torch.stack([x, y, w * h, w / h], dim=-1)


def _z_to_xywh(z, eps=1e-6):
    x, y = z[..., 0], z[..., 1]
    s = z[..., 2].clamp(min=eps); r = z[..., 3].clamp(min=eps)
    return torch.stack([x, y, torch.sqrt(s * r), torch.sqrt(s / r)], dim=-1)


def _var_xywh_to_zdiag(var_xywh, w, h, eps=1e-6):
    """Same normalized xywh->(x,y,s,r) variance mapping the tracker uses (iw=ih=1)."""
    w = w.clamp(min=eps); h = h.clamp(min=eps)
    vx, vy, vw, vh = var_xywh[:, 0], var_xywh[:, 1], var_xywh[:, 2], var_xywh[:, 3]
    vs = h ** 2 * vw + w ** 2 * vh
    vr = vw / h ** 2 + w ** 2 * vh / h ** 4
    return torch.stack([vx, vy, vs.clamp(min=1e-9), vr.clamp(min=1e-9)], dim=-1)


def differentiable_kf_track_loss(
    log_var_q, log_var_r, meas_xywh, gt_xywh, observed,
    var_floor=VAR_FLOOR, vel_floor=0.01, gap_weight=3.0,
):
    """
    Batched, differentiable SORT Kalman filter (same 7-D (x,y,s,r,vx,vy,vs) state, F/H
    and xywh->z variance mapping as the tracker) run forward over the window: feed the
    real detections as measurements (skip update on gap frames), and penalize the filter's
    OUTPUT box (posterior when observed, prior during gaps) against GT.

    This supervises Q and R *jointly through the Kalman gain* K = P Ht (H P Ht + R)^-1 — the
    network must emit (Q, R) whose gain fuses prediction + detection toward GT. Captures the
    Q/R co-adaptation the separate innovation-NLL / R-supervision terms cannot. Keep those
    terms on as anchors: this objective only constrains the Q/R *ratio*, not absolute scale.
    """
    B, T, _ = gt_xywh.shape
    if T < 3:
        return gt_xywh.new_tensor(0.0), gt_xywh.new_tensor(0.0)
    dtype = log_var_q.dtype
    Fm, Hm = _kf_matrices(gt_xywh.device, dtype)
    var_q = exp_var(log_var_q, var_floor)
    var_r = exp_var(log_var_r, var_floor)
    gt_z = _xywh_to_z(gt_xywh)

    x = gt_z.new_zeros(B, 7, 1)
    x[:, :4, 0] = gt_z[:, 1]
    x[:, 4:, 0] = (gt_z[:, 1] - gt_z[:, 0])[:, :3]
    P = torch.diag_embed(gt_z.new_tensor([10., 10., 10., 10., 1e4, 1e4, 1e4])
                         ).unsqueeze(0).expand(B, 7, 7).contiguous()
    I7 = torch.eye(7, device=x.device, dtype=dtype)
    I4 = torch.eye(4, device=x.device, dtype=dtype)

    losses, weights = [], []
    for t in range(2, T):
        w_t, h_t = gt_xywh[:, t, 2], gt_xywh[:, t, 3]
        q7 = torch.cat([_var_xywh_to_zdiag(var_q[:, t], w_t, h_t),
                        x.new_full((B, 3), vel_floor)], dim=-1)
        Q = torch.diag_embed(q7)
        R = torch.diag_embed(_var_xywh_to_zdiag(var_r[:, t], w_t, h_t))

        x = Fm @ x
        P = Fm @ P @ Fm.transpose(-1, -2) + Q

        obs_t = observed[:, t].view(B, 1, 1)
        z = _xywh_to_z(meas_xywh[:, t]).unsqueeze(-1)
        y = z - Hm @ x
        S = Hm @ P @ Hm.transpose(-1, -2) + R + 1e-6 * I4
        K = P @ Hm.transpose(-1, -2) @ torch.linalg.inv(S)
        x = torch.where(obs_t, x + K @ y, x)
        P = torch.where(obs_t, (I7 - K @ Hm) @ P, P)

        out = _z_to_xywh(x[:, :4, 0])
        l = F.smooth_l1_loss(out, gt_xywh[:, t], beta=0.05, reduction="none").mean(-1)
        wgt = torch.where(observed[:, t], out.new_tensor(1.0), out.new_tensor(gap_weight))
        losses.append(l * wgt); weights.append(wgt)

    total = torch.stack(losses).sum()
    denom = torch.stack(weights).sum().clamp(min=1.0)
    return total / denom, total.detach() / denom

    
def nfc(n_layers, input_dim, output_dim, dropout):
    components = []
    dims = torch.linspace(input_dim, output_dim, n_layers + 1)
    dims = [int(x) for x in dims]
    for i in range(len(dims) - 2):
        components.extend([
            nn.Linear(dims[i], dims[i+1]),
            nn.LayerNorm(dims[i+1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        ])
    components.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*components)


class AdaptiveKalmanLoss(nn.Module):
    """
    Context-split Kalman noise loss (thesis-aligned):

    - **Q / innovation NLL (primary Q driver):** gap-weighted CV error explained
      by process noise Q. The NLL naturally calibrates var_q to the *conditional
      mean* of innovation², so it is heteroscedastic (larger in gaps, grows with
      gap length) and beats a fixed variance. Observed steps are down-weighted so
      easy frames do not dominate Q.
    - **R supervision:** on observed noisy steps, match log_var_r to log(det_error²).
    - **Q gap supervision (OFF by default, ``q_gap_coeff=0``):** a per-element
      ``smooth_l1`` in log-space against ``log(innov²)``. On heavily-interpolated
      GT (e.g. MOT17) most innovations are ~0, so this log match collapses var_q
      to the floor (it targets the log-median, not the mean) and *fights* the NLL.
      Kept as an option for very dynamic data, but disabled by default.
    - **Q gap trend:** among gap frames only, encourage larger Q for longer gaps.
    - **Q easy penalty:** lightly penalize large Q on high-confidence observed steps.
    """

    _LOG_2PI = math.log(2 * math.pi)

    def __init__(
        self,
        innovation_coeff: float = 1.0,
        r_supervise_coeff: float = 2.0,
        q_gap_coeff: float = 0.0,
        q_easy_coeff: float = 0.01,
        q_gap_trend_coeff: float = 1.0,
        innov_obs_weight: float = 0.25,
        conf_alpha: float = 2.0,
        var_floor: float = VAR_FLOOR,
        kf_track_coeff: float = 0.0,
        kf_gap_weight: float = 3.0,
    ):
        super().__init__()
        self.innovation_coeff = innovation_coeff
        self.r_supervise_coeff = r_supervise_coeff
        self.q_gap_coeff = q_gap_coeff
        self.q_easy_coeff = q_easy_coeff
        self.q_gap_trend_coeff = q_gap_trend_coeff
        self.innov_obs_weight = innov_obs_weight
        self.conf_alpha = conf_alpha
        self.var_floor = var_floor
        # Joint Q/R supervision via a differentiable Kalman filter (0 = off).
        # Keep innovation_coeff / r_supervise_coeff > 0 as scale anchors when using this.
        self.kf_track_coeff = kf_track_coeff
        self.kf_gap_weight = kf_gap_weight

    def forward(
        self,
        log_var_q: torch.Tensor,
        log_var_r: torch.Tensor,
        innovations: torch.Tensor,
        trg: torch.Tensor,
        gt_trg: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # log_var_q / log_var_r are true log-variances. Use exp — never softplus —
        # so gradients stay well-conditioned (no softplus saturation).
        # Keep within the same range as the stable NLL path (avoid exp overflow).
        log_var_q_s = log_var_q.clamp(min=-20.0, max=8.0)
        log_var_r_s = log_var_r.clamp(min=-20.0, max=8.0)
        var_q = exp_var(log_var_q_s, floor=self.var_floor)
        var_r = exp_var(log_var_r_s, floor=self.var_floor)

        innov_sq = innovations.pow(2).clamp(min=self.var_floor, max=1.0e4)

        scores = trg[..., 12:13].clamp(0.0, 1.0)
        gap_len = trg[..., 13:14].clamp(0.0, 1.0)
        observed = trg[..., 14:15] > 0.5
        gap = ~observed

        # Gap-weighted innovation NLL: gaps drive Q calibration; observed steps
        # keep a light anchor so Q does not explode on easy CV residuals.
        nll = _gaussian_nll_logvar(innovations.clamp(-10.0, 10.0), log_var_q_s, self._LOG_2PI)
        gap_m = gap.expand_as(nll)
        obs_m = observed.expand_as(nll)
        parts = []
        if gap_m.any():
            parts.append(nll[gap_m].mean())
        if obs_m.any() and self.innov_obs_weight > 0:
            parts.append(self.innov_obs_weight * nll[obs_m].mean())
        loss_innov = (
            torch.stack(parts).sum()
            if parts
            else innovations.new_tensor(0.0)
        )

        # R supervision: observed frames with any measurable detector error.
        # Match log_var_r directly (short gradient path, same spirit as Q-gap).
        meas_sq = (trg[..., :4] - gt_trg[..., :4]).pow(2).clamp(min=self.var_floor, max=1.0e4)
        per_step_meas = meas_sq.max(dim=-1, keepdim=True).values
        has_det_noise = per_step_meas > (self.var_floor * 10)
        r_mask = observed & has_det_noise
        if r_mask.any():
            log_r_pred = log_var_r_s[r_mask.expand_as(log_var_r_s)]
            log_r_tgt = torch.log(meas_sq[r_mask.expand_as(meas_sq)].clamp(min=1e-8))
            loss_r = F.smooth_l1_loss(log_r_pred, log_r_tgt)
        else:
            loss_r = innovations.new_tensor(0.0)

        # Q gap supervision: match log_var_q directly to log(innov²) on dropped frames.
        if gap.any():
            log_var_q_gap = log_var_q_s[gap.expand_as(log_var_q_s)]
            gap_target = innov_sq[gap.expand_as(innov_sq)]
            log_gap_target = torch.log(gap_target.clamp(min=1e-8))
            loss_q_gap = F.smooth_l1_loss(log_var_q_gap, log_gap_target)
        else:
            loss_q_gap = innovations.new_tensor(0.0)

        # Encourage Q to grow with gap length — among gap frames only.
        # Normalize only gap_len; do NOT divide by log_q.std() (that exploded to
        # NaN once Q became nearly constant on gaps around epoch ~10).
        gap_step = gap.squeeze(-1)
        log_q_step = log_var_q_s.mean(dim=-1)
        gl = gap_len.squeeze(-1)
        if int(gap_step.sum()) >= 4:
            gl_g = gl[gap_step]
            lq_g = log_q_step[gap_step]
            gl_std = gl_g.std()
            if float(gl_std) > 1e-4:
                gl_n = (gl_g - gl_g.mean()) / (gl_std + 1e-8)
                lq_c = (lq_g - lq_g.mean()).clamp(-8.0, 8.0)
                # assoc ≈ corr * std(log_q); target a mild positive association.
                assoc = (gl_n * lq_c).mean()
                loss_q_gap_trend = F.relu(0.15 - assoc)
            else:
                loss_q_gap_trend = innovations.new_tensor(0.0)
        else:
            loss_q_gap_trend = innovations.new_tensor(0.0)

        # Keep Q small when we have a confident observation
        easy = scores * observed.float()
        loss_q_easy = (var_q * easy).mean()

        # Joint Q/R supervision through a differentiable Kalman gain (optional).
        if self.kf_track_coeff > 0:
            loss_kf_track, kf_track_val = differentiable_kf_track_loss(
                log_var_q, log_var_r,
                trg[..., :4], gt_trg[..., :4], observed.squeeze(-1),
                var_floor=self.var_floor, gap_weight=self.kf_gap_weight,
            )
        else:
            loss_kf_track = innovations.new_tensor(0.0)
            kf_track_val = innovations.new_tensor(0.0)

        loss = (
            self.innovation_coeff * loss_innov
            + self.r_supervise_coeff * loss_r
            + self.q_gap_coeff * loss_q_gap
            + self.q_gap_trend_coeff * loss_q_gap_trend
            + self.q_easy_coeff * loss_q_easy
            + self.kf_track_coeff * loss_kf_track
        )

        with torch.no_grad():
            calib_q = (
                (innov_sq[gap.expand_as(innov_sq)].mean() / var_q[gap.expand_as(var_q)].mean())
                if gap.any()
                else innovations.new_tensor(float("nan"))
            )
            metrics = {
                "loss_innov": float(loss_innov),
                "loss_r": float(loss_r),
                "loss_q_gap": float(loss_q_gap),
                "loss_q_gap_trend": float(loss_q_gap_trend),
                "loss_q_easy": float(loss_q_easy),
                "loss_kf_track": float(kf_track_val),
                "mean_var_q": float(var_q.mean()),
                "mean_var_r": float(var_r.mean()),
                "mean_innov_sq": float(innov_sq.mean()),
                "calib_q_gap": float(calib_q),
                "frac_gap": float(gap.float().mean()),
                "frac_r_supervised": float(r_mask.float().mean()),
            }
            if gap.any():
                metrics["mean_var_q_gap"] = float(var_q[gap.expand_as(var_q)].mean())
            if observed.any():
                metrics["mean_var_r_obs"] = float(var_r[observed.expand_as(var_r)].mean())
                metrics["mean_var_q_observed"] = float(var_q[observed.expand_as(var_q)].mean())

        return loss, metrics


class _AdaptiveKalmanHead(nn.Module):
    """Shared Q/R output heads with confidence-R prior on R."""

    def __init__(self, hidden_dim: int, dropout: float, conf_alpha: float = 2.0, q_init_bias: float = -12.0):
        super().__init__()
        self.conf_alpha = conf_alpha
        # Q head output is interpreted directly as log(var_q) by the loss.
        # Bias = -12 → exp(-12) ≈ 6e-6 at init: a mid-scale start that sits
        # between low-motion (MOT17 ~6e-7) and high-motion (DanceTrack ~6e-5)
        # process noise, so the head does not have to travel far in either
        # direction and low-motion data is not stuck high early in training.
        
        self.q_head = nfc(
            n_layers=3,
            input_dim=hidden_dim,
            output_dim=4,
            dropout=dropout * 0.5
        )
        self.r_residual_head = nfc(
            n_layers=3,
            input_dim=hidden_dim,
            output_dim=4,
            dropout=dropout * 0.5
        )
        # self.q_head = nn.Linear(hidden_dim, 4)
        # nn.init.constant_(self.q_head.bias, q_init_bias)
        # nn.init.xavier_uniform_(self.q_head.weight, gain=0.1)
        # Additive log-space delta on the confidence prior. Zero init → start at
        # prior; signed delta lets R go above or below the prior (unlike the old
        # prior_var + exp(logit) floor which froze loss_r when prior was too high).
        # self.r_residual_head = nn.Linear(hidden_dim, 4)
        # nn.init.zeros_(self.r_residual_head.weight)
        # nn.init.zeros_(self.r_residual_head.bias)

    def forward(
        self, hidden: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_var_q = self.q_head(hidden)
        log_r_prior = confidence_log_r_prior(scores, alpha=self.conf_alpha)
        delta = self.r_residual_head(hidden).clamp(-6.0, 6.0)
        log_var_r = (log_r_prior + delta).clamp(-16.0, 8.0)
        return log_var_q, log_var_r



class AdaptiveKalmanTransformer(nn.Module):
    """
    Transformer encoder over [history | future-context] → per-step log_var_q, log_var_r.
    No bbox output.
    """

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        conf_alpha: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.conf_alpha = conf_alpha

        self.in_fc = nfc(
            n_layers=3,
            input_dim=input_dim,
            output_dim=d_model,
            dropout=dropout * 0.5
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                batch_first=True,
                dropout=dropout,
                activation="gelu",
            ),
            mask_check=False,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = _AdaptiveKalmanHead(d_model, dropout, conf_alpha=conf_alpha)

    @staticmethod
    def _causal_mask(src_len: int, ctx_len: int, device: torch.device) -> torch.Tensor:
        total = src_len + ctx_len
        mask = torch.triu(torch.ones(total, total, device=device), diagonal=1)
        mask[:, :src_len] = 0
        return mask.bool()

    def _encode(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, src_len, _ = src.shape
        x = torch.cat([src, ctx], dim=1)
        emb = self.pos_enc(self.in_fc(x) * math.sqrt(self.d_model))
        mask = self._causal_mask(src_len, ctx.size(1), x.device)
        pad = None
        if src_key_padding_mask is not None:
            pad = torch.zeros(
                b, src_len + ctx.size(1), dtype=torch.bool, device=x.device
            )
            pad[:, :src_len] = src_key_padding_mask
        if pad is None:
            out = self.transformer(emb, mask=mask)
        else:
            out = self.transformer(emb, mask=mask, src_key_padding_mask=pad)
        return out[:, -ctx.size(1) :, :]

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: (B, S, F) history, ctx: (B, T, F) future context (noisy features, no GT leak).
        Returns log_var_q, log_var_r: (B, T, 4) each.
        """
        hidden = self._encode(src, ctx, src_key_padding_mask)
        scores = ctx[..., 12:13]
        return self.head(hidden, scores)

    @torch.no_grad()
    def predict_noise(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step Q/R from observation history (tracker predict/update)."""
        b = src.size(0)
        ctx = src[:, -1:, :].clone()
        hidden = self._encode(src, ctx, src_key_padding_mask)
        scores = ctx[..., 12:13]
        log_q, log_r = self.head(hidden[:, -1:, :], scores)
        return log_q[:, 0, :], log_r[:, 0, :]

    def train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.train()
        total = 0.0
        agg: dict = {}
        n_ok = 0
        n_skip = 0
        for src, trg, gt_src, gt_trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)

            optimizer.zero_grad(set_to_none=True)
            log_q, log_r = self.forward(src, trg)
            innovations = build_cv_innovations(
                gt_src, gt_trg, observed=trg[..., 14]
            )
            loss, metrics = criterion(log_q, log_r, innovations, trg, gt_trg)
            if not _optimizer_step_finite(self, optimizer, loss):
                n_skip += 1
                continue
            total += float(loss.detach())
            n_ok += 1
            for k, v in metrics.items():
                if isinstance(v, float) and math.isfinite(v):
                    agg[k] = agg.get(k, 0.0) + v
        if n_skip:
            print(f"  skipped {n_skip} non-finite train batches")
        n = max(n_ok, 1)
        out_metrics = {k: v / n for k, v in agg.items()}
        out_metrics["n_batches_ok"] = float(n_ok)
        out_metrics["n_batches_skip"] = float(n_skip)
        return (total / n) if n_ok else float("nan"), out_metrics

    def evaluate(
        self,
        dataloader,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.eval()
        total = 0.0
        agg: dict = {}
        n = max(len(dataloader), 1)
        with torch.no_grad():
            for src, trg, gt_src, gt_trg in dataloader:
                src = src.to(device)
                trg = trg.to(device)
                gt_src = gt_src.to(device)
                gt_trg = gt_trg.to(device)
                log_q, log_r = self.forward(src, trg)
                innovations = build_cv_innovations(
                    gt_src, gt_trg, observed=trg[..., 14]
                )
                loss, metrics = criterion(log_q, log_r, innovations, trg, gt_trg)
                total += loss.item()
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + v
        return total / n, {k: v / n for k, v in agg.items()}

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load(path, map_location=loc, weights_only=True)
        )


class AdaptiveKalmanLSTM(nn.Module):
    """LSTM encoder over history + autoregressive context steps → Q/R only."""

    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        conf_alpha: float = 2.0,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
        )
        self.lstm = nn.LSTM(
            d_model,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _AdaptiveKalmanHead(hidden_dim, conf_alpha=conf_alpha)

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)

        prev = ctx[:, 0:1, :]
        log_qs, log_rs = [], []
        for t in range(ctx.size(1)):
            inp = self.in_fc(prev)
            out, (h, c) = self.lstm(inp, (h, c))
            score = prev[:, :, 12:13]
            lq, lr = self.head(out, score)
            log_qs.append(lq)
            log_rs.append(lr)
            if t + 1 < ctx.size(1):
                use_teacher = random.random() < teacher_forcing_ratio
                prev = ctx[:, t + 1 : t + 2, :] if use_teacher else prev

        return torch.cat(log_qs, dim=1), torch.cat(log_rs, dim=1)

    @torch.no_grad()
    def predict_noise(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)
        prev = src[:, -1:, :]
        out, (h, c) = self.lstm(self.in_fc(prev), (h, c))
        score = prev[:, :, 12:13]
        log_q, log_r = self.head(out, score)
        return log_q[:, 0, :], log_r[:, 0, :]

    def train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.train()
        total = 0.0
        agg: dict = {}
        n_ok = 0
        n_skip = 0
        for src, trg, gt_src, gt_trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_q, log_r = self.forward(src, trg)
            innovations = build_cv_innovations(
                gt_src, gt_trg, observed=trg[..., 14]
            )
            loss, metrics = criterion(log_q, log_r, innovations, trg, gt_trg)
            if not _optimizer_step_finite(self, optimizer, loss):
                n_skip += 1
                continue
            total += float(loss.detach())
            n_ok += 1
            for k, v in metrics.items():
                if isinstance(v, float) and math.isfinite(v):
                    agg[k] = agg.get(k, 0.0) + v
        if n_skip:
            print(f"  skipped {n_skip} non-finite train batches")
        n = max(n_ok, 1)
        out_metrics = {k: v / n for k, v in agg.items()}
        out_metrics["n_batches_ok"] = float(n_ok)
        out_metrics["n_batches_skip"] = float(n_skip)
        return (total / n) if n_ok else float("nan"), out_metrics

    def evaluate(
        self,
        dataloader,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.eval()
        total = 0.0
        agg: dict = {}
        n = max(len(dataloader), 1)
        with torch.no_grad():
            for src, trg, gt_src, gt_trg in dataloader:
                src = src.to(device)
                trg = trg.to(device)
                gt_src = gt_src.to(device)
                gt_trg = gt_trg.to(device)
                log_q, log_r = self.forward(
                    src, trg, teacher_forcing_ratio=1.0
                )
                innovations = build_cv_innovations(
                    gt_src, gt_trg, observed=trg[..., 14]
                )
                loss, metrics = criterion(log_q, log_r, innovations, trg, gt_trg)
                total += loss.item()
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + v
        return total / n, {k: v / n for k, v in agg.items()}

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load(path, map_location=loc, weights_only=True)
        )


def build_adaptive_kalman_model(
    model_type: str = "transformer",
    input_dim: int = 15,
    **kwargs,
) -> nn.Module:
    if model_type == "transformer":
        return AdaptiveKalmanTransformer(input_dim=input_dim, **kwargs)
    if model_type == "lstm":
        return AdaptiveKalmanLSTM(input_dim=input_dim, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")
