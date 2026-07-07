"""
Context-aware adaptive Kalman noise predictor (Q/R only).

Thesis-aligned design (see analysis/THESIS_DIAGNOSTICS_AND_TRACKER_GUIDANCE.md):
- Predict diagonal log-variance for process noise Q and measurement noise R.
- No bbox regression head (kalman_fusion_blend = 0 at tracker integration time).
- R uses a confidence prior R ∝ exp(α(1 - score)) plus a learned residual.
- Training loss: innovation NLL under a constant-velocity Kalman proxy + R supervision.

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

from transformer_encoder import PositionalEncoding


def softplus_var(log_v: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return F.softplus(log_v) + floor


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
) -> torch.Tensor:
    """
    Per-step innovations y - μ_CV for each target timestep.

    gt_src: (B, S, D), gt_trg: (B, T, D)
    Returns: (B, T, 4)
    """
    b, t_len, _ = gt_trg.shape
    prev = gt_src[:, -1, :4]
    prev_prev = gt_src[:, -2, :4]
    innovations = []
    for t in range(t_len):
        mu = constant_velocity_predict(prev, prev_prev)
        innovations.append(gt_trg[:, t, :4] - mu)
        prev_prev = prev
        prev = gt_trg[:, t, :4]
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


class AdaptiveKalmanLoss(nn.Module):
    """
    Context-split Kalman noise loss (thesis-aligned):

    - **Q / innovation NLL:** CV prediction error is explained by process noise Q
      (not Q+R — that let Q collapse to the softplus floor).
    - **R supervision:** on observed steps only, match var_r to per-step detector
      error (noisy box vs GT), in log-variance space after softplus.
    - **Q gap supervision:** on dropped/missing steps, match var_q to |innovation|².
    - **Q easy penalty:** lightly penalize large Q on high-confidence observed steps.
    """

    _LOG_2PI = math.log(2 * math.pi)

    def __init__(
        self,
        innovation_coeff: float = 1.0,
        r_supervise_coeff: float = 0.5,
        q_gap_coeff: float = 0.3,
        q_easy_coeff: float = 0.01,
        conf_alpha: float = 2.0,
        var_floor: float = 1e-6,
        norm_var_floor: float = 1e-5,
    ):
        super().__init__()
        self.innovation_coeff = innovation_coeff
        self.r_supervise_coeff = r_supervise_coeff
        self.q_gap_coeff = q_gap_coeff
        self.q_easy_coeff = q_easy_coeff
        self.conf_alpha = conf_alpha
        self.var_floor = var_floor
        self.norm_var_floor = norm_var_floor

    def forward(
        self,
        log_var_q: torch.Tensor,
        log_var_r: torch.Tensor,
        innovations: torch.Tensor,
        trg: torch.Tensor,
        gt_trg: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        var_q = softplus_var(log_var_q, floor=self.var_floor)
        var_r = softplus_var(log_var_r, floor=self.var_floor)

        innov_sq = innovations.pow(2).clamp(min=self.var_floor)
        # Additive floor keeps gradients on var_q (max() was freezing Q at ~1e-4)
        var_q_nll = var_q + self.norm_var_floor

        # Q explains CV innovations (predict / process step)
        loss_innov = _gaussian_nll(innovations, var_q_nll, self._LOG_2PI).mean()
        scores = trg[..., 12:13].clamp(0.0, 1.0)
        observed = trg[..., 14:15] > 0.5
        gap = ~observed

        # R: only where detector noise exists (skip clean frames — meas_sq≈0)
        meas_sq = (trg[..., :4] - gt_trg[..., :4]).pow(2).clamp(min=self.var_floor)
        per_step_meas = meas_sq.max(dim=-1, keepdim=True).values
        has_det_noise = per_step_meas > (self.var_floor * 100)
        r_mask = observed & has_det_noise
        if r_mask.any():
            loss_r = _log_var_match(
                var_r[r_mask.expand_as(var_r)], meas_sq[r_mask.expand_as(meas_sq)]
            )
        else:
            loss_r = innovations.new_tensor(0.0)

        # Q: on gap frames, match to actual innovation² (no artificial 1e-4 floor)
        if gap.any():
            gap_var_q = var_q[gap.expand_as(var_q)]
            gap_target = innov_sq[gap.expand_as(innov_sq)]
            loss_q_gap = _log_var_match(gap_var_q, gap_target)
        else:
            loss_q_gap = innovations.new_tensor(0.0)

        # Keep Q small when we have a confident observation
        easy = scores * observed.float()
        loss_q_easy = (var_q * easy).mean()

        loss = (
            self.innovation_coeff * loss_innov
            + self.r_supervise_coeff * loss_r
            + self.q_gap_coeff * loss_q_gap
            + self.q_easy_coeff * loss_q_easy
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
                "loss_q_easy": float(loss_q_easy),
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

        return loss, metrics


class _AdaptiveKalmanHead(nn.Module):
    """Shared Q/R output heads with confidence-R prior on R."""

    def __init__(self, hidden_dim: int, conf_alpha: float = 2.0, q_init_bias: float = -9.0):
        super().__init__()
        self.conf_alpha = conf_alpha
        self.q_head = nn.Linear(hidden_dim, 4)
        self.r_residual_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.r_residual_head.weight)
        nn.init.zeros_(self.r_residual_head.bias)
        # Avoid Q collapsing to softplus floor at init (softplus(-5) ~ 0.007)
        nn.init.constant_(self.q_head.bias, q_init_bias)
        nn.init.xavier_uniform_(self.q_head.weight, gain=0.1)

    def forward(
        self, hidden: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_var_q = self.q_head(hidden)
        log_r_prior = confidence_log_r_prior(scores, alpha=self.conf_alpha)
        prior_var = softplus_var(log_r_prior)
        # Residual is additive in variance space (keeps R >= conf-R prior)
        delta_r = F.softplus(self.r_residual_head(hidden))
        var_r = prior_var + delta_r
        log_var_r = torch.log(var_r.clamp(min=1e-8))
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
        self.head = _AdaptiveKalmanHead(d_model, conf_alpha=conf_alpha)

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
        n = max(len(dataloader), 1)
        for src, trg, gt_src, gt_trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)

            optimizer.zero_grad()
            ctx = trg[:, :-1, :]
            log_q, log_r = self.forward(src, ctx)
            innovations = build_cv_innovations(gt_src, gt_trg[:, 1:, :])
            loss, metrics = criterion(
                log_q, log_r, innovations, trg[:, 1:, :], gt_trg[:, 1:, :]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
        return total / n, {k: v / n for k, v in agg.items()}

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
                log_q, log_r = self.forward(src, trg[:, :-1, :])
                innovations = build_cv_innovations(gt_src, gt_trg[:, 1:, :])
                loss, metrics = criterion(
                    log_q, log_r, innovations, trg[:, 1:, :], gt_trg[:, 1:, :]
                )
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
        n = max(len(dataloader), 1)
        for src, trg, gt_src, gt_trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)
            optimizer.zero_grad()
            log_q, log_r = self.forward(src, trg[:, :-1, :])
            innovations = build_cv_innovations(gt_src, gt_trg[:, 1:, :])
            loss, metrics = criterion(
                log_q, log_r, innovations, trg[:, 1:, :], gt_trg[:, 1:, :]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
        return total / n, {k: v / n for k, v in agg.items()}

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
                    src, trg[:, :-1, :], teacher_forcing_ratio=1.0
                )
                innovations = build_cv_innovations(gt_src, gt_trg[:, 1:, :])
                loss, metrics = criterion(
                    log_q, log_r, innovations, trg[:, 1:, :], gt_trg[:, 1:, :]
                )
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
