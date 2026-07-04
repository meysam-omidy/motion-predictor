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
    score: torch.Tensor, alpha: float = 2.0, floor: float = 1e-6
) -> torch.Tensor:
    """
    log R prior matching OC-SORT conf-R: R ∝ exp(alpha * (1 - score)).
    score: (..., 1) in [0, 1].
    Returns log prior with same trailing shape, broadcastable to 4 bbox dims.
    """
    s = score.clamp(0.0, 1.0)
    return torch.log(torch.exp(alpha * (1.0 - s)) + floor)


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


class AdaptiveKalmanLoss(nn.Module):
    """
    Innovation NLL (heteroscedastic) + R supervision + optional Q regularization.

    Total predicted variance for the innovation combines process and measurement
    uncertainty: var = var_q + var_r (diagonal, per bbox dimension).
    """

    _LOG_2PI = math.log(2 * math.pi)

    def __init__(
        self,
        innovation_coeff: float = 1.0,
        r_supervise_coeff: float = 0.5,
        q_smooth_coeff: float = 0.05,
        conf_alpha: float = 2.0,
    ):
        super().__init__()
        self.innovation_coeff = innovation_coeff
        self.r_supervise_coeff = r_supervise_coeff
        self.q_smooth_coeff = q_smooth_coeff
        self.conf_alpha = conf_alpha

    def forward(
        self,
        log_var_q: torch.Tensor,
        log_var_r: torch.Tensor,
        innovations: torch.Tensor,
        src: torch.Tensor,
        gt_src: torch.Tensor,
        trg_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        var_q = softplus_var(log_var_q)
        var_r = softplus_var(log_var_r)
        var_total = var_q + var_r

        nll = 0.5 * (
            innovations.pow(2) / var_total
            + torch.log(var_total)
            + self._LOG_2PI
        )
        loss_innov = nll.mean()

        meas_sq = (src[..., :4] - gt_src[..., :4]).pow(2).mean(
            dim=(1, 2), keepdim=True
        )
        target_log_r = torch.log(meas_sq + 1e-6).expand_as(log_var_r)
        loss_r = F.smooth_l1_loss(log_var_r, target_log_r)

        # Mild penalty when Q explodes on easy (high-confidence) steps
        scores = trg_scores.clamp(0, 1)
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1)
        easy = scores[..., :1]
        loss_q = (var_q * easy).mean()

        loss = (
            self.innovation_coeff * loss_innov
            + self.r_supervise_coeff * loss_r
            + self.q_smooth_coeff * loss_q
        )
        metrics = {
            "loss_innov": float(loss_innov.detach()),
            "loss_r": float(loss_r.detach()),
            "loss_q": float(loss_q.detach()),
        }
        return loss, metrics


class _AdaptiveKalmanHead(nn.Module):
    """Shared Q/R output heads with confidence-R prior on R."""

    def __init__(self, hidden_dim: int, conf_alpha: float = 2.0):
        super().__init__()
        self.conf_alpha = conf_alpha
        self.q_head = nn.Linear(hidden_dim, 4)
        self.r_residual_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.r_residual_head.weight)
        nn.init.zeros_(self.r_residual_head.bias)

    def forward(
        self, hidden: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_var_q = self.q_head(hidden)
        log_r_prior = confidence_log_r_prior(scores, alpha=self.conf_alpha)
        log_var_r = log_r_prior + self.r_residual_head(hidden)
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
        agg = {"loss_innov": 0.0, "loss_r": 0.0, "loss_q": 0.0}
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
                log_q, log_r, innovations, src, gt_src, trg[:, 1:, 12:13]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            for k in agg:
                agg[k] += metrics[k]
        return total / n, {k: v / n for k, v in agg.items()}

    def evaluate(
        self,
        dataloader,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.eval()
        total = 0.0
        agg = {"loss_innov": 0.0, "loss_r": 0.0, "loss_q": 0.0}
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
                    log_q, log_r, innovations, src, gt_src, trg[:, 1:, 12:13]
                )
                total += loss.item()
                for k in agg:
                    agg[k] += metrics[k]
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
        agg = {"loss_innov": 0.0, "loss_r": 0.0, "loss_q": 0.0}
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
                log_q, log_r, innovations, src, gt_src, trg[:, 1:, 12:13]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            for k in agg:
                agg[k] += metrics[k]
        return total / n, {k: v / n for k, v in agg.items()}

    def evaluate(
        self,
        dataloader,
        criterion: AdaptiveKalmanLoss,
        device: str = "cuda",
    ) -> Tuple[float, dict]:
        self.eval()
        total = 0.0
        agg = {"loss_innov": 0.0, "loss_r": 0.0, "loss_q": 0.0}
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
                    log_q, log_r, innovations, src, gt_src, trg[:, 1:, 12:13]
                )
                total += loss.item()
                for k in agg:
                    agg[k] += metrics[k]
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
