"""
Learned diagonal covariances as Kalman-style Q (process) and R (measurement) proxies.

- Q-like: per-step predicted variance on bbox residuals (motion / model uncertainty).
- R-like: predicted variance aligned (optionally) with squared error between noisy
  observations and clean GT on the input window (detector / measurement noise).

Inference for a tracker: use predicted conf and 1 / (1 + tr(var)) style gates, or
sample-free updates by treating predicted boxes as a learned filter output.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import LossFunction
from lstm_improved import ImprovedLSTMPredictor
from transformer_encoder import MotionTransformer


def softplus_var(log_v: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    return F.softplus(log_v) + floor


class LearnedNoiseMotionLoss(nn.Module):
    """
    Heteroscedastic NLL on bbox (Q proxy) + CIoU + confidence + optional R supervision.

    The Gaussian term is the **negative log-likelihood** of independent 1-D normals per
    bbox dimension: 0.5 * (log(2πσ²) + (y-μ)²/σ²). For continuous y this quantity **can
    still be negative** when the density at y is > 1 (sharp peak); that is valid and
    does not break training. The sum with CIoU / conf can also go negative. Minimizing
    the same expression is well-defined either way.
    """

    # Per-dimension: 0.5 * log(2π); gradient w.r.t. model params is zero (display / convention only).
    _LOG_2PI = math.log(2 * math.pi)

    def __init__(
        self,
        nll_coeff: float = 1.0,
        ciou_coeff: float = 0.5,
        conf_coeff: float = 0.25,
        r_supervise_coeff: float = 0.1,
    ):
        super().__init__()
        self.nll_coeff = nll_coeff
        self.ciou_coeff = ciou_coeff
        self.conf_coeff = conf_coeff
        self.r_supervise_coeff = r_supervise_coeff
        self._ciou = LossFunction()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var_q: torch.Tensor,
        log_var_r: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        gt_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_box = pred[..., :4]
        tgt_box = target[..., :4]
        var_q = softplus_var(log_var_q)
        # Full 1-D Gaussian NLL per dim: 0.5 * (log(2π) + log σ² + (y-μ)²/σ²)
        nll = 0.5 * (
            (tgt_box - pred_box) ** 2 / var_q
            + torch.log(var_q)
            + self._LOG_2PI
        )
        nll = nll.mean()

        ciou = self._ciou.ciou(tgt_box, pred_box)
        loss_ciou = (1 - ciou).mean()

        pred_c = pred[..., 4:5]
        loss_conf = F.smooth_l1_loss(pred_c, ciou.detach().unsqueeze(-1))

        loss = (
            self.nll_coeff * nll
            + self.ciou_coeff * loss_ciou
            + self.conf_coeff * loss_conf
        )

        if (
            self.r_supervise_coeff > 0
            and src is not None
            and gt_src is not None
        ):
            # Squared error between noisy input and clean GT (xywh), mean over time & dims -> log target
            meas_sq = (src[..., :4] - gt_src[..., :4]).pow(2).mean(dim=(1, 2), keepdim=True)
            target_log = torch.log(meas_sq + 1e-6).expand_as(log_var_r)
            loss_r = F.smooth_l1_loss(log_var_r, target_log)
            loss = loss + self.r_supervise_coeff * loss_r

        return loss


class MotionTransformerLearnedNoise(MotionTransformer):
    """
    Same masked encoder as MotionTransformer; head outputs
    [delta_xywh, conf_raw, log_var_q (4), log_var_r (4)].
    """

    def __init__(self, input_dim: int = 13, d_model: int = 256, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_dim"] = 13
        kwargs.setdefault("input_dim", input_dim)
        super().__init__(**kwargs)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, src_len, _ = src.shape
        input_tensor = torch.cat([src, trg], dim=1)
        enc_emb = self.pos_enc(self.in_fc(input_tensor) * math.sqrt(self.d_model))
        mask = self._mask(src.size(1), trg.size(1), input_tensor.device)
        pad = None
        if src_key_padding_mask is not None:
            pad = torch.zeros(
                b, src_len + trg.size(1), dtype=torch.bool, device=input_tensor.device
            )
            pad[:, :src_len] = src_key_padding_mask
        if pad is None:
            out = self.transformer.forward(enc_emb, mask=mask)
        else:
            out = self.transformer.forward(
                enc_emb, mask=mask, src_key_padding_mask=pad
            )
        raw = self.out_fc(out[:, -trg.size(1) :, :])
        dbox = raw[..., :4]
        conf = torch.sigmoid(raw[..., 4:5])
        log_vq = raw[..., 5:9]
        log_vr = raw[..., 9:13]
        pred_box = trg[..., :4] + dbox
        pred = torch.cat([pred_box, conf], dim=-1)
        return pred, log_vq, log_vr

    @torch.no_grad()
    def inference(
        self, src: torch.Tensor, trg: torch.Tensor, num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive rollout: grow 13-D target inputs from last predicted 5-D state."""
        preds, vqs, vrs = [], [], []
        cur_trg = trg[:, :1, :].clone()
        for _ in range(num_steps):
            out, lq, lr = self.forward(src, cur_trg)
            p = out[:, -1:, :]
            preds.append(p)
            vqs.append(lq[:, -1:, :])
            vrs.append(lr[:, -1:, :])
            if _ + 1 >= num_steps:
                break
            nxt = cur_trg[:, -1:, :].clone()
            nxt[:, :, :4] = p[:, :, :4]
            nxt[:, :, 4:12] = 0.0
            nxt[:, :, 12:13] = p[:, :, 4:5]
            cur_trg = torch.cat([cur_trg, nxt], dim=1)
        return (
            torch.cat(preds, dim=1),
            torch.cat(vqs, dim=1),
            torch.cat(vrs, dim=1),
        )

    def train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion: LearnedNoiseMotionLoss,
        device: str = "cuda",
    ) -> float:
        self.train()
        total = 0.0
        for src, trg, gt_src, gt_trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)
            optimizer.zero_grad()
            pred, log_vq, log_vr = self.forward(src, trg[:, :-1])
            loss = criterion(pred, gt_trg[:, 1:], log_vq, log_vr, src, gt_src)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)

    def evaluate(
        self,
        dataloader,
        criterion: LearnedNoiseMotionLoss,
        device: str = "cuda",
    ) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for src, trg, gt_src, gt_trg in dataloader:
                src = src.to(device)
                trg = trg.to(device)
                gt_src = gt_src.to(device)
                gt_trg = gt_trg.to(device)
                pred, log_vq, log_vr = self.forward(src, trg[:, :-1])
                loss = criterion(pred, gt_trg[:, 1:], log_vq, log_vr, src, gt_src)
                total += loss.item()
        return total / max(len(dataloader), 1)

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load(path, map_location=loc, weights_only=True)
        )


class ImprovedLSTMLearnedNoise(ImprovedLSTMPredictor):
    """LSTM predictor with Q/R proxy heads (13-dim output)."""

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_dim"] = 13
        super().__init__(*args, **kwargs)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        _, trg_len, _ = trg.size()
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)

        prev = trg[:, 0:1, :]
        outs, vqs, vrs = [], [], []
        for t in range(trg_len):
            inp_embed = self.in_fc(prev)
            out, (h, c) = self.lstm(inp_embed, (h, c))
            raw = self.out_fc(out)
            dbox = raw[:, :, :4]
            conf = torch.sigmoid(raw[:, :, 4:5])
            log_vq = raw[:, :, 5:9]
            log_vr = raw[:, :, 9:13]
            pred_box = prev[:, :, :4] + dbox
            pred = torch.cat([pred_box, conf], dim=-1)
            outs.append(pred)
            vqs.append(log_vq)
            vrs.append(log_vr)
            if t + 1 < trg_len:
                use_teacher = random.random() < teacher_forcing_ratio
                if use_teacher:
                    prev = trg[:, t + 1 : t + 2, :]
                else:
                    next_in = prev.clone()
                    next_in[:, :, :4] = pred[:, :, :4]
                    next_in[:, :, 12:13] = pred[:, :, 4:5]
                    next_in[:, :, 4:12] = 0
                    prev = next_in

        return (
            torch.cat(outs, dim=1),
            torch.cat(vqs, dim=1),
            torch.cat(vrs, dim=1),
        )

    @torch.no_grad()
    def inference(
        self, src: torch.Tensor, trg: torch.Tensor, num_steps: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)
        prev = trg[:, 0:1, :]
        outs, vqs, vrs = [], [], []
        for _ in range(num_steps):
            inp_embed = self.in_fc(prev)
            out, (h, c) = self.lstm(inp_embed, (h, c))
            raw = self.out_fc(out)
            dbox = raw[:, :, :4]
            conf = torch.sigmoid(raw[:, :, 4:5])
            log_vq = raw[:, :, 5:9]
            log_vr = raw[:, :, 9:13]
            pred_box = prev[:, :, :4] + dbox
            pred = torch.cat([pred_box, conf], dim=-1)
            outs.append(pred)
            vqs.append(log_vq)
            vrs.append(log_vr)
            next_in = prev.clone()
            next_in[:, :, :4] = pred[:, :, :4]
            next_in[:, :, 12:13] = pred[:, :, 4:5]
            next_in[:, :, 4:12] = 0
            prev = next_in
        return (
            torch.cat(outs, dim=1),
            torch.cat(vqs, dim=1),
            torch.cat(vrs, dim=1),
        )

    def train_one_epoch(
        self,
        dataloader,
        optimizer,
        criterion: LearnedNoiseMotionLoss,
        device: str = "cuda",
    ) -> float:
        self.train()
        total = 0.0
        for batch in dataloader:
            src, trg, gt_src, gt_trg = batch
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)
            optimizer.zero_grad()
            pred, log_vq, log_vr = self.forward(src, trg[:, :-1])
            loss = criterion(pred, gt_trg[:, 1:], log_vq, log_vr, src, gt_src)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)

    def evaluate(
        self,
        dataloader,
        criterion: LearnedNoiseMotionLoss,
        device: str = "cuda",
    ) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch in dataloader:
                src, trg, gt_src, gt_trg = batch
                src = src.to(device)
                trg = trg.to(device)
                gt_src = gt_src.to(device)
                gt_trg = gt_trg.to(device)
                pred, log_vq, log_vr = self.forward(src, trg[:, :-1])
                loss = criterion(pred, gt_trg[:, 1:], log_vq, log_vr, src, gt_src)
                total += loss.item()
        return total / max(len(dataloader), 1)

    def save_weight(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path: str, map_location: Optional[str] = None) -> None:
        loc = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load(path, map_location=loc, weights_only=True)
        )
