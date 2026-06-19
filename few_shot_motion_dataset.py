"""
Few-shot / short-context motion episodes for MOT-style training.

Each sample uses only **k** contiguous past frames (k drawn in [min_context, max_context])
immediately before the prediction horizon. Sources are **left-padded** to `max_context` so
batching stays compatible with fixed-seq models. Motion features are computed on the
**concatenated** [context | future] window so velocity at the first future step uses the
last observed bbox.

Use `actual_context_len` (returned when `return_context_len=True`) to build
`src_key_padding_mask` for `MotionTransformer(..., src_key_padding_mask=...)`.
"""

from __future__ import annotations

import configparser
import os
import random
from copy import copy
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import GTSequenceDataset, batch_iou, has_jump


def _load_few_shot_sequence_simple(
    seq_path: str,
    max_context: int,
    min_context: int,
    seq_out_len: int,
    steps: int,
    noise_prob: float,
    noise_coeff: float,
    random_drop_prob: Optional[float],
    use_motion_features: bool,
    samples_per_window: int,
    rng: random.Random,
) -> Tuple[list, list, list, list, list]:
    """Contiguous few-shot windows only (no random_jump)."""
    sources: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    gt_sources: List[np.ndarray] = []
    gt_targets: List[np.ndarray] = []
    actual_lens: List[int] = []

    gt_path = os.path.join(seq_path, "gt", "gt.txt")
    if not os.path.exists(gt_path):
        return sources, targets, gt_sources, gt_targets, actual_lens

    cfp = configparser.ConfigParser()
    cfp.read(os.path.join(seq_path, "seqinfo.ini"))
    image_width = float(np.array(cfp["Sequence"]["imWidth"]).astype(float))
    image_height = float(np.array(cfp["Sequence"]["imHeight"]).astype(float))
    borders = np.array(
        [image_width, image_height, image_width, image_height], dtype=float
    )

    df = pd.read_csv(gt_path, header=None)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]

    seq_total_len = max_context + seq_out_len

    for _, obj_df in df.groupby("id"):
        obj_df = obj_df.sort_values("frame").copy()
        obj_df["x"] = obj_df["x"] + obj_df["w"] / 2
        obj_df["y"] = obj_df["y"] + obj_df["h"] / 2

        bboxes = obj_df[["x", "y", "w", "h"]].to_numpy().astype(float)
        bboxes /= borders
        frames_total = obj_df["frame"].to_numpy()

        for i in range(0, len(bboxes) - seq_total_len, steps):
            seq = copy(bboxes[i : i + seq_total_len])
            noise = np.random.randn(*(seq.shape))
            noise[:, 0:2] *= seq[:, 2:4] * noise_coeff
            noise[:, 2:4] *= seq[:, 2:4] * noise_coeff
            seq_noised = np.where(
                np.random.random(size=seq.shape) < noise_prob, seq + noise, seq
            )
            frames = frames_total[i : i + seq_total_len]

            for _ in range(max(1, samples_per_window)):
                k = rng.randint(min_context, max_context)
                ctx_start = max_context - k
                if has_jump(frames[ctx_start:max_context]) or has_jump(
                    frames[max_context : max_context + seq_out_len]
                ):
                    continue

                combined_noised = seq_noised[ctx_start : max_context + seq_out_len]
                combined_clean = seq[ctx_start : max_context + seq_out_len]

                if use_motion_features:
                    feature_dim = 13
                    enh_n = np.zeros((len(combined_noised), feature_dim), dtype=np.float32)
                    enh_c = np.zeros_like(enh_n)
                    enh_n[:, :12] = GTSequenceDataset.compute_motion_features(
                        combined_noised
                    )
                    enh_c[:, :12] = GTSequenceDataset.compute_motion_features(
                        combined_clean
                    )
                    enh_n[:, -1] = np.diag(
                        batch_iou(combined_clean, combined_noised)
                    )
                    enh_c[:, -1] = 1.0
                else:
                    feature_dim = 5
                    enh_n = np.zeros((len(combined_noised), feature_dim), dtype=np.float32)
                    enh_c = np.zeros_like(enh_n)
                    enh_n[:, :4] = combined_noised
                    enh_c[:, :4] = combined_clean

                if random_drop_prob is not None and random_drop_prob > 0:
                    drop_mask = (
                        np.random.random(size=len(enh_n)) < random_drop_prob
                    )
                    enh_n[drop_mask, -1] = 0
                    enh_n[drop_mask, :4] = 0
                    if use_motion_features and enh_n.shape[1] >= 13:
                        enh_n[drop_mask, 4:12] = 0

                src_raw = enh_n[:k].astype(np.float32)
                tgt_raw = enh_n[k : k + seq_out_len].astype(np.float32)
                gt_src_raw = enh_c[:k].astype(np.float32)
                gt_tgt_raw = enh_c[k : k + seq_out_len].astype(np.float32)

                pad = np.zeros((max_context - k, feature_dim), dtype=np.float32)
                sources.append(np.vstack([pad, src_raw]))
                targets.append(tgt_raw)
                gt_sources.append(np.vstack([pad, gt_src_raw]))
                gt_targets.append(gt_tgt_raw)
                actual_lens.append(k)

    return sources, targets, gt_sources, gt_targets, actual_lens


class FewShotPaddedMotionDataset(Dataset):
    """
    Padded short-context sources of shape (max_context, F), same targets as full training.

    When ``return_context_len`` is True, ``__getitem__`` returns a 5-tuple with the last
    element ``actual_k`` (scalar tensor) for building padding masks.
    """

    def __init__(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        gt_sources: np.ndarray,
        gt_targets: np.ndarray,
        actual_context_lens: np.ndarray,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        return_context_len: bool = False,
    ):
        self.sources = sources
        self.targets = targets
        self.gt_sources = gt_sources
        self.gt_targets = gt_targets
        self.actual_context_lens = actual_context_lens.astype(np.int64)
        self.image_width = image_width
        self.image_height = image_height
        self.return_context_len = return_context_len

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        src = torch.tensor(self.sources[idx], dtype=torch.float32)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)
        gts = torch.tensor(self.gt_sources[idx], dtype=torch.float32)
        gtt = torch.tensor(self.gt_targets[idx], dtype=torch.float32)
        if self.return_context_len:
            k = int(self.actual_context_lens[idx])
            return src, tgt, gts, gtt, torch.tensor(k, dtype=torch.long)
        return src, tgt, gts, gtt

    @classmethod
    def from_roots(
        cls,
        root_dirs: Sequence[str],
        max_context: int = 20,
        min_context: int = 3,
        seq_out_len: int = 20,
        steps: int = 4,
        random_jump: bool = False,
        noise_prob: float = 0.0,
        noise_coeff: float = 0.0,
        random_drop_prob: Optional[float] = None,
        use_motion_features: bool = True,
        samples_per_window: int = 1,
        seed: Optional[int] = None,
        return_context_len: bool = False,
    ) -> "FewShotPaddedMotionDataset":
        rng = random.Random(seed)
        sources: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        gt_sources: List[np.ndarray] = []
        gt_targets: List[np.ndarray] = []
        lens: List[int] = []

        if random_jump:
            raise ValueError(
                "FewShotPaddedMotionDataset: random_jump=True is not supported yet; "
                "use random_jump=False for variable k contiguous context."
            )

        for root in root_dirs:
            sequences = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
            for seq_path in sequences:
                s, t, gs, gt, ak = _load_few_shot_sequence_simple(
                    seq_path,
                    max_context,
                    min_context,
                    seq_out_len,
                    steps,
                    noise_prob,
                    noise_coeff,
                    random_drop_prob,
                    use_motion_features,
                    samples_per_window,
                    rng,
                )
                sources.extend(s)
                targets.extend(t)
                gt_sources.extend(gs)
                gt_targets.extend(gt)
                lens.extend(ak)

        if not sources:
            return cls(
                np.zeros((0, max_context, 13 if use_motion_features else 5), np.float32),
                np.zeros((0, seq_out_len, 13 if use_motion_features else 5), np.float32),
                np.zeros((0, max_context, 13 if use_motion_features else 5), np.float32),
                np.zeros((0, seq_out_len, 13 if use_motion_features else 5), np.float32),
                np.zeros((0,), np.int64),
                return_context_len=return_context_len,
            )

        obj = cls(
            np.stack(sources, axis=0).astype(np.float32),
            np.stack(targets, axis=0).astype(np.float32),
            np.stack(gt_sources, axis=0).astype(np.float32),
            np.stack(gt_targets, axis=0).astype(np.float32),
            np.array(lens, dtype=np.int64),
            return_context_len=return_context_len,
        )
        return obj

    @classmethod
    def from_sequence(
        cls,
        seq_path: str,
        max_context: int = 20,
        min_context: int = 3,
        seq_out_len: int = 20,
        steps: int = 4,
        noise_prob: float = 0.0,
        noise_coeff: float = 0.0,
        random_drop_prob: Optional[float] = None,
        use_motion_features: bool = True,
        samples_per_window: int = 2,
        seed: int = 0,
        return_context_len: bool = False,
    ) -> "FewShotPaddedMotionDataset":
        rng = random.Random(seed)
        s, t, gs, gt, ak = _load_few_shot_sequence_simple(
            seq_path,
            max_context,
            min_context,
            seq_out_len,
            steps,
            noise_prob,
            noise_coeff,
            random_drop_prob,
            use_motion_features,
            samples_per_window,
            rng,
        )
        if not s:
            fd = 13 if use_motion_features else 5
            return cls(
                np.zeros((0, max_context, fd), np.float32),
                np.zeros((0, seq_out_len, fd), np.float32),
                np.zeros((0, max_context, fd), np.float32),
                np.zeros((0, seq_out_len, fd), np.float32),
                np.zeros((0,), np.int64),
                return_context_len=return_context_len,
            )
        return cls(
            np.stack(s, axis=0).astype(np.float32),
            np.stack(t, axis=0).astype(np.float32),
            np.stack(gs, axis=0).astype(np.float32),
            np.stack(gt, axis=0).astype(np.float32),
            np.array(ak, dtype=np.int64),
            return_context_len=return_context_len,
        )


def build_src_key_padding_mask(
    actual_lens: torch.Tensor, max_context: int, device: torch.device
) -> torch.Tensor:
    """
    True = padded position (ignore in attention).

    actual_lens: (B,) or (B, 1) with true context length k per sample.
    """
    k = actual_lens.view(-1).to(device)
    cols = torch.arange(max_context, device=device).unsqueeze(0).expand(k.shape[0], -1)
    pad_rows = (max_context - k).unsqueeze(1).clamp(min=0)
    return cols < pad_rows
