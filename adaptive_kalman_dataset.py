"""
Dataset for context-aware adaptive Kalman Q/R training.

Extends GTSequenceDataset with:
- frames_since_last_observation (normalized gap length)
- observed flag (1 = real detection, 0 = dropped / missing frame)

Feature layout (15-D, ``use_motion_features=True``):
  [x, y, w, h, vx..vh, ax..ah, det_score, frames_since_obs, is_observed]
"""

from __future__ import annotations

import os
import random
from copy import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import GTSequenceDataset, batch_iou, has_jump


FEATURE_DIM = 15
SCORE_IDX = 12
FRAMES_SINCE_IDX = 13
OBSERVED_IDX = 14


def _attach_gap_features(seq_enhanced: np.ndarray, max_gap_norm: float = 30.0) -> None:
    """In-place: add frames_since_obs and is_observed from detection score column."""
    n = len(seq_enhanced)
    gap = 0.0
    for i in range(n):
        score = float(seq_enhanced[i, SCORE_IDX])
        observed = score > 1e-4 and np.any(seq_enhanced[i, :4] != 0)
        if observed:
            gap = 0.0
        else:
            gap += 1.0
        seq_enhanced[i, FRAMES_SINCE_IDX] = min(gap / max_gap_norm, 1.0)
        seq_enhanced[i, OBSERVED_IDX] = 1.0 if observed else 0.0


def _enhance_sequence(
    seq: np.ndarray,
    seq_noised: np.ndarray,
    use_motion_features: bool,
    max_gap_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_motion_features:
        seq_enhanced = np.zeros((len(seq), FEATURE_DIM), dtype=np.float32)
        seq_enhanced[:, :12] = GTSequenceDataset.compute_motion_features(seq_noised)
        seq_enhanced_gt = np.zeros((len(seq), FEATURE_DIM), dtype=np.float32)
        seq_enhanced_gt[:, :12] = GTSequenceDataset.compute_motion_features(seq)
    else:
        raise ValueError("AdaptiveKalmanDataset requires use_motion_features=True")

    seq_enhanced[:, SCORE_IDX] = np.diag(batch_iou(seq, seq_noised))
    seq_enhanced_gt[:, SCORE_IDX] = 1.0
    _attach_gap_features(seq_enhanced, max_gap_norm)
    _attach_gap_features(seq_enhanced_gt, max_gap_norm)
    return seq_enhanced, seq_enhanced_gt


class AdaptiveKalmanDataset(Dataset):
    """MOT-style sliding windows with gap context for Q/R-only training."""

    @staticmethod
    def load_sequence(
        seq_path: str,
        seq_in_len: int,
        seq_out_len: int,
        seq_total_len: int,
        steps: int,
        random_jump: bool,
        noise_prob: float,
        noise_coeff: float,
        random_drop_prob: Optional[float],
        max_gap_norm: float = 30.0,
    ) -> Tuple[list, list, list, list]:
        sources: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        gt_sources: List[np.ndarray] = []
        gt_targets: List[np.ndarray] = []

        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            return sources, targets, gt_sources, gt_targets

        import configparser

        cfp = configparser.ConfigParser()
        cfp.read(os.path.join(seq_path, "seqinfo.ini"))
        image_width = float(np.array(cfp["Sequence"]["imWidth"]).astype(float))
        image_height = float(np.array(cfp["Sequence"]["imHeight"]).astype(float))
        borders = np.array(
            [image_width, image_height, image_width, image_height], dtype=float
        )

        import pandas as pd

        df = pd.read_csv(gt_path, header=None)
        df.columns = [
            "frame",
            "id",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "class",
            "visibility",
        ]

        for _, obj_df in df.groupby("id"):
            obj_df = obj_df.sort_values("frame")
            obj_df = obj_df.copy()
            obj_df["x"] = obj_df["x"] + obj_df["w"] / 2
            obj_df["y"] = obj_df["y"] + obj_df["h"] / 2

            bboxes = obj_df[["x", "y", "w", "h"]].to_numpy().astype(float)
            bboxes /= borders
            frames_total = obj_df["frame"].to_numpy()

            for i in range(0, len(bboxes) - seq_total_len, steps):
                seq = copy(bboxes[i : i + seq_total_len])
                noise = np.random.randn(*seq.shape)
                noise[:, 0:2] *= seq[:, 2:4] * noise_coeff
                noise[:, 2:4] *= seq[:, 2:4] * noise_coeff
                seq_noised = np.where(
                    np.random.random(size=seq.shape) < noise_prob, seq + noise, seq
                )

                seq_enhanced, seq_enhanced_gt = _enhance_sequence(
                    seq, seq_noised, True, max_gap_norm
                )

                if random_drop_prob is not None and random_drop_prob > 0:
                    drop_mask = np.random.random(size=len(seq_enhanced)) < random_drop_prob
                    seq_enhanced[drop_mask, SCORE_IDX] = 0.0
                    seq_enhanced[drop_mask, :4] = 0.0
                    seq_enhanced[drop_mask, 4:12] = 0.0
                    _attach_gap_features(seq_enhanced, max_gap_norm)

                frames = frames_total[i : i + seq_total_len]

                if not random_jump:
                    if has_jump(frames[:seq_in_len]) or has_jump(
                        frames[-seq_out_len:]
                    ):
                        continue
                    sources.append(seq_enhanced[:seq_in_len])
                    targets.append(seq_enhanced[-seq_out_len:])
                    gt_sources.append(seq_enhanced_gt[:seq_in_len])
                    gt_targets.append(seq_enhanced_gt[-seq_out_len:])
                else:
                    index_1 = random.randint(0, int(seq_total_len / 2) - seq_in_len - 1)
                    index_2 = random.randint(
                        0, int(seq_total_len / 2) - seq_in_len - 1
                    )
                    if has_jump(
                        frames[index_1 : index_1 + seq_in_len]
                    ) or has_jump(
                        frames[
                            int(seq_total_len / 2)
                            + index_2 : int(seq_total_len / 2)
                            + index_2
                            + seq_out_len
                        ]
                    ):
                        continue
                    sources.append(seq_enhanced[index_1 : index_1 + seq_in_len])
                    targets.append(
                        seq_enhanced[
                            int(seq_total_len / 2)
                            + index_2 : int(seq_total_len / 2)
                            + index_2
                            + seq_out_len
                        ]
                    )
                    gt_sources.append(
                        seq_enhanced_gt[index_1 : index_1 + seq_in_len]
                    )
                    gt_targets.append(
                        seq_enhanced_gt[
                            int(seq_total_len / 2)
                            + index_2 : int(seq_total_len / 2)
                            + index_2
                            + seq_out_len
                        ]
                    )

        return sources, targets, gt_sources, gt_targets

    @classmethod
    def from_roots(
        cls,
        root_dirs: Sequence[str],
        seq_in_len: int = 30,
        seq_out_len: int = 10,
        seq_total_len: int = 40,
        steps: int = 4,
        random_jump: bool = False,
        noise_prob: float = 0.3,
        noise_coeff: float = 0.1,
        random_drop_prob: Optional[float] = 0.3,
        max_gap_norm: float = 30.0,
        dataset_weights: Optional[dict] = None,
    ) -> "AdaptiveKalmanDataset":
        """
        dataset_weights: optional {root_substring: repeat_factor} e.g.
        {'DanceTrack': 3} to oversample DanceTrack sequences.
        """
        sources, targets, gt_sources, gt_targets = [], [], [], []

        for root in root_dirs:
            repeat = 1
            if dataset_weights:
                for key, factor in dataset_weights.items():
                    if key.lower() in root.replace("\\", "/").lower():
                        repeat = int(factor)
                        break

            sequences = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
            for seq_path in sequences:
                s, t, gs, gt = cls.load_sequence(
                    seq_path,
                    seq_in_len,
                    seq_out_len,
                    seq_total_len,
                    steps,
                    random_jump,
                    noise_prob,
                    noise_coeff,
                    random_drop_prob,
                    max_gap_norm,
                )
                for _ in range(repeat):
                    sources.extend(s)
                    targets.extend(t)
                    gt_sources.extend(gs)
                    gt_targets.extend(gt)

        obj = cls()
        obj.sources = np.array(sources, dtype=np.float32)
        obj.targets = np.array(targets, dtype=np.float32)
        obj.gt_sources = np.array(gt_sources, dtype=np.float32)
        obj.gt_targets = np.array(gt_targets, dtype=np.float32)
        return obj

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sources[idx]),
            torch.tensor(self.targets[idx]),
            torch.tensor(self.gt_sources[idx]),
            torch.tensor(self.gt_targets[idx]),
        )
