"""
Label-free dataset for self-supervised adaptive Kalman Q/R training (paper upgrade U1).

Windows come from the TRACKER's own tracklets instead of ground-truth tracks, so
any video with a detector file can be used (including test splits, which have no
labels). Inputs per sequence:

  - ``seqinfo.ini``           image size (no images or ground truth needed)
  - tracker output (MOT txt)  frame, id, x, y, w, h, score, ...   (one tracklet per id)
  - raw detections            frame, x1, y1, x2, y2, score        (exact boxes/scores)

Each output box is matched back to the raw detection it came from (IoU >= 0.9) to
recover the exact box and full-precision score; frames where an id is absent are
gaps. Features use the same 15-D (or 18-D with detection context) gap-safe builder
as the other datasets.

There is NO ground truth here: the returned ``gt_src``/``gt_trg`` are copies of the
observations so the training loop's tensors keep their shape, and the training
script refuses any loss term that would read them (only the innovation-likelihood
loss, ``--innovation_kf_coeff``, and GT-free regularizers may be used).
"""
from __future__ import annotations

import configparser
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from adaptive_kalman_dataset import (
    DETECTION_CONTEXT_DIM, build_adaptive_kalman_features, compute_detection_context,
)

MATCH_IOU = 0.9


def _iou_one_to_many(a, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=float)
    lt = np.maximum(a[:2], boxes[:, :2])
    rb = np.minimum(a[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def _image_size(seq_dir: str) -> tuple[float, float]:
    cfp = configparser.ConfigParser()
    cfp.read(os.path.join(seq_dir, "seqinfo.ini"))
    return float(cfp["Sequence"]["imWidth"]), float(cfp["Sequence"]["imHeight"])


def load_tracklet_windows(
    seq_dir: str,
    tracker_output: str,
    detection_file: str,
    seq_in_len: int = 30,
    seq_out_len: int = 20,
    steps: int = 3,
    max_gap_norm: float = 30.0,
    min_observed_frac: float = 0.5,
    random_drop_prob: float = 0.0,
    use_detection_context: bool = False,
    seed: Optional[int] = None,
):
    """Return (sources, targets) feature windows for every tracklet of a sequence."""
    rng = np.random.default_rng(seed)
    iw, ih = _image_size(seq_dir)
    borders = np.array([iw, ih, iw, ih], dtype=float)
    seq_total_len = seq_in_len + seq_out_len

    det = pd.read_csv(detection_file, header=None).to_numpy(dtype=float)
    det_by_frame, context_by_frame = {}, {}
    for fr in np.unique(det[:, 0]):
        rows = det[det[:, 0] == fr]
        det_by_frame[int(fr)] = (rows[:, 1:5], rows[:, 5])
        if use_detection_context:
            context_by_frame[int(fr)] = compute_detection_context(rows[:, 1:5], iw, ih)

    out = pd.read_csv(tracker_output, header=None).to_numpy(dtype=float)
    sources, targets = [], []
    for track_id in np.unique(out[:, 1]):
        rows = out[out[:, 1] == track_id]
        rows = rows[np.argsort(rows[:, 0])]
        first, last = int(rows[0, 0]), int(rows[-1, 0])
        n = last - first + 1
        if n < seq_total_len:
            continue
        box = np.zeros((n, 4)); score = np.zeros(n); flag = np.zeros(n, dtype=bool)
        context = np.zeros((n, DETECTION_CONTEXT_DIM))
        for row in rows:
            k = int(row[0]) - first
            tlbr = np.array([row[2], row[3], row[2] + row[4], row[3] + row[5]])
            s = float(row[6])
            frame_dets = det_by_frame.get(int(row[0]))
            if frame_dets is not None and len(frame_dets[0]):
                ious = _iou_one_to_many(tlbr, frame_dets[0])
                j = int(np.argmax(ious))
                if ious[j] >= MATCH_IOU:
                    tlbr, s = frame_dets[0][j], float(frame_dets[1][j])
                    if use_detection_context:
                        context[k] = context_by_frame[int(row[0])][j]
            w, h = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
            box[k] = [tlbr[0] + w / 2, tlbr[1] + h / 2, w, h]
            score[k], flag[k] = s, True
        box /= borders
        for i in range(0, n - seq_total_len + 1, steps):
            sl = slice(i, i + seq_total_len)
            mask = flag[sl].copy()
            if mask[:seq_in_len].mean() < min_observed_frac or not mask[seq_in_len:].any():
                continue
            if random_drop_prob > 0:
                mask &= ~(rng.random(seq_total_len) < random_drop_prob)
            feats = build_adaptive_kalman_features(
                box[sl], score[sl], mask, max_gap_norm,
                context=context[sl] if use_detection_context else None,
            )
            sources.append(feats[:seq_in_len])
            targets.append(feats[seq_in_len:])
    return sources, targets


class AdaptiveKalmanTrackletDataset(Dataset):
    """Windows over tracker tracklets; ground-truth slots hold the observations."""

    @classmethod
    def from_runs(
        cls,
        runs: Sequence[tuple[str, str, str]],
        **kwargs,
    ) -> "AdaptiveKalmanTrackletDataset":
        """``runs``: (sequence_dir, tracker_output_txt, detection_txt) triples."""
        sources, targets = [], []
        for index, (seq_dir, output, detection) in enumerate(runs):
            s, t = load_tracklet_windows(seq_dir, output, detection, seed=1_000_003 + index, **kwargs)
            sources.extend(s); targets.extend(t)
        obj = cls()
        dim = (15 + DETECTION_CONTEXT_DIM) if kwargs.get("use_detection_context") else 15
        seq_in = kwargs.get("seq_in_len", 30); seq_out = kwargs.get("seq_out_len", 20)
        obj.sources = torch.tensor(np.array(sources), dtype=torch.float32).reshape(-1, seq_in, dim)
        obj.targets = torch.tensor(np.array(targets), dtype=torch.float32).reshape(-1, seq_out, dim)
        return obj

    @staticmethod
    def discover_runs(sequences_root: str, outputs_dir: str, detections_dir: str):
        """Pair every sequence folder with <outputs_dir>/<seq>.txt and <detections_dir>/<seq>.txt."""
        runs = []
        for seq in sorted(os.listdir(sequences_root)):
            seq_dir = os.path.join(sequences_root, seq)
            output = os.path.join(outputs_dir, f"{seq}.txt")
            detection = os.path.join(detections_dir, f"{seq}.txt")
            if os.path.isdir(seq_dir) and os.path.isfile(output) and os.path.isfile(detection):
                runs.append((seq_dir, output, detection))
        return runs

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        # No ground truth: the GT slots repeat the observations (never read by
        # the self-supervised loss; the training script enforces that).
        return self.sources[idx], self.targets[idx], self.sources[idx], self.targets[idx]
