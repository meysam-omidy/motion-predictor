"""
Real-detection-paired dataset for adaptive Kalman Q/R training.

Unlike AdaptiveKalmanDataset (GT + synthetic Gaussian noise + fake IoU-confidence),
this pairs each GT track to the REAL detector output per frame:
  - observation box  = matched real detection (IoU > match_iou), else a gap
  - det_score        = real detection confidence
  - R supervision target = (real_detection - GT)^2  (the true measurement error)
  - gaps             = frames where no detection matches the GT (real misses),
                       plus optional random drops to enrich occlusion/Q learning

Produces the same 15-D feature layout as AdaptiveKalmanDataset so the model,
loss, and training loop are unchanged:
  [x, y, w, h, vx..vh, ax..ah, det_score, frames_since_obs, is_observed]
"""
from __future__ import annotations

import os
from copy import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from adaptive_kalman_dataset import (
    FEATURE_DIM, SCORE_IDX, FRAMES_SINCE_IDX, OBSERVED_IDX,
    _compute_motion_features, _attach_gap_features, _has_jump,
)


def _tlbr_from_center(b):  # b: (...,4) center xywh -> tlbr
    o = np.zeros_like(b, dtype=float)
    o[..., 0] = b[..., 0] - b[..., 2] / 2
    o[..., 1] = b[..., 1] - b[..., 3] / 2
    o[..., 2] = b[..., 0] + b[..., 2] / 2
    o[..., 3] = b[..., 1] + b[..., 3] / 2
    return o


def _iou_one_to_many(a, B):  # a:(4,) tlbr, B:(N,4) tlbr
    if len(B) == 0:
        return np.zeros((0,), dtype=float)
    xx1 = np.maximum(a[0], B[:, 0]); yy1 = np.maximum(a[1], B[:, 1])
    xx2 = np.minimum(a[2], B[:, 2]); yy2 = np.minimum(a[3], B[:, 3])
    w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (area_a + area_B - inter + 1e-9)


class AdaptiveKalmanRealDataset(Dataset):
    """MOT sliding windows where the observation is the matched REAL detection."""

    @staticmethod
    def load_sequence(
        seq_path: str,
        det_path: str,
        seq_in_len: int,
        seq_out_len: int,
        seq_total_len: int,
        steps: int,
        match_iou: float,
        random_drop_prob: Optional[float],
        max_gap_norm: float,
        min_observed_frac: float = 0.0,
    ) -> Tuple[list, list, list, list]:
        # print(seq_path)
        # print(det_path)
        sources, targets, gt_sources, gt_targets = [], [], [], []
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        if not (os.path.exists(gt_path) and os.path.exists(det_path)):
            return sources, targets, gt_sources, gt_targets
        

        import configparser
        cfp = configparser.ConfigParser()
        cfp.read(os.path.join(seq_path, "seqinfo.ini"))
        iw = float(cfp["Sequence"]["imWidth"]); ih = float(cfp["Sequence"]["imHeight"])
        borders = np.array([iw, ih, iw, ih], dtype=float)

        # detections: frame, x1, y1, x2, y2, score  (tlbr, pixel)
        det = np.loadtxt(det_path, delimiter=",")
        if det.ndim == 1:
            det = det[None, :]
        det_by_frame = {}
        for fr in np.unique(det[:, 0]):
            rows = det[det[:, 0] == fr]
            det_by_frame[int(fr)] = (rows[:, 1:5].astype(float), rows[:, 5].astype(float))

        df = pd.read_csv(gt_path, header=None)
        df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
        # MOT17/20 GT mixes non-pedestrian classes (7=static person, 9/4/2/8=...) and
        # ignore regions (conf/consider flag = 0). Train only on considered pedestrians.
        # DanceTrack/SportsMOT GT is all (conf=1, class=1) so this filter is a no-op there.
        df = df[(df["conf"] == 1) & (df["class"] == 1)]
        if len(df) == 0:
            return sources, targets, gt_sources, gt_targets
        
        # print('1')

        for _, obj_df in df.groupby("id"):
            obj_df = obj_df.sort_values("frame").copy()
            gcx = (obj_df["x"] + obj_df["w"] / 2).to_numpy()
            gcy = (obj_df["y"] + obj_df["h"] / 2).to_numpy()
            gw = obj_df["w"].to_numpy().astype(float)
            gh = obj_df["h"].to_numpy().astype(float)
            frames_total = obj_df["frame"].to_numpy().astype(int)
            n = len(frames_total)

            gt_box = np.stack([gcx, gcy, gw, gh], axis=1).astype(float)  # pixel center xywh
            # matched real-detection box per frame (pixel center xywh) + score + observed
            obs_box = np.zeros((n, 4), dtype=float)
            obs_score = np.zeros((n,), dtype=float)
            obs_flag = np.zeros((n,), dtype=bool)
            # print(gt_box.shape)
            for k in range(n):
                fr = int(frames_total[k])
                if fr not in det_by_frame:
                    continue
                dboxes_tlbr, dscores = det_by_frame[fr]
                ious = _iou_one_to_many(_tlbr_from_center(gt_box[k]), dboxes_tlbr)
                # print(ious)
                # print(_tlbr_from_center(gt_box[k]))
                # print(dboxes_tlbr)
                # raise SystemExit
                if len(ious) == 0:
                    continue
                j = int(np.argmax(ious))
                if ious[j] < match_iou:
                    continue
                db = dboxes_tlbr[j]
                dw = db[2] - db[0]; dh = db[3] - db[1]
                obs_box[k] = [db[0] + dw / 2, db[1] + dh / 2, dw, dh]
                obs_score[k] = float(dscores[j])
                obs_flag[k] = True

            # normalize
            gt_norm = gt_box / borders
            obs_norm = obs_box / borders

            for i in range(0, n - seq_total_len, steps):
                sl = slice(i, i + seq_total_len)
                frames = frames_total[sl]
                if _has_jump(frames[:seq_in_len]) or _has_jump(frames[-seq_out_len:]):
                    continue
                seq_gt = copy(gt_norm[sl])
                seq_obs = copy(obs_norm[sl])
                seq_score = copy(obs_score[sl])
                seq_flag = copy(obs_flag[sl])

                # Skip windows with too little REAL detection signal in the input
                # context (measured on real matches, BEFORE synthetic drops): a mostly-
                # gap window has nothing to condition on and biases Q toward "always
                # uncertain". 0.0 = keep everything (original behavior).
                if min_observed_frac > 0.0 and seq_flag[:seq_in_len].mean() < min_observed_frac:
                    continue

                # zero-out unmatched frames (real gaps) so gap features fire
                seq_obs[~seq_flag] = 0.0
                seq_score[~seq_flag] = 0.0

                # optional extra random drops (enrich occlusion/Q learning)
                if random_drop_prob and random_drop_prob > 0:
                    drop = np.random.random(size=seq_total_len) < random_drop_prob
                    seq_obs[drop] = 0.0
                    seq_score[drop] = 0.0

                # features from observed detection boxes; GT features from clean GT
                enh = np.zeros((seq_total_len, FEATURE_DIM), dtype=np.float32)
                enh[:, :12] = _compute_motion_features(seq_obs)
                enh[:, SCORE_IDX] = seq_score
                _attach_gap_features(enh, max_gap_norm)

                enh_gt = np.zeros((seq_total_len, FEATURE_DIM), dtype=np.float32)
                enh_gt[:, :12] = _compute_motion_features(seq_gt)
                enh_gt[:, SCORE_IDX] = 1.0
                _attach_gap_features(enh_gt, max_gap_norm)

                sources.append(enh[:seq_in_len])
                targets.append(enh[-seq_out_len:])
                gt_sources.append(enh_gt[:seq_in_len])
                gt_targets.append(enh_gt[-seq_out_len:])

        return sources, targets, gt_sources, gt_targets

    @classmethod
    def from_roots(
        cls,
        root_dirs: Sequence[str],
        detections_dirs: Sequence[str],
        seq_in_len: int = 30,
        seq_out_len: int = 20,
        seq_total_len: int = 50,
        steps: int = 3,
        match_iou: float = 0.5,
        random_drop_prob: Optional[float] = 0.1,
        max_gap_norm: float = 30.0,
        min_observed_frac: float = 0.0,
        dataset_weights: Optional[dict] = None,
    ) -> "AdaptiveKalmanRealDataset":
        sources, targets, gt_sources, gt_targets = [], [], [], []
        for root, det_root in zip(root_dirs, detections_dirs):
            repeat = 1
            if dataset_weights:
                for key, factor in dataset_weights.items():
                    if key.lower() in root.replace("\\", "/").lower():
                        repeat = int(factor); break
            seqs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq in seqs:
                s, t, gs, gt = cls.load_sequence(
                    os.path.join(root, seq), os.path.join(det_root, seq + ".txt"),
                    seq_in_len, seq_out_len, seq_total_len, steps,
                    match_iou, random_drop_prob, max_gap_norm,
                    min_observed_frac=min_observed_frac,
                )
                for _ in range(repeat):
                    sources.extend(s); targets.extend(t)
                    gt_sources.extend(gs); gt_targets.extend(gt)

        obj = cls()
        obj.sources = torch.tensor(np.array(sources), dtype=torch.float32)
        obj.targets = torch.tensor(np.array(targets), dtype=torch.float32)
        obj.gt_sources = torch.tensor(np.array(gt_sources), dtype=torch.float32)
        obj.gt_targets = torch.tensor(np.array(gt_targets), dtype=torch.float32)
        return obj

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return (self.sources[idx], self.targets[idx],
                self.gt_sources[idx], self.gt_targets[idx])
