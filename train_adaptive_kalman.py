"""
Train context-aware adaptive Kalman Q/R predictor (no bbox head).

Recommended defaults follow analysis/THESIS_DIAGNOSTICS_AND_TRACKER_GUIDANCE.md:
- seq_in_len 30–50 aligned with tracker update window
- random_drop_prob 0.25–0.4 for occlusion gaps
- gap-weighted innovation NLL + R supervision (signed log-delta on conf prior)
- Q gap match + gap-only trend correlation; optional DanceTrack oversampling

Example:
  python train_adaptive_kalman.py \\
    --dancetrack_train_path ../../Datasets/DanceTrack/train \\
    --dancetrack_val_path ../../Datasets/DanceTrack/val \\
    --mot17_train_path ../../Datasets/MOT17/train \\
    --mot17_val_path ../../Datasets/MOT17/val \\
    --model_type transformer \\
    --random_drop_prob 0.3 \\
    --dancetrack_weight 3 \\
    --save_dir ./checkpoints/adaptive_kalman
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from adaptive_kalman_dataset import AdaptiveKalmanDataset, FEATURE_DIM
from adaptive_kalman_motion import AdaptiveKalmanLoss, build_adaptive_kalman_model


LOSS_COMPONENTS = (
    ("loss_innov", "innov"),
    ("loss_r", "r"),
    ("loss_q_gap", "q_gap"),
    ("loss_q_gap_trend", "q_trend"),
    ("loss_q_easy", "q_easy"),
    ("loss_kf_track", "kf_track"),
)


def format_loss_components(metrics):
    return " | ".join(
        f"{label} {metrics.get(key, float('nan')):.6f}"
        for key, label in LOSS_COMPONENTS
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collect_paths(args, split: str) -> list[str]:
    paths = []
    for name in ("mot17", "mot20", "dancetrack", "sportsmot"):
        p = getattr(args, f"{name}_{split}_path", None)
        if p:
            paths.append(p)
    return paths


def build_datasets(args):
    weights = {}
    if args.mot17_weight > 1:
        weights["MOT17"] = args.mot17_weight
    if args.mot20_weight > 1:
        weights["MOT20"] = args.mot20_weight
    if args.dancetrack_weight > 1:
        weights["DanceTrack"] = args.dancetrack_weight
    if args.sportsmot_weight > 1:
        weights["SportsMOT"] = args.sportsmot_weight

    train_kw = dict(
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        seq_total_len=args.seq_total_len,
        steps=args.steps,
        random_jump=args.random_jump,
        noise_prob=args.noise_prob,
        noise_coeff=args.noise_coeff,
        random_drop_prob=args.random_drop_prob,
        max_gap_norm=args.max_gap_norm,
        dataset_weights=weights or None,
    )
    val_kw = dict(
        train_kw,
        random_jump=False,
        random_drop_prob=args.val_random_drop_prob,
        noise_prob=args.val_noise_prob,
        noise_coeff=args.val_noise_coeff,
        dataset_weights=None,
    )

    train = AdaptiveKalmanDataset.from_roots(
        collect_paths(args, "train"), **train_kw
    )
    val = AdaptiveKalmanDataset.from_roots(collect_paths(args, "val"), **val_kw)
    return train, val


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Feature dim: {FEATURE_DIM}")

    train_ds, val_ds = build_datasets(args)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty train or val dataset — check dataset paths.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_kw = dict(
        input_dim=FEATURE_DIM,
        d_model=args.d_model,
        dropout=args.dropout,
        conf_alpha=args.conf_alpha,
        max_gap_norm=args.max_gap_norm,
        kalman_head_layers=args.kalman_head_layers,
    )
    if args.model_type == "transformer":
        model_kw.update(
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
        )
    else:
        model_kw.update(
            hidden_dim=args.lstm_hidden_dim,
            num_layers=args.lstm_num_layers,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
        )

    model = build_adaptive_kalman_model(args.model_type, **model_kw).to(device)
    print(f"Model: {args.model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AdaptiveKalmanLoss(
        innovation_coeff=args.innovation_coeff,
        r_supervise_coeff=args.r_supervise_coeff,
        q_gap_coeff=args.q_gap_coeff,
        q_easy_coeff=args.q_easy_coeff,
        q_gap_trend_coeff=args.q_gap_trend_coeff,
        innov_obs_weight=args.innov_obs_weight,
        conf_alpha=args.conf_alpha,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    if args.noise_prob <= 0 and args.random_drop_prob <= 0:
        print(
            "WARNING: noise_prob and random_drop_prob are both 0. "
            "Training uses clean GT with no detector noise or occlusion gaps — "
            "Q/R heads cannot learn meaningful heteroscedastic noise. "
            "Use --noise_prob 0.3 --random_drop_prob 0.3 (defaults) for MOT training."
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = model.train_one_epoch(
            train_loader, optimizer, criterion, str(device)
        )
        val_loss, val_metrics = model.evaluate(
            val_loader, criterion, str(device)
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"lr {lr:.2e} | {time.time()-t0:.1f}s\n"
            f"  train losses | {format_loss_components(train_metrics)}\n"
            f"  val losses   | {format_loss_components(val_metrics)}\n"
            f"  val stats    | var_q {val_metrics.get('mean_var_q', float('nan')):.3e} | "
            f"var_r {val_metrics.get('mean_var_r', float('nan')):.3e} | "
            f"calib_q {val_metrics.get('calib_q_gap', float('nan')):.4f} | "
            f"gap_frac {val_metrics.get('frac_gap', float('nan')):.4f} | "
            f"r_supervised {val_metrics.get('frac_r_supervised', float('nan')):.4f}"
        )

        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            print(
                "Non-finite loss detected — stopping. "
                f"Best checkpoint retained (val {best_val:.4f})."
            )
            break

        if math.isfinite(val_loss):
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "args": vars(args),
                "feature_dim": FEATURE_DIM,
                "adaptive_qr_version": 2,
                "history_len": args.seq_in_len,
                "model_type": args.model_type,
            }
            torch.save(ckpt, save_dir / "best_model.pth")
            model.save_weight(str(save_dir / "best_weights.pth"))
            print(f"  -> best model saved (val {val_loss:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % args.save_every == 0:
            with open(save_dir / f"checkpoint_epoch_{epoch}.json", "w") as f:
                json.dump(
                    {"epoch": epoch, "history": history, "args": vars(args)},
                    f,
                    indent=2,
                )

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val loss: {best_val:.4f} -> {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train adaptive Kalman Q/R motion predictor (thesis model)"
    )

    p.add_argument("--mot17_train_path", type=str, default="C:/Projects/.Datasets/MOT17/train")
    # p.add_argument("--mot20_train_path", type=str, default=None)
    p.add_argument("--mot20_train_path", type=str, default="C:/Projects/.Datasets/MOT20/train")
    # p.add_argument("--dancetrack_train_path", type=str, default=None)
    p.add_argument("--dancetrack_train_path", type=str, default="C:/Projects/.Datasets/DanceTrack/train")
    # p.add_argument("--sportsmot_train_path", type=str, default=None)
    p.add_argument("--sportsmot_train_path", type=str, default="C:/Projects/.Datasets/SportsMOT/train")
    p.add_argument("--mot17_val_path", type=str, default="C:/Projects/.Datasets/MOT17/val")
    # p.add_argument("--mot20_val_path", type=str, default=None)
    p.add_argument("--mot20_val_path", type=str, default="C:/Projects/.Datasets/MOT20/val")
    # p.add_argument("--dancetrack_val_path", type=str, default=None)
    p.add_argument("--dancetrack_val_path", type=str, default="C:/Projects/.Datasets/DanceTrack/val")
    # p.add_argument("--sportsmot_val_path", type=str, default=None)
    p.add_argument("--sportsmot_val_path", type=str, default="C:/Projects/.Datasets/SportsMOT/val")

    p.add_argument("--seq_in_len", type=int, default=30)
    p.add_argument("--seq_out_len", type=int, default=20)
    p.add_argument("--seq_total_len", type=int, default=50)
    p.add_argument("--steps", type=int, default=3)
    # p.add_argument("--random_jump", action="store_false")
    # p.add_argument("--noise_prob", type=float, default=0)
    # p.add_argument("--noise_coeff", type=float, default=0.1)
    # p.add_argument("--random_drop_prob", type=float, default=0)
    # p.add_argument("--val_noise_prob", type=float, default=0)
    # p.add_argument("--val_noise_coeff", type=float, default=0.1)
    # p.add_argument("--val_random_drop_prob", type=float, default=0)
    # p.add_argument("--max_gap_norm", type=float, default=30.0)
    p.add_argument("--random_jump", action="store_true")
    p.add_argument("--noise_prob", type=float, default=0.3)
    p.add_argument("--noise_coeff", type=float, default=0.1)
    p.add_argument("--random_drop_prob", type=float, default=0.3)
    p.add_argument("--val_noise_prob", type=float, default=0.2)
    p.add_argument("--val_noise_coeff", type=float, default=0.1)
    p.add_argument("--val_random_drop_prob", type=float, default=0.2)
    p.add_argument("--max_gap_norm", type=float, default=30.0)
    p.add_argument("--mot17_weight", type=int, default=4)
    p.add_argument("--mot20_weight", type=int, default=1)
    p.add_argument("--dancetrack_weight", type=int, default=2)
    p.add_argument("--sportsmot_weight", type=int, default=1)

    p.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["transformer", "lstm"],
    )
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kalman_head_layers", type=int, default=3,
                   help="number of Linear projections in each Q/R Kalman-noise head (>= 1)")
    p.add_argument("--lstm_hidden_dim", type=int, default=256)
    p.add_argument("--lstm_num_layers", type=int, default=1)
    p.add_argument("--teacher_forcing_ratio", type=float, default=1)

    p.add_argument("--innovation_coeff", type=float, default=1.0)
    p.add_argument("--r_supervise_coeff", type=float, default=2.0)
    # Off by default: the log-space gap match collapses var_q to the floor on
    # heavily-interpolated GT (most innov²≈0). The innovation NLL calibrates Q.
    p.add_argument("--q_gap_coeff", type=float, default=0.0)
    p.add_argument("--q_easy_coeff", type=float, default=0.01)
    p.add_argument("--q_gap_trend_coeff", type=float, default=1.0)
    p.add_argument(
        "--innov_obs_weight",
        type=float,
        default=0.25,
        help="Weight on innovation NLL for observed steps (gaps use weight 1).",
    )
    p.add_argument("--conf_alpha", type=float, default=2.0)

    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="./checkpoints/adaptive_kalman_3")
    p.add_argument("--save_every", type=int, default=5)

    main(p.parse_args())
