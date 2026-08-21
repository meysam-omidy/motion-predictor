"""
Train adaptive Kalman Q/R on REAL detections paired to GT (fixes the train/inference
mismatch: the old dataset used GT + synthetic Gaussian noise + fake IoU-confidence).

Same model + loss as train_adaptive_kalman.py; only the dataset changes. R supervision
target (trg - gt_trg)^2 is now the true detector error (matched_detection - GT).

Example:
  python train_adaptive_kalman_real.py --epochs 60 --save_dir ./checkpoints/adaptive_kalman_real
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from adaptive_kalman_dataset import FEATURE_DIM
from adaptive_kalman_dataset_real import AdaptiveKalmanRealDataset
from adaptive_kalman_motion import AdaptiveKalmanLoss, build_adaptive_kalman_model


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)


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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Feature dim: {FEATURE_DIM}")

    common = dict(seq_in_len=args.seq_in_len, seq_out_len=args.seq_out_len,
                  seq_total_len=args.seq_total_len, steps=args.steps,
                  match_iou=args.match_iou, max_gap_norm=args.max_gap_norm,
                  min_observed_frac=args.min_observed_frac)

    # (name, train_path, val_path, detection_dir, oversample_weight). A dataset is
    # used only if BOTH its data root and its detection dir exist (SportsMOT has no
    # detections, so it is skipped automatically).
    datasets = [
        ("MOT17", args.mot17_train_path, args.mot17_val_path, args.mot17_det_dir, args.mot17_weight),
        ("MOT20", args.mot20_train_path, args.mot20_val_path, args.mot20_det_dir, args.mot20_weight),
        ("DanceTrack", args.dancetrack_train_path, args.dancetrack_val_path, args.dancetrack_det_dir, args.dancetrack_weight),
        ("SportsMOT", args.sportsmot_train_path, args.sportsmot_val_path, args.sportsmot_det_dir, args.sportsmot_weight),
    ]
    train_roots, train_dets, val_roots, val_dets, weights, steps_map = [], [], [], [], {}, {}
    for name, tr, va, det, w in datasets:
        det_ok = det and os.path.isdir(det) and len(os.listdir(det)) > 0
        if not det_ok:
            print(f"  skip {name}: no detections at {det}")
            continue
        st = getattr(args, f"{name.lower()}_step", None)  # per-dataset stride override
        if tr and os.path.isdir(tr):
            train_roots.append(tr); train_dets.append(det)
            if w > 1:
                weights[name] = w
            if st is not None:
                steps_map[name] = st
        if va and os.path.isdir(va):
            val_roots.append(va); val_dets.append(det)
    print(f"Datasets used: {[r.split('/')[-2] for r in train_roots]} | weights {weights} "
          f"| steps {steps_map or f'all={args.steps}'}")

    t_gather = time.time()
    train_ds = AdaptiveKalmanRealDataset.from_roots(
        train_roots, train_dets, random_drop_prob=args.random_drop_prob,
        dataset_weights=weights or None, dataset_steps=steps_map or None,
        num_workers=args.gather_workers, **common)
    val_ds = AdaptiveKalmanRealDataset.from_roots(
        val_roots, val_dets, random_drop_prob=args.val_random_drop_prob,
        dataset_weights=None, dataset_steps=steps_map or None,
        num_workers=args.gather_workers, **common)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}  "
          f"(gathered in {time.time()-t_gather:.1f}s)")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty dataset — check paths / detections.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Optionally continue from an already-trained checkpoint. Load it FIRST so the
    # architecture-defining args match the saved weights (else load_state_dict fails).
    ckpt = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("adaptive_qr_version", 1) < 2:
            raise ValueError(
                "Cannot resume a pre-v2 checkpoint: the causal two-stage Q/R "
                "architecture requires retraining."
            )
        ck_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        for k in ("seq_in_len", "max_gap_norm"):
            if k in ck_args and getattr(args, k) != ck_args[k]:
                raise ValueError(
                    f"Resume mismatch for {k}: CLI={getattr(args, k)} but "
                    f"checkpoint={ck_args[k]}. These values define the causal "
                    "training/inference contract."
                )
        for k in ("model_type", "d_model", "nhead", "num_layers", "dim_ff",
                  "lstm_hidden_dim", "lstm_num_layers"):
            if k in ck_args and getattr(args, k, None) != ck_args[k]:
                print(f"  [resume] override {k}: {getattr(args, k, None)} -> {ck_args[k]} (from checkpoint)")
                setattr(args, k, ck_args[k])

    model_kw = dict(input_dim=FEATURE_DIM, d_model=args.d_model, dropout=args.dropout,
                    conf_alpha=args.conf_alpha, max_gap_norm=args.max_gap_norm)
    if args.model_type == "transformer":
        model_kw.update(nhead=args.nhead, num_layers=args.num_layers, dim_ff=args.dim_ff)
    else:
        model_kw.update(hidden_dim=args.lstm_hidden_dim, num_layers=args.lstm_num_layers,
                        teacher_forcing_ratio=args.teacher_forcing_ratio)
    model = build_adaptive_kalman_model(args.model_type, **model_kw).to(device)
    print(f"Model: {args.model_type}, params: {sum(p.numel() for p in model.parameters()):,}")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict)
                              and "model_state_dict" in ckpt else ckpt)
        print(f"  [resume] loaded weights (checkpoint epoch {ckpt.get('epoch', '?')}, "
              f"val {ckpt.get('val_loss', float('nan')):.4f})")

    criterion = AdaptiveKalmanLoss(
        innovation_coeff=args.innovation_coeff, r_supervise_coeff=args.r_supervise_coeff,
        q_gap_coeff=args.q_gap_coeff, q_easy_coeff=args.q_easy_coeff,
        q_gap_trend_coeff=args.q_gap_trend_coeff, innov_obs_weight=args.innov_obs_weight,
        conf_alpha=args.conf_alpha,
        kf_track_coeff=args.kf_track_coeff, kf_gap_weight=args.kf_gap_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(1, args.patience // 3))

    start_epoch, best_val = 1, float("inf")
    if ckpt is not None:
        # Full resume if the checkpoint carries optimizer/scheduler state; otherwise
        # warm-start (fine-tune) from the loaded weights with a fresh optimizer.
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for g in optimizer.param_groups:      # honor the CLI --lr over the saved one
                    g["lr"] = args.lr
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                print(f"  [resume] restored optimizer/scheduler; continuing from epoch {start_epoch}")
            except Exception as e:
                print(f"  [resume] could not restore optimizer state ({e}); fresh optimizer")
        else:
            print("  [resume] warm-start fine-tune (no optimizer state in checkpoint)")
        # A normal resume preserves the existing best threshold so a worse epoch
        # cannot overwrite it. Fine-tuning intentionally starts best-model
        # selection over for this run, even when the checkpoint records a val loss.
        if args.ft:
            print("  [resume] fine-tune mode: ignoring checkpoint best validation loss")
        else:
            best_val = (float(ckpt.get("val_loss", float("inf")))
                        if isinstance(ckpt, dict) else float("inf"))

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    patience = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_metrics": [], "val_metrics": [],
    }

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        train_loss, tm = model.train_one_epoch(train_loader, optimizer, criterion, str(device))
        val_loss, vm = model.evaluate(val_loader, criterion, str(device))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(tm)
        history["val_metrics"].append(vm)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{end_epoch} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | lr {lr:.2e} | {time.time()-t0:.1f}s\n"
            f"  train losses | {format_loss_components(tm)}\n"
            f"  val losses   | {format_loss_components(vm)}\n"
            f"  val stats    | var_q {vm.get('mean_var_q', float('nan')):.3e} | "
            f"var_r {vm.get('mean_var_r', float('nan')):.3e} | "
            f"calib_q {vm.get('calib_q_gap', float('nan')):.4f} | "
            f"gap_frac {vm.get('frac_gap', float('nan')):.4f} | "
            f"r_supervised {vm.get('frac_r_supervised', float('nan')):.4f}"
        )
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            print("Non-finite loss — stopping."); break
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": val_loss, "val_metrics": vm, "args": vars(args),
                        "feature_dim": FEATURE_DIM, "model_type": args.model_type,
                        "adaptive_qr_version": 2, "history_len": args.seq_in_len},
                       save_dir / "best_model.pth")
            print(f"  -> best saved (val {val_loss:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop at epoch {epoch}"); break
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val: {best_val:.4f} -> {save_dir/'best_model.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    DS = "C:/Projects/.Datasets"; DET = "C:/Projects/.Detections/YOLOXx"
    p.add_argument("--mot17_train_path", type=str, default=f"{DS}/MOT17/train")
    p.add_argument("--mot17_val_path", type=str, default=f"{DS}/MOT17/val")
    p.add_argument("--mot17_det_dir", type=str, default=f"{DET}/MOT17")
    p.add_argument("--mot17_weight", type=int, default=1)
    p.add_argument("--mot20_train_path", type=str, default=f"{DS}/MOT20/train")
    p.add_argument("--mot20_val_path", type=str, default=f"{DS}/MOT20/val")
    p.add_argument("--mot20_det_dir", type=str, default=f"{DET}/MOT20")
    p.add_argument("--mot20_weight", type=int, default=1)  # already sample-dominant (dense crowds)
    p.add_argument("--dancetrack_train_path", type=str, default=f"{DS}/DanceTrack/train")
    p.add_argument("--dancetrack_val_path", type=str, default=f"{DS}/DanceTrack/val")
    p.add_argument("--dancetrack_det_dir", type=str, default=f"{DET}/DanceTrack")
    p.add_argument("--dancetrack_weight", type=int, default=1)
    p.add_argument("--sportsmot_train_path", type=str, default=f"{DS}/SportsMOT/train")
    p.add_argument("--sportsmot_val_path", type=str, default=f"{DS}/SportsMOT/val")
    p.add_argument("--sportsmot_det_dir", type=str, default=f"{DET}/SportsMOT")
    p.add_argument("--sportsmot_weight", type=int, default=1)
    p.add_argument("--seq_in_len", type=int, default=30)
    p.add_argument("--seq_out_len", type=int, default=20)
    p.add_argument("--seq_total_len", type=int, default=50)
    p.add_argument("--steps", type=int, default=5, help="default sliding-window stride (fallback)")
    # Per-dataset stride overrides — larger stride = fewer windows, to balance the
    # very different sample counts across datasets (e.g. stride up the large ones).
    p.add_argument("--mot17_step", type=int, default=1)
    # p.add_argument("--mot17_step", type=int, default=1)
    p.add_argument("--mot20_step", type=int, default=15)
    p.add_argument("--dancetrack_step", type=int, default=4)
    p.add_argument("--sportsmot_step", type=int, default=4)
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument("--min_observed_frac", type=float, default=0.0,
                   help="Skip windows whose input context has < this fraction of REAL matched detections (0 = keep all)")
    p.add_argument("--random_drop_prob", type=float, default=0.3)
    p.add_argument("--val_random_drop_prob", type=float, default=0.3)
    p.add_argument("--max_gap_norm", type=float, default=30.0)
    p.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--dim_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lstm_hidden_dim", type=int, default=128)
    p.add_argument("--lstm_num_layers", type=int, default=1)
    p.add_argument("--teacher_forcing_ratio", type=float, default=1)
    p.add_argument("--innovation_coeff", type=float, default=1.0)
    p.add_argument("--r_supervise_coeff", type=float, default=2.0)
    p.add_argument("--q_gap_coeff", type=float, default=0.01)
    p.add_argument("--q_easy_coeff", type=float, default=0.01)
    p.add_argument("--q_gap_trend_coeff", type=float, default=1)
    p.add_argument("--kf_track_coeff", type=float, default=0,
                   help="weight of the differentiable-Kalman-gain loss (joint Q/R vs GT); "
                        "0 = off. Keep innovation_coeff/r_supervise_coeff > 0 as anchors.")
    p.add_argument("--kf_gap_weight", type=float, default=3.0,
                   help="upweight gap frames in the KF-track loss (where coasting/Q matters)")
    p.add_argument("--innov_obs_weight", type=float, default=0.25)
    p.add_argument("--conf_alpha", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gather_workers", type=int, default=0,
                   help="processes for dataset gathering (0=auto min(cpu,8), 1=serial)")
    p.add_argument("--save_dir", type=str, default="./checkpoints/kf_adaptive_kalman_real_med_data_light_lstm_kfoff")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint (e.g. .../best_model.pth) to continue training from. "
                        "Loads model weights (adopting its architecture args); also restores "
                        "optimizer/scheduler/epoch if the checkpoint carries them (full resume), "
                        "otherwise warm-start fine-tune with a fresh optimizer. --epochs is the "
                        "number of epochs THIS run adds.")
    p.add_argument("--ft", action="store_true",
                   help="fine-tuning mode for --resume: ignore the checkpoint's saved "
                        "validation loss when selecting this run's best model")
    main(p.parse_args())
