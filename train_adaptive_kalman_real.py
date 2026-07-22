"""
Train adaptive Kalman Q/R on REAL detections paired to GT (fixes the train/inference
mismatch: the old dataset used GT + synthetic Gaussian noise + fake IoU-confidence).

Same model + loss as train_adaptive_kalman.py; only the dataset changes. R supervision
target (trg - gt_trg)^2 is now the true detector error (matched_detection - GT).

Example:
  python train_adaptive_kalman_real.py --epochs 60 --save_dir ./checkpoints/adaptive_kalman_real
"""
from __future__ import annotations
import argparse, json, math, random, time
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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Feature dim: {FEATURE_DIM}")

    common = dict(seq_in_len=args.seq_in_len, seq_out_len=args.seq_out_len,
                  seq_total_len=args.seq_total_len, steps=args.steps,
                  match_iou=args.match_iou, max_gap_norm=args.max_gap_norm)
    train_ds = AdaptiveKalmanRealDataset.from_roots(
        [args.dancetrack_train_path], [args.detections_dir],
        random_drop_prob=args.random_drop_prob, **common)
    val_ds = AdaptiveKalmanRealDataset.from_roots(
        [args.dancetrack_val_path], [args.detections_dir],
        random_drop_prob=args.val_random_drop_prob, **common)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty dataset — check paths / detections.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model_kw = dict(input_dim=FEATURE_DIM, d_model=args.d_model, dropout=args.dropout,
                    conf_alpha=args.conf_alpha)
    if args.model_type == "transformer":
        model_kw.update(nhead=args.nhead, num_layers=args.num_layers, dim_ff=args.dim_ff)
    else:
        model_kw.update(hidden_dim=args.lstm_hidden_dim, num_layers=args.lstm_num_layers,
                        teacher_forcing_ratio=args.teacher_forcing_ratio)
    model = build_adaptive_kalman_model(args.model_type, **model_kw).to(device)
    print(f"Model: {args.model_type}, params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AdaptiveKalmanLoss(
        innovation_coeff=args.innovation_coeff, r_supervise_coeff=args.r_supervise_coeff,
        q_gap_coeff=args.q_gap_coeff, q_easy_coeff=args.q_easy_coeff,
        q_gap_trend_coeff=args.q_gap_trend_coeff, innov_obs_weight=args.innov_obs_weight,
        conf_alpha=args.conf_alpha)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(1, args.patience // 3))

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); patience = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, tm = model.train_one_epoch(train_loader, optimizer, criterion, str(device))
        val_loss, vm = model.evaluate(val_loader, criterion, str(device))
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.4f} "
              f"(innov {tm.get('loss_innov',0):.4f}, r {tm.get('loss_r',0):.4f}) | "
              f"val {val_loss:.4f} (var_r {vm.get('mean_var_r',float('nan')):.2e}, "
              f"var_q {vm.get('mean_var_q',0):.2e}) | {time.time()-t0:.1f}s")
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            print("Non-finite loss — stopping."); break
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "args": vars(args),
                        "feature_dim": FEATURE_DIM, "model_type": args.model_type},
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
    p.add_argument("--dancetrack_train_path", type=str, default="C:/Projects/.Datasets/DanceTrack/train")
    p.add_argument("--dancetrack_val_path", type=str, default="C:/Projects/.Datasets/DanceTrack/val")
    p.add_argument("--detections_dir", type=str, default="C:/Projects/.Detections/DanceTrack")
    p.add_argument("--seq_in_len", type=int, default=30)
    p.add_argument("--seq_out_len", type=int, default=20)
    p.add_argument("--seq_total_len", type=int, default=50)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument("--random_drop_prob", type=float, default=0.15)
    p.add_argument("--val_random_drop_prob", type=float, default=0.1)
    p.add_argument("--max_gap_norm", type=float, default=30.0)
    p.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lstm_hidden_dim", type=int, default=256)
    p.add_argument("--lstm_num_layers", type=int, default=1)
    p.add_argument("--teacher_forcing_ratio", type=float, default=1)
    p.add_argument("--innovation_coeff", type=float, default=1.0)
    p.add_argument("--r_supervise_coeff", type=float, default=2.0)
    p.add_argument("--q_gap_coeff", type=float, default=0.0)
    p.add_argument("--q_easy_coeff", type=float, default=0.01)
    p.add_argument("--q_gap_trend_coeff", type=float, default=1.0)
    p.add_argument("--innov_obs_weight", type=float, default=0.25)
    p.add_argument("--conf_alpha", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="./checkpoints/adaptive_kalman_real")
    main(p.parse_args())
