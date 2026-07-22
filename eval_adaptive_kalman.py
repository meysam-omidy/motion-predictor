"""
Evaluate a trained adaptive Kalman Q/R model.

Reports metrics beyond raw validation loss (see thesis guidance):
- Innovation NLL vs fixed-variance and confidence-R-only baselines
- Variance calibration (predicted vs actual squared innovations)
- Stratified performance: observed vs gap, low vs high detection score
- Q sensitivity to occlusion gap length; R sensitivity to detection score
- Per-dataset breakdown when multiple val roots are given

Example:
  python eval_adaptive_kalman.py \\
    --checkpoint ./checkpoints/adaptive_kalman/best_model.pth \\
    --mot17_val_path ../../Datasets/MOT17/val \\
    --dancetrack_val_path ../../Datasets/DanceTrack/val \\
    --output_dir ./eval/adaptive_kalman
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from adaptive_kalman_dataset import (
    FEATURE_DIM,
    FRAMES_SINCE_IDX,
    OBSERVED_IDX,
    SCORE_IDX,
    AdaptiveKalmanDataset,
)
from adaptive_kalman_motion import (
    AdaptiveKalmanLoss,
    AdaptiveKalmanLSTM,
    build_adaptive_kalman_model,
    build_cv_innovations,
    confidence_log_r_prior,
    exp_var,
    softplus_var,
    _gaussian_nll_logvar,
)


@dataclass
class MetricAccumulator:
    n: int = 0
    sums: Dict[str, float] = field(default_factory=dict)

    def add(self, metrics: Dict[str, float], count: int = 1) -> None:
        self.n += count
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * count

    def mean(self) -> Dict[str, float]:
        if self.n == 0:
            return {}
        return {k: v / self.n for k, v in self.sums.items()}


def _log_2pi_nll(innovations: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian NLL, shape matches innovations."""
    return 0.5 * (
        innovations.pow(2) / var + torch.log(var) + math.log(2 * math.pi)
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_model_and_config(
    checkpoint_path: Path, device: torch.device, model_type: Optional[str] = None
) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    mtype = model_type or ckpt.get("model_type") or train_args.get(
        "model_type", "transformer"
    )

    model_kw = dict(
        input_dim=ckpt.get("feature_dim", FEATURE_DIM),
        d_model=train_args.get("d_model", 256),
        dropout=train_args.get("dropout", 0.1),
        conf_alpha=train_args.get("conf_alpha", 2.0),
    )
    if mtype == "transformer":
        model_kw.update(
            nhead=train_args.get("nhead", 8),
            num_layers=train_args.get("num_layers", 6),
            dim_ff=train_args.get("dim_ff", 1024),
        )
    else:
        model_kw.update(
            hidden_dim=train_args.get("lstm_hidden_dim", 256),
            num_layers=train_args.get("lstm_num_layers", 2),
        )

    model = build_adaptive_kalman_model(mtype, **model_kw).to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, train_args


@torch.no_grad()
def evaluate_batch(
    model: torch.nn.Module,
    src: torch.Tensor,
    trg: torch.Tensor,
    gt_src: torch.Tensor,
    gt_trg: torch.Tensor,
    criterion: AdaptiveKalmanLoss,
    conf_alpha: float,
    fixed_var: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Return scalar metrics and numpy arrays for correlation / calibration."""
    if isinstance(model, AdaptiveKalmanLSTM):
        log_q, log_r = model(src, trg, teacher_forcing_ratio=1.0)
    else:
        log_q, log_r = model(src, trg)

    trg_step = trg
    gt_step = gt_trg
    innovations = build_cv_innovations(
        gt_src, gt_trg, observed=trg_step[:, :, OBSERVED_IDX]
    )

    loss, loss_parts = criterion(log_q, log_r, innovations, trg_step, gt_step)

    # Both heads emit true log-variances; recover with exp (matches AdaptiveKalmanLoss).
    log_q_s = log_q.clamp(min=-20.0, max=10.0)
    log_r_s = log_r.clamp(min=-20.0, max=10.0)
    var_q = exp_var(log_q_s)
    var_r = exp_var(log_r_s)
    innov_sq = innovations.pow(2)

    # Honest fixed baseline: if fixed_var <= 0, use this batch's mean innov^2.
    emp_var = innov_sq.mean().clamp(min=1e-6)
    use_fixed = emp_var if fixed_var <= 0 else torch.tensor(fixed_var, device=emp_var.device)
    nll_model = _gaussian_nll_logvar(innovations, log_q_s, math.log(2 * math.pi)).mean()
    nll_fixed = _log_2pi_nll(innovations, torch.full_like(var_q, float(use_fixed))).mean()

    # Conf-R prior used as a Q substitute is a weak baseline; keep it, but also
    # score how well learned R beats the prior on detector error matching.
    log_r_prior = confidence_log_r_prior(trg_step[..., 12:13], alpha=conf_alpha)
    var_conf_only = exp_var(log_r_prior.expand_as(var_q))
    nll_conf_r = _log_2pi_nll(innovations, var_conf_only).mean()

    observed = trg_step[:, :, OBSERVED_IDX] > 0.5
    gap = ~observed
    meas_sq = (trg_step[..., :4] - gt_step[..., :4]).pow(2).clamp(min=1e-6)
    r_mask = observed.unsqueeze(-1) & (meas_sq.max(dim=-1, keepdim=True).values > 1e-5)
    if r_mask.any():
        log_meas = torch.log(meas_sq[r_mask.expand_as(meas_sq)].clamp(min=1e-8))
        mae_r = (log_r_s.expand_as(var_r)[r_mask.expand_as(var_r)] - log_meas).abs().mean()
        mae_prior = (
            log_r_prior.expand_as(var_r)[r_mask.expand_as(var_r)] - log_meas
        ).abs().mean()
        r_beats_prior = float(mae_r < mae_prior)
        mae_r_val = float(mae_r)
        mae_prior_val = float(mae_prior)
    else:
        r_beats_prior = float("nan")
        mae_r_val = float("nan")
        mae_prior_val = float("nan")

    calib_ratio = (
        (innov_sq[gap.unsqueeze(-1).expand_as(innov_sq)].mean()
         / var_q[gap.unsqueeze(-1).expand_as(var_q)].mean()).item()
        if gap.any()
        else float("nan")
    )

    batch_metrics = {
        "loss_total": loss.item(),
        **loss_parts,
        "nll_model": nll_model.item(),
        "nll_fixed_var": nll_fixed.item(),
        "nll_conf_r_only": nll_conf_r.item(),
        "fixed_var_used": float(use_fixed),
        "mae_log_r": mae_r_val,
        "mae_log_r_prior": mae_prior_val,
        "r_beats_prior": r_beats_prior,
        "calib_ratio": calib_ratio,
        "mean_var_q": var_q.mean().item(),
        "mean_var_r": var_r.mean().item(),
        "mean_innov_sq": innov_sq.mean().item(),
        "calib_q_gap": calib_ratio,
    }

    if observed.any():
        obs_m = observed.unsqueeze(-1).expand_as(innovations)
        batch_metrics["nll_observed"] = (
            _gaussian_nll_logvar(
                innovations[obs_m], log_q_s.expand_as(innovations)[obs_m],
                math.log(2 * math.pi),
            ).mean().item()
        )
        batch_metrics["mean_var_r_obs"] = var_r[obs_m].mean().item()
        batch_metrics["mean_var_q_observed"] = var_q[obs_m].mean().item()
    if gap.any():
        gap_m = gap.unsqueeze(-1).expand_as(innovations)
        batch_metrics["nll_gap"] = (
            _gaussian_nll_logvar(
                innovations[gap_m], log_q_s.expand_as(innovations)[gap_m],
                math.log(2 * math.pi),
            ).mean().item()
        )
        batch_metrics["mean_var_q_gap"] = var_q[gap_m].mean().item()

    arrays = {
        "innov_sq": innov_sq.reshape(-1).cpu().numpy(),
        "var_total": var_q.reshape(-1).cpu().numpy(),
        "var_q": var_q.reshape(-1).cpu().numpy(),
        "var_r": var_r.reshape(-1).cpu().numpy(),
        "scores": trg_step[..., SCORE_IDX].reshape(-1).cpu().numpy(),
        "gap_len": trg_step[:, :, FRAMES_SINCE_IDX].reshape(-1).cpu().numpy(),
        "observed": observed.reshape(-1).cpu().numpy().astype(bool),
    }
    return batch_metrics, arrays


def aggregate_arrays(arrays_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, List[np.ndarray]] = {}
    for arrs in arrays_list:
        for k, v in arrs.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def compute_correlations(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    # Per-step aggregates (mean over xywh dims)
    n_steps = len(arrays["scores"])
    step = 4
    if len(arrays["var_r"]) != n_steps * step:
        step = max(1, len(arrays["var_r"]) // max(n_steps, 1))
    scores = arrays["scores"]
    gap = arrays["gap_len"]
    observed = arrays["observed"].astype(bool)
    var_r = arrays["var_r"].reshape(-1, step).mean(axis=1)
    var_q = arrays["var_q"].reshape(-1, step).mean(axis=1)
    innov_sq = arrays["innov_sq"].reshape(-1, step).mean(axis=1)
    var_total = arrays["var_total"].reshape(-1, step).mean(axis=1)

    # R↔score: observed frames only (gap score is forced to 0 and confounds corr).
    if observed.any():
        corr_r = _pearson(scores[observed], var_r[observed])
        corr_r_oms = _pearson(1.0 - scores[observed], var_r[observed])
    else:
        corr_r = float("nan")
        corr_r_oms = float("nan")

    return {
        "corr_r_score": corr_r,
        "corr_r_one_minus_score": corr_r_oms,
        "corr_r_score_all": _pearson(scores, var_r),
        "corr_q_gap": _pearson(gap, var_q),
        "corr_var_innov_sq": _pearson(var_total, innov_sq),
    }


def calibration_bins(
    predicted_var: np.ndarray,
    actual_sq: np.ndarray,
    n_bins: int = 10,
) -> List[dict]:
    """Bin by predicted variance deciles; report mean predicted vs mean actual."""
    if len(predicted_var) == 0:
        return []
    qs = np.quantile(predicted_var, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            mask = (predicted_var >= lo) & (predicted_var <= hi)
        else:
            mask = (predicted_var >= lo) & (predicted_var < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": i,
                "pred_var_mean": float(predicted_var[mask].mean()),
                "actual_sq_mean": float(actual_sq[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return rows


def verdict(summary: dict) -> dict:
    """Simple pass/fail hints for quick reading."""
    checks = {}

    nll_m = summary.get("nll_model", float("inf"))
    nll_f = summary.get("nll_fixed_var", float("inf"))
    checks["beats_fixed_variance"] = nll_m < nll_f

    rbp = summary.get("r_beats_prior", float("nan"))
    checks["r_beats_prior"] = (rbp == rbp) and rbp > 0.5  # mean of 0/1 flags

    calib = summary.get("calib_q_gap", summary.get("calib_ratio", 0.0))
    checks["calibration_ok"] = (
        calib == calib and 0.25 <= calib <= 4.0  # not NaN
    )

    cq = summary.get("corr_q_gap", float("nan"))
    checks["q_increases_with_gap"] = not math.isnan(cq) and cq > 0.05

    cr = summary.get("corr_r_score", float("nan"))
    checks["r_decreases_with_score"] = not math.isnan(cr) and cr < -0.05

    gap_q = summary.get("mean_var_q_gap", 0.0)
    obs_q = summary.get("mean_var_q_observed", 0.0)
    if gap_q and obs_q:
        checks["q_higher_in_gaps"] = gap_q > obs_q * 1.05
    else:
        checks["q_higher_in_gaps"] = None

    passed = sum(1 for v in checks.values() if v is True)
    total = sum(1 for v in checks.values() if v is not None)
    checks["score"] = f"{passed}/{total}"
    return checks


def print_summary(name: str, summary: dict, checks: dict) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}")
    print(f"  Samples (steps):     {summary.get('n_steps', 0):,}")
    print(f"  Loss (total):        {summary.get('loss_total', 0):.4f}")
    print(f"  Innovation NLL:      {summary.get('nll_model', 0):.4f}")
    print(f"    vs fixed-var:      {summary.get('nll_fixed_var', 0):.4f}  "
          f"(fixed={summary.get('fixed_var_used', float('nan')):.2e}; "
          f"{'better' if checks.get('beats_fixed_variance') else 'worse'})")
    print(f"  Calibration (Q,gap): {summary.get('calib_q_gap', summary.get('calib_ratio', float('nan'))):.3f}  "
          f"(~1.0 ideal; {'ok' if checks.get('calibration_ok') else 'check'})")
    print(f"  Mean var Q / R:      {summary.get('mean_var_q', 0):.2e} / "
          f"{summary.get('mean_var_r', 0):.2e}")
    if "nll_observed" in summary:
        print(f"  NLL observed / gap:  {summary['nll_observed']:.4f} / "
              f"{summary.get('nll_gap', float('nan')):.4f}")
        print(f"  Mean var Q obs/gap:  {summary.get('mean_var_q_observed', 0):.2e} / "
              f"{summary.get('mean_var_q_gap', 0):.2e}")
    print(f"  R log-MAE vs prior:  {summary.get('mae_log_r', float('nan')):.4f} / "
          f"{summary.get('mae_log_r_prior', float('nan')):.4f}  "
          f"({'better' if checks.get('r_beats_prior') else 'worse/same'})")
    print(f"  corr(R, score|obs):  {summary.get('corr_r_score', float('nan')):.3f}  "
          f"(expect negative)")
    print(f"  corr(Q, gap_len):    {summary.get('corr_q_gap', float('nan')):.3f}  "
          f"(expect positive)")
    print(f"  corr(var, innov_sq): {summary.get('corr_var_innov_sq', float('nan')):.3f}")
    print(f"  Checks passed:       {checks.get('score', '?')}")
    for k, v in checks.items():
        if k == "score":
            continue
        if v is None:
            print(f"    {k}: n/a")
        else:
            print(f"    {k}: {'PASS' if v else 'FAIL'}")


def maybe_plot(output_dir: Path, name: str, summary: dict, calib_bins: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if calib_bins:
        fig, ax = plt.subplots(figsize=(6, 5))
        pred = [b["pred_var_mean"] for b in calib_bins]
        actual = [b["actual_sq_mean"] for b in calib_bins]
        ax.plot(pred, actual, "o-", label="binned means")
        lim = max(max(pred), max(actual)) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal")
        ax.set_xlabel("Predicted variance (mean in bin)")
        ax.set_ylabel("Actual innovation sq (mean in bin)")
        ax.set_title(f"Calibration — {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"calibration_{name.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

    arrays_path = output_dir / f"arrays_{name.replace(' ', '_')}.npz"
    if arrays_path.exists():
        data = np.load(arrays_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(
            data["scores"], data["var_r"], s=1, alpha=0.15, c="steelblue"
        )
        axes[0].set_xlabel("Detection score")
        axes[0].set_ylabel("Predicted var R")
        axes[0].set_title("R vs score")
        axes[1].scatter(
            data["gap_len"], data["var_q"], s=1, alpha=0.15, c="darkorange"
        )
        axes[1].set_xlabel("Frames since obs (norm)")
        axes[1].set_ylabel("Predicted var Q")
        axes[1].set_title("Q vs gap length")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(output_dir / f"correlations_{name.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: AdaptiveKalmanLoss,
    conf_alpha: float,
    fixed_var: float,
    device: torch.device,
) -> Tuple[dict, dict, List[dict]]:
    acc = MetricAccumulator()
    all_arrays: List[Dict[str, np.ndarray]] = []

    for src, trg, gt_src, gt_trg in loader:
        src = src.to(device)
        trg = trg.to(device)
        gt_src = gt_src.to(device)
        gt_trg = gt_trg.to(device)
        bsz = src.size(0)
        steps = trg.size(1)

        batch_m, arrs = evaluate_batch(
            model, src, trg, gt_src, gt_trg, criterion, conf_alpha, fixed_var
        )
        acc.add(batch_m, count=bsz * steps)
        all_arrays.append(arrs)

    summary = acc.mean()
    summary["n_steps"] = acc.n
    merged = aggregate_arrays(all_arrays)
    summary.update(compute_correlations(merged))
    bins = calibration_bins(merged["var_total"], merged["innov_sq"])
    return summary, merged, bins


def dataset_eval_kwargs(args) -> dict:
    return dict(
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        seq_total_len=args.seq_total_len,
        steps=args.steps,
        random_jump=False,
        noise_prob=args.noise_prob,
        noise_coeff=args.noise_coeff,
        random_drop_prob=args.random_drop_prob,
        max_gap_norm=args.max_gap_norm,
    )


def main(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, train_args = load_model_and_config(ckpt_path, device, args.model_type)
    conf_alpha = args.conf_alpha or train_args.get("conf_alpha", 2.0)
    criterion = AdaptiveKalmanLoss(conf_alpha=conf_alpha)

    if train_args and args.use_ckpt_seq_config:
        for key in (
            "seq_in_len",
            "seq_out_len",
            "seq_total_len",
            "steps",
            "max_gap_norm",
        ):
            if key in train_args:
                setattr(args, key, train_args[key])
        # Keep eval-specific augmentation (do not copy training drop/noise)
        print(
            f"Sequence config from checkpoint: "
            f"in={args.seq_in_len}, out={args.seq_out_len}, total={args.seq_total_len}"
        )

    val_roots: List[Tuple[str, str]] = []
    for name in ("mot17", "mot20", "dancetrack", "sportsmot"):
        p = getattr(args, f"{name}_val_path", None)
        if p:
            val_roots.append((name.upper(), p))

    if not val_roots and args.val_path:
        val_roots.append(("VAL", args.val_path))

    if not val_roots:
        raise ValueError("Provide at least one --*_val_path or --val_path")

    ds_kw = dataset_eval_kwargs(args)
    report = {"checkpoint": str(ckpt_path), "config": vars(args), "datasets": {}}

    all_summaries = []

    for ds_name, root in val_roots:
        print(f"\nEvaluating on {ds_name}: {root}")
        dataset = AdaptiveKalmanDataset.from_roots([root], **ds_kw)
        if len(dataset) == 0:
            print(f"  WARNING: no samples for {ds_name}, skipping")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        summary, arrays, bins = run_eval(
            model, loader, criterion, conf_alpha, args.fixed_var, device
        )
        checks = verdict(summary)
        report["datasets"][ds_name] = {
            "root": root,
            "n_samples": len(dataset),
            "summary": summary,
            "checks": checks,
            "calibration_bins": bins,
        }
        all_summaries.append(summary)

        slug = ds_name.lower()
        np.savez_compressed(out_dir / f"arrays_{slug}.npz", **arrays)
        print_summary(ds_name, summary, checks)
        if args.plot:
            maybe_plot(out_dir, ds_name, summary, bins)

    if len(all_summaries) > 1:
        combined = MetricAccumulator()
        for s in all_summaries:
            combined.add({k: v for k, v in s.items() if k != "n_steps"}, count=s["n_steps"])
        overall = combined.mean()
        overall["n_steps"] = combined.n
        # Recompute correlations from merged npz if saved
        merged_arrays = []
        for ds_name, _ in val_roots:
            p = out_dir / f"arrays_{ds_name.lower()}.npz"
            if p.exists():
                merged_arrays.append(dict(np.load(p)))
        if merged_arrays:
            overall.update(compute_correlations(aggregate_arrays(merged_arrays)))
        checks = verdict(overall)
        report["overall"] = {"summary": overall, "checks": checks}
        print_summary("OVERALL", overall, checks)

    report_path = out_dir / "eval_report.json"

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and math.isnan(obj):
            return None
        raise TypeError(type(obj))

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate adaptive Kalman Q/R checkpoint")

    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pth or best_weights.pth",
    )
    p.add_argument("--output_dir", type=str, default="./eval/adaptive_kalman")
    p.add_argument("--val_path", type=str, default=None, help="Single generic val root")
    p.add_argument("--mot17_val_path", type=str, default=None)
    # p.add_argument("--mot17_val_path", type=str, default="C:/Projects/.Datasets/MOT17/val")
    p.add_argument("--mot20_val_path", type=str, default=None)
    # p.add_argument("--mot20_val_path", type=str, default="C:/Projects/.Datasets/MOT20/val")
    # p.add_argument("--dancetrack_val_path", type=str, default=None)
    p.add_argument("--dancetrack_val_path", type=str, default="C:/Projects/.Datasets/DanceTrack/val")
    p.add_argument("--sportsmot_val_path", type=str, default=None)
    # p.add_argument("--sportsmot_val_path", type=str, default="C:/Projects/.Datasets/SportsMOT/val")

    p.add_argument(
        "--use_ckpt_seq_config",
        action="store_true",
        default=True,
        help="Load seq_in/out/total and augmentation from checkpoint args",
    )
    p.add_argument(
        "--no_use_ckpt_seq_config",
        action="store_false",
        dest="use_ckpt_seq_config",
    )
    p.add_argument("--seq_in_len", type=int, default=30)
    p.add_argument("--seq_out_len", type=int, default=10)
    p.add_argument("--seq_total_len", type=int, default=40)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--noise_prob", type=float, default=0.2)
    p.add_argument("--noise_coeff", type=float, default=0.1)
    p.add_argument("--random_drop_prob", type=float, default=0.2)
    p.add_argument("--max_gap_norm", type=float, default=30.0)

    p.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--conf_alpha", type=float, default=None)
    p.add_argument(
        "--fixed_var",
        type=float,
        default=-1.0,
        help="Baseline fixed variance; <=0 means use batch mean innov^2 (honest)",
    )

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_false")
    p.add_argument("--plot", action="store_true", help="Save calibration/correlation plots")

    main(p.parse_args())
