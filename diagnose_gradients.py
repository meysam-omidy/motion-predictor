"""
Gradient health check for the adaptive Kalman Q/R model.

Takes one (or a few) dataset samples, runs forward + backward, and reports
whether gradients look healthy (finite, non-vanishing, non-exploding),
especially for the Q/R heads.

Examples:
  # Synthetic batch (no dataset needed)
  python diagnose_gradients.py --synthetic

  # Real MOT sample
  python diagnose_gradients.py --mot17_train_path C:/Projects/.Datasets/MOT17/train

  # Optional checkpoint
  python diagnose_gradients.py --synthetic --checkpoint ./checkpoints/adaptive_kalman_nn/best_model.pth
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from adaptive_kalman_dataset import FEATURE_DIM, AdaptiveKalmanDataset
from adaptive_kalman_motion import (
    AdaptiveKalmanLoss,
    build_adaptive_kalman_model,
    build_cv_innovations,
)


# Heuristic thresholds for "okay" gradients
VANISH_ABS = 1e-12
SMALL_ABS = 1e-8
EXPLODE_ABS = 1e3
EXPLODE_NORM = 1e2
HEAD_MIN_NORM = 1e-8
R_HEAD_DEAD_NORM = 1e-6


@dataclass
class GradStats:
    name: str
    numel: int
    n_finite: int
    n_nan: int
    n_inf: int
    n_zero: int
    n_tiny: int
    abs_mean: float
    abs_max: float
    abs_median: float
    l2_norm: float
    flags: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.startswith("FAIL") for f in self.flags)


def _tensor_grad_stats(name: str, grad: Optional[torch.Tensor]) -> GradStats:
    if grad is None:
        return GradStats(
            name=name,
            numel=0,
            n_finite=0,
            n_nan=0,
            n_inf=0,
            n_zero=0,
            n_tiny=0,
            abs_mean=0.0,
            abs_max=0.0,
            abs_median=0.0,
            l2_norm=0.0,
            flags=["FAIL: no gradient (None)"],
        )

    g = grad.detach().float().reshape(-1)
    abs_g = g.abs()
    finite = torch.isfinite(g)
    n_nan = int((~torch.isfinite(g) & torch.isnan(g)).sum())
    n_inf = int((~torch.isfinite(g) & ~torch.isnan(g)).sum())
    n_finite = int(finite.sum())
    n_zero = int((abs_g == 0).sum())
    n_tiny = int(((abs_g > 0) & (abs_g < VANISH_ABS)).sum())

    if n_finite == 0:
        stats = GradStats(
            name=name,
            numel=g.numel(),
            n_finite=0,
            n_nan=n_nan,
            n_inf=n_inf,
            n_zero=n_zero,
            n_tiny=n_tiny,
            abs_mean=float("nan"),
            abs_max=float("nan"),
            abs_median=float("nan"),
            l2_norm=float("nan"),
            flags=["FAIL: no finite values"],
        )
        return stats

    fin = abs_g[finite]
    stats = GradStats(
        name=name,
        numel=g.numel(),
        n_finite=n_finite,
        n_nan=n_nan,
        n_inf=n_inf,
        n_zero=n_zero,
        n_tiny=n_tiny,
        abs_mean=float(fin.mean()),
        abs_max=float(fin.max()),
        abs_median=float(fin.median()),
        l2_norm=float(g[finite].norm()),
    )

    if n_nan > 0:
        stats.flags.append(f"FAIL: {n_nan} NaNs")
    if n_inf > 0:
        stats.flags.append(f"FAIL: {n_inf} Infs")
    if n_zero == g.numel():
        stats.flags.append("FAIL: all zeros (dead)")
    elif n_zero / g.numel() > 0.95 and g.numel() > 4:
        stats.flags.append(f"WARN: {100 * n_zero / g.numel():.0f}% zeros")
    if n_tiny > 0 and n_tiny / max(n_finite, 1) > 0.5:
        stats.flags.append(f"WARN: {n_tiny} near-vanishing (<{VANISH_ABS:g})")
    if stats.abs_max > EXPLODE_ABS or stats.l2_norm > EXPLODE_NORM:
        stats.flags.append(
            f"FAIL: exploding (max={stats.abs_max:.2e}, ||g||={stats.l2_norm:.2e})"
        )
    elif stats.abs_max < SMALL_ABS and g.numel() > 0:
        stats.flags.append(f"WARN: very small max |g|={stats.abs_max:.2e}")

    if not stats.flags:
        stats.flags.append("OK")
    return stats


def _fmt_stats(s: GradStats) -> str:
    flag = ", ".join(s.flags)
    return (
        f"  {s.name:<28} "
        f"||g||={s.l2_norm:9.2e}  "
        f"mean|g|={s.abs_mean:9.2e}  "
        f"max|g|={s.abs_max:9.2e}  "
        f"zero={s.n_zero}/{s.numel}  "
        f"[{flag}]"
    )


def make_synthetic_batch(
    batch_size: int,
    seq_in: int,
    seq_out: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthetic MOT-like windows with noise + random drops so Q and R
    both get non-trivial loss signal.
    """
    total = seq_in + seq_out
    gt = torch.zeros(batch_size, total, FEATURE_DIM, device=device)
    # smooth-ish random walk in xywh
    gt[:, 0, :4] = torch.tensor([0.4, 0.4, 0.1, 0.2], device=device)
    for t in range(1, total):
        gt[:, t, :4] = gt[:, t - 1, :4] + 0.01 * torch.randn(
            batch_size, 4, device=device
        )
        gt[:, t, 2:4] = gt[:, t, 2:4].clamp(0.02, 0.5)
    # velocity / accel
    gt[:, 1:, 4:8] = gt[:, 1:, :4] - gt[:, :-1, :4]
    gt[:, 2:, 8:12] = gt[:, 2:, 4:8] - gt[:, 1:-1, 4:8]
    gt[..., 12] = 0.85  # score
    gt[..., 13] = 0.0  # gap
    gt[..., 14] = 1.0  # observed

    noisy = gt.clone()
    # detector noise on future part
    noise = 0.05 * torch.randn(batch_size, seq_out, 4, device=device)
    noisy[:, seq_in:, :4] = noisy[:, seq_in:, :4] + noise
    noisy[:, seq_in:, 12] = (0.5 + 0.4 * torch.rand(batch_size, seq_out, device=device))

    # random drops in future
    drop = torch.rand(batch_size, seq_out, device=device) < 0.3
    noisy[:, seq_in:, :12][drop] = 0.0
    noisy[:, seq_in:, 12][drop] = 0.0
    noisy[:, seq_in:, 14][drop] = 0.0
    # crude gap feature
    gap = torch.zeros(batch_size, seq_out, device=device)
    for t in range(seq_out):
        if t == 0:
            gap[:, t] = drop[:, t].float()
        else:
            gap[:, t] = torch.where(drop[:, t], gap[:, t - 1] + 1.0, torch.zeros_like(gap[:, t]))
    noisy[:, seq_in:, 13] = gap / 30.0

    src = noisy[:, :seq_in]
    trg = noisy[:, seq_in - 1 : seq_in + seq_out]  # includes overlap like training? training uses trg full future
    # Match training: trg is seq_out length future window from dataset.
    # Dataset returns targets of length seq_out_len. Training uses trg[:, :-1] as ctx.
    # Here we build trg with length seq_out (future only), same as dataset targets.
    trg = noisy[:, seq_in : seq_in + seq_out]
    gt_src = gt[:, :seq_in]
    gt_trg = gt[:, seq_in : seq_in + seq_out]
    # ensure trg length >= 2 for teacher-forcing slice
    if trg.size(1) < 2:
        raise ValueError("seq_out must be >= 2")
    return src, trg, gt_src, gt_trg


def load_real_batch(args, device: torch.device):
    roots = []
    for name in ("mot17", "mot20", "dancetrack", "sportsmot"):
        p = getattr(args, f"{name}_train_path", None)
        if p:
            roots.append(p)
    if not roots:
        raise ValueError("No dataset path given. Pass --mot17_train_path or use --synthetic.")

    ds = AdaptiveKalmanDataset.from_roots(
        roots,
        seq_in_len=args.seq_in_len,
        seq_out_len=args.seq_out_len,
        seq_total_len=args.seq_total_len,
        steps=args.steps,
        noise_prob=args.noise_prob,
        noise_coeff=args.noise_coeff,
        random_drop_prob=args.random_drop_prob,
    )
    if len(ds) == 0:
        raise ValueError("Dataset is empty - check paths.")

    idx = args.sample_idx % len(ds)
    src, trg, gt_src, gt_trg = ds[idx]
    src = src.unsqueeze(0).to(device)
    trg = trg.unsqueeze(0).to(device)
    gt_src = gt_src.unsqueeze(0).to(device)
    gt_trg = gt_trg.unsqueeze(0).to(device)

    # optional mini-batch around that sample
    if args.batch_size > 1:
        idxs = [(idx + i) % len(ds) for i in range(args.batch_size)]
        batch = [ds[i] for i in idxs]
        src = torch.stack([b[0] for b in batch]).to(device)
        trg = torch.stack([b[1] for b in batch]).to(device)
        gt_src = torch.stack([b[2] for b in batch]).to(device)
        gt_trg = torch.stack([b[3] for b in batch]).to(device)

    return src, trg, gt_src, gt_trg, idx, len(ds)


def build_model(args, device: torch.device) -> nn.Module:
    model_kw = dict(
        input_dim=FEATURE_DIM,
        d_model=args.d_model,
        dropout=0.0,  # deterministic grads for diagnosis
        conf_alpha=args.conf_alpha,
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
            teacher_forcing_ratio=1.0,
        )
    model = build_adaptive_kalman_model(args.model_type, **model_kw).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using randomly initialized model (no --checkpoint)")

    return model


def analyze_nll_grad_signal(log_var_q: torch.Tensor, innovations: torch.Tensor) -> Dict[str, float]:
    """
    Analytic Q NLL gradient signal: 0.5 * (1 - innov^2/var_q).
    Healthy learning wants this not all ~0 and not all huge.
    """
    ell = log_var_q.detach().clamp(-20, 10)
    var_q = ell.exp().clamp(min=1e-6)
    innov_sq = innovations.detach().pow(2).clamp(min=1e-6)
    g = 0.5 * (1.0 - innov_sq / var_q)
    return {
        "nll_grad_mean": float(g.mean()),
        "nll_grad_abs_mean": float(g.abs().mean()),
        "nll_grad_max": float(g.abs().max()),
        "frac_want_larger_q": float((g < 0).float().mean()),  # innov^2 > var_q
        "frac_want_smaller_q": float((g > 0).float().mean()),
        "mean_var_q": float(var_q.mean()),
        "mean_innov_sq": float(innov_sq.mean()),
        "calib_ratio_innov_over_q": float(innov_sq.mean() / var_q.mean()),
    }


@torch.no_grad()
def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def run_term_attribution(
    model: nn.Module,
    criterion: AdaptiveKalmanLoss,
    src: torch.Tensor,
    trg: torch.Tensor,
    gt_src: torch.Tensor,
    gt_trg: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Re-run backward for each loss term alone and measure Q/R head grad norms.
    Shows which term actually drives learning.
    """
    terms = {
        "innov_only": dict(
            innovation_coeff=1.0, r_supervise_coeff=0.0, q_gap_coeff=0.0,
            q_easy_coeff=0.0, q_gap_trend_coeff=0.0, innov_obs_weight=0.25,
        ),
        "r_only": dict(
            innovation_coeff=0.0, r_supervise_coeff=1.0, q_gap_coeff=0.0,
            q_easy_coeff=0.0, q_gap_trend_coeff=0.0,
        ),
        "gap_only": dict(
            innovation_coeff=0.0, r_supervise_coeff=0.0, q_gap_coeff=1.0,
            q_easy_coeff=0.0, q_gap_trend_coeff=0.0,
        ),
        "trend_only": dict(
            innovation_coeff=0.0, r_supervise_coeff=0.0, q_gap_coeff=0.0,
            q_easy_coeff=0.0, q_gap_trend_coeff=1.0,
        ),
        "easy_only": dict(
            innovation_coeff=0.0, r_supervise_coeff=0.0, q_gap_coeff=0.0,
            q_easy_coeff=1.0, q_gap_trend_coeff=0.0,
        ),
    }
    out: Dict[str, Dict[str, float]] = {}

    for name, coeffs in terms.items():
        crit = AdaptiveKalmanLoss(conf_alpha=criterion.conf_alpha, **coeffs)
        model.train()
        _zero_grads(model)
        log_q, log_r = model(src, trg)
        innovations = build_cv_innovations(gt_src, gt_trg, observed=trg[..., 14])
        loss, _ = crit(log_q, log_r, innovations, trg, gt_trg)
        if float(loss) == 0.0:
            out[name] = {"loss": 0.0, "q_head_grad_norm": 0.0, "r_head_grad_norm": 0.0}
            continue
        loss.backward()
        q_g = model.head.q_head.weight.grad
        r_g = model.head.r_residual_head.weight.grad
        out[name] = {
            "loss": float(loss.detach()),
            "q_head_grad_norm": float(q_g.norm()) if q_g is not None else 0.0,
            "r_head_grad_norm": float(r_g.norm()) if r_g is not None else 0.0,
        }
    return out


def diagnose(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    if args.synthetic:
        src, trg, gt_src, gt_trg = make_synthetic_batch(
            args.batch_size, args.seq_in_len, args.seq_out_len, device
        )
        print(f"Synthetic batch: B={src.size(0)}, S={src.size(1)}, T={trg.size(1)}")
        sample_info = "synthetic"
    else:
        src, trg, gt_src, gt_trg, idx, n = load_real_batch(args, device)
        print(f"Dataset sample idx={idx}/{n}, B={src.size(0)}, S={src.size(1)}, T={trg.size(1)}")
        sample_info = f"idx={idx}"

    # quick sample context
    observed = (trg[..., 14] > 0.5).float().mean().item()
    print(f"Sample context: observed_frac={observed:.2f}, mean_score={trg[..., 12].mean().item():.3f}")

    model = build_model(args, device)
    criterion = AdaptiveKalmanLoss(
        innovation_coeff=args.innovation_coeff,
        r_supervise_coeff=args.r_supervise_coeff,
        q_gap_coeff=args.q_gap_coeff,
        q_easy_coeff=args.q_easy_coeff,
        q_gap_trend_coeff=args.q_gap_trend_coeff,
        innov_obs_weight=args.innov_obs_weight,
        conf_alpha=args.conf_alpha,
    )

    model.train()
    _zero_grads(model)

    log_q, log_r = model(src, trg)
    innovations = build_cv_innovations(gt_src, gt_trg, observed=trg[..., 14])
    loss, metrics = criterion(log_q, log_r, innovations, trg, gt_trg)

    print("\n=== Forward ===")
    print(f"sample: {sample_info}")
    print(f"total loss: {float(loss):.6f}")
    for k in (
        "loss_innov", "loss_r", "loss_q_gap", "loss_q_gap_trend", "loss_q_easy",
        "mean_var_q", "mean_var_r", "frac_gap", "frac_r_supervised",
    ):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    if not torch.isfinite(loss):
        print("\nFAIL: loss is not finite - cannot diagnose gradients.")
        return 2

    loss.backward()

    print("\n=== Parameter gradient health ===")
    interesting = []
    all_stats: List[GradStats] = []
    total_sq = 0.0
    n_params_with_grad = 0
    n_params_none = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        st = _tensor_grad_stats(name, p.grad)
        all_stats.append(st)
        if p.grad is None:
            n_params_none += 1
        else:
            n_params_with_grad += 1
            if torch.isfinite(p.grad).any():
                total_sq += float(p.grad[torch.isfinite(p.grad)].float().pow(2).sum())

        # always show heads; show others if not OK or if --verbose
        is_head = "head." in name
        if is_head or args.verbose or (st.flags and st.flags[0] != "OK"):
            interesting.append(st)

    for st in interesting:
        print(_fmt_stats(st))

    global_norm = math.sqrt(total_sq)
    print(f"\nglobal grad L2 norm: {global_norm:.4e}")
    print(f"params with grad: {n_params_with_grad}, without: {n_params_none}")

    # Q NLL analytic signal
    print("\n=== Q NLL analytic gradient signal (0.5*(1 - innov^2/var_q)) ===")
    sig = analyze_nll_grad_signal(log_q, innovations)
    for k, v in sig.items():
        print(f"  {k}: {v:.6g}")

    # Per-term attribution
    print("\n=== Per-loss-term head gradient attribution ===")
    attrib = run_term_attribution(model, criterion, src, trg, gt_src, gt_trg)
    for term, d in attrib.items():
        print(
            f"  {term:<12} loss={d['loss']:.4e}  "
            f"||g_q_head||={d['q_head_grad_norm']:.3e}  "
            f"||g_r_head||={d['r_head_grad_norm']:.3e}"
        )

    # Verdict
    print("\n=== Verdict ===")
    fails = [s for s in all_stats if any(f.startswith("FAIL") for f in s.flags)]
    warns = [s for s in all_stats if any(f.startswith("WARN") for f in s.flags)]
    q_head = next((s for s in all_stats if s.name.endswith("q_head.weight")), None)
    r_head = next((s for s in all_stats if s.name.endswith("r_residual_head.weight")), None)

    issues: List[str] = []
    goods: List[str] = []

    if fails:
        issues.append(f"{len(fails)} parameter tensor(s) FAILED health checks")
    else:
        goods.append("no NaN/Inf/all-zero/exploding grads on checked tensors")

    if global_norm == 0.0:
        issues.append("global grad norm is 0 - nothing is learning")
    elif global_norm > EXPLODE_NORM:
        issues.append(f"global grad norm large ({global_norm:.2e}) - possible explosion")
    else:
        goods.append(f"global grad norm in a reasonable range ({global_norm:.2e})")

    if q_head is not None:
        if q_head.l2_norm < HEAD_MIN_NORM:
            issues.append("Q head weight grad ~0 - Q may not be learning on this sample")
        else:
            goods.append(f"Q head receiving signal (||g||={q_head.l2_norm:.2e})")

    if r_head is not None:
        frac_r = metrics.get("frac_r_supervised", 0.0)
        r_only_norm = attrib["r_only"]["r_head_grad_norm"]
        if frac_r > 0 and r_only_norm < R_HEAD_DEAD_NORM:
            issues.append(
                f"R head DEAD under r_only (||g||={r_only_norm:.2e} < {R_HEAD_DEAD_NORM:g}) "
                f"while frac_r_supervised={frac_r:.2f} - R head not receiving grads"
            )
        elif frac_r == 0 and r_head.l2_norm < HEAD_MIN_NORM:
            goods.append("R head quiet (expected if no noisy observed frames in sample)")
        elif r_head.l2_norm < R_HEAD_DEAD_NORM:
            issues.append(
                f"R head weight grad near-dead (||g||={r_head.l2_norm:.2e}) "
                "despite possible R supervision"
            )
        else:
            goods.append(f"R head receiving signal (||g||={r_head.l2_norm:.2e})")

    # softplus-saturation style check: if var_q tiny and nll wants larger Q but q head dead
    if sig["calib_ratio_innov_over_q"] > 10 and q_head is not None and q_head.l2_norm < HEAD_MIN_NORM:
        issues.append(
            "innov^2 >> var_q but Q head grad dead - classic vanishing/saturation symptom"
        )
    elif sig["nll_grad_abs_mean"] > 0:
        goods.append(
            f"NLL wants Q {'up' if sig['frac_want_larger_q'] > 0.5 else 'down/mixed'} "
            f"(frac_larger={sig['frac_want_larger_q']:.2f})"
        )

    if attrib["innov_only"]["q_head_grad_norm"] < HEAD_MIN_NORM and metrics.get("loss_innov", 0) > 0:
        issues.append("innovation term alone gives ~0 Q-head grad")
    else:
        goods.append("innovation term produces Q-head gradients")

    if (
        metrics.get("frac_r_supervised", 0) > 0
        and attrib["r_only"]["r_head_grad_norm"] >= R_HEAD_DEAD_NORM
    ):
        goods.append(
            f"r_only produces R-head gradients (||g||={attrib['r_only']['r_head_grad_norm']:.2e})"
        )

    for g in goods:
        print(f"  OK   {g}")
    for w in warns[:10]:
        print(f"  WARN {w.name}: {', '.join(w.flags)}")
    for i in issues:
        print(f"  FAIL {i}")

    if issues or fails:
        print("\nOverall: GRADIENTS NOT OK (see FAIL lines)")
        return 1

    print("\nOverall: GRADIENTS LOOK OK on this sample")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose Q/R model gradient health on one sample")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic batch (no dataset)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--verbose", action="store_true", help="Print all parameter grad stats")

    p.add_argument("--mot17_train_path", type=str, default=None)
    p.add_argument("--mot20_train_path", type=str, default=None)
    p.add_argument("--dancetrack_train_path", type=str, default=None)
    p.add_argument("--sportsmot_train_path", type=str, default=None)

    p.add_argument("--seq_in_len", type=int, default=30)
    p.add_argument("--seq_out_len", type=int, default=20)
    p.add_argument("--seq_total_len", type=int, default=50)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--noise_prob", type=float, default=0.3)
    p.add_argument("--noise_coeff", type=float, default=0.1)
    p.add_argument("--random_drop_prob", type=float, default=0.3)

    p.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=512)
    p.add_argument("--lstm_hidden_dim", type=int, default=256)
    p.add_argument("--lstm_num_layers", type=int, default=1)
    p.add_argument("--conf_alpha", type=float, default=2.0)

    p.add_argument("--innovation_coeff", type=float, default=1.0)
    p.add_argument("--r_supervise_coeff", type=float, default=2.0)
    p.add_argument("--q_gap_coeff", type=float, default=0.0)
    p.add_argument("--q_easy_coeff", type=float, default=0.01)
    p.add_argument("--q_gap_trend_coeff", type=float, default=1.0)
    p.add_argument("--innov_obs_weight", type=float, default=0.25)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(diagnose(parse_args()))
