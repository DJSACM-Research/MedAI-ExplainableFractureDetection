#!/usr/bin/env python3
"""
Bootstrapped 95% Confidence Intervals for two headline metrics:

  1. 89.3% accuracy on the HBFMID-derived test split (n=112)
     Source: outputs/conformal/conformal_results.json
     (per-sample argmax_correct flags already persisted — no inference).

  2. 0.652 AUC on FracAtlas — Triplet (w/o RAD-DINO) config (n≈199)
     Option A: loads the three models (MaxViT, YOLO, HyperColumn-CBAM),
     runs inference on the *exact* file list locked in
     outputs/fracatlas_eval/fracatlas_binary_eval_results.json (seed=42),
     caches per-sample scores to outputs/new/fracatlas_scores_cache.npz
     so subsequent runs skip inference entirely.

Outputs (written to outputs/new/):
  bootstrap_ci_results.json      — CI summary
  bootstrap_distributions.png    — histogram of both bootstrap distributions

Usage:
  python scripts/compute_bootstrap_cis.py
  python scripts/compute_bootstrap_cis.py --n-bootstrap 50000
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────
# Script lives at scripts/evaluation/; parents[2] is project root.
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CONFORMAL_JSON = PROJECT_ROOT / "outputs" / "conformal" / "conformal_results.json"
FRACATLAS_JSON = PROJECT_ROOT / "outputs" / "fracatlas_eval" / "fracatlas_binary_eval_results.json"
SCORES_CACHE   = PROJECT_ROOT / "outputs" / "fracatlas_eval" / "fracatlas_scores_cache.npz"
EFRAC_PATH     = PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_fracatlas.py"

OUTPUT_EVAL_DIR    = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Accuracy CI ────────────────────────────────────────────────────

def bootstrap_accuracy(n_bootstrap: int, seed: int):
    """Bootstrap 95% CI for ensemble accuracy using pre-saved per-sample flags."""
    with open(CONFORMAL_JSON) as f:
        data = json.load(f)

    # All alpha keys share the same per_sample list; pick any key
    first_key = next(iter(data))
    per_sample = data[first_key]["per_sample"]
    correct = np.array([int(s["argmax_correct"]) for s in per_sample])
    n = len(correct)
    obs_acc = float(correct.mean())

    rng = np.random.default_rng(seed)
    boot_accs = np.array(
        [rng.choice(correct, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])
    return obs_acc, float(ci_lo), float(ci_hi), boot_accs, n


# ── 2. AUC CI — Option A ─────────────────────────────────────────────

def load_fracatlas_scores() -> tuple:
    """Return (y_true, y_scores) for Triplet (w/o RAD-DINO), using cache if available."""
    if SCORES_CACHE.exists():
        print(f"  Loading cached scores from {SCORES_CACHE.relative_to(PROJECT_ROOT)}")
        data = np.load(SCORES_CACHE)
        return data["y_true"], data["y_scores"]

    print("  Cache not found — running inference for Triplet (w/o RAD-DINO) ...")
    return _run_fracatlas_inference()


def _load_efrac():
    """Import evaluate_fracatlas.py as a module without executing its main()."""
    spec = importlib.util.spec_from_file_location("evaluate_fracatlas", EFRAC_PATH)
    efm  = importlib.util.module_from_spec(spec)
    # Temporarily point sys.argv at something harmless so argparse inside the
    # module's top-level code (if any) does not consume our own argv.
    _orig_argv = sys.argv
    sys.argv = [str(EFRAC_PATH)]
    try:
        spec.loader.exec_module(efm)
    finally:
        sys.argv = _orig_argv
    return efm


def _run_fracatlas_inference():
    """Re-run the Triplet (w/o RAD-DINO) config on the locked FracAtlas file list."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    efm = _load_efrac()
    device = efm.DEVICE

    # ── Load models ───────────────────────────────────────────────────
    print("    [1/3] Loading MaxViT ...")
    maxvit = efm.load_maxvit().to(device).eval()

    print("    [2/3] Loading YOLO ...")
    yolo = efm.load_yolo()

    print("    [3/3] Loading HyperColumn-CBAM-DenseNet169 ...")
    hcbam = efm.load_hypercolumn().to(device).eval()

    transform = efm.get_legacy_transform()

    # ── Reconstruct the locked file list ─────────────────────────────
    with open(FRACATLAS_JSON) as f:
        frac_data = json.load(f)

    si = frac_data["sampling_info"]

    def _resolve_dir(rel: str) -> Path:
        """Try project root first, then data/ prefix (common alternate layout)."""
        for prefix in [PROJECT_ROOT, PROJECT_ROOT / "data"]:
            p = prefix / rel
            if p.exists():
                return p
        raise FileNotFoundError(
            f"FracAtlas directory not found. Tried:\n"
            f"  {PROJECT_ROOT / rel}\n"
            f"  {PROJECT_ROOT / 'data' / rel}"
        )

    frac_dir     = _resolve_dir(si["fractured_dir"])
    non_frac_dir = _resolve_dir(si["non_fractured_dir"])

    samples = (
        [(str(frac_dir     / fn), 1) for fn in si["sampled_files"]["Fractured"]]
      + [(str(non_frac_dir / fn), 0) for fn in si["sampled_files"]["Non_fractured"]]
    )

    # ── Inference ─────────────────────────────────────────────────────
    y_true, y_scores = [], []

    for img_path, label in tqdm(samples, desc="    Inference (Triplet w/o RAD-DINO)"):
        try:
            img = Image.open(img_path).convert("RGB")

            probs = {}
            probs["maxvit"]                    = efm.predict_pytorch(maxvit, img, transform)
            probs["hypercolumn_cbam_densenet169"] = efm.predict_pytorch(hcbam, img, transform)
            probs["yolo"]                      = efm.predict_yolo(yolo, img_path)

            combined       = efm.combine_weighted(probs)
            fracture_score = 1.0 - float(combined[efm.HEALTHY_IDX])

            y_true.append(label)
            y_scores.append(fracture_score)
        except Exception as e:
            print(f"    SKIP {Path(img_path).name}: {e}")

    y_true_arr   = np.array(y_true,   dtype=np.int32)
    y_scores_arr = np.array(y_scores, dtype=np.float64)

    np.savez(SCORES_CACHE, y_true=y_true_arr, y_scores=y_scores_arr)
    print(f"  Scores cached → {SCORES_CACHE.relative_to(PROJECT_ROOT)}")
    return y_true_arr, y_scores_arr


def bootstrap_auc(y_true, y_scores, n_bootstrap: int, seed: int):
    """Bootstrap 95% CI for AUC, skipping degenerate resamples."""
    obs_auc = float(roc_auc_score(y_true, y_scores))
    n = len(y_true)
    idx = np.arange(n)

    rng = np.random.default_rng(seed)
    boot_aucs = []
    for _ in range(n_bootstrap):
        s = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[s], y_scores[s]))

    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    return obs_auc, float(ci_lo), float(ci_hi), boot_aucs, n


# ── Plotting ──────────────────────────────────────────────────────────

def _plot(boot_accs, acc, acc_lo, acc_hi, n_acc,
          boot_aucs, auc, auc_lo, auc_hi, n_auc,
          n_bootstrap: int):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    specs = [
        (axes[0], boot_accs, acc, acc_lo, acc_hi,
         f"Accuracy  (HBFMID-derived split, n={n_acc})", "Accuracy"),
        (axes[1], boot_aucs, auc, auc_lo, auc_hi,
         f"AUC  (FracAtlas, Triplet w/o RAD-DINO, n={n_auc})", "AUC"),
    ]

    for ax, vals, obs, lo, hi, title, xlabel in specs:
        ax.hist(vals, bins=60, color="#4C72B0", alpha=0.75, edgecolor="none")
        ax.axvline(obs, color="#C44E52",  lw=2.0,
                   label=f"Observed: {obs:.4f}")
        ax.axvline(lo,  color="#DD8452",  lw=1.5, ls="--",
                   label=f"95% CI: [{lo:.4f}, {hi:.4f}]")
        ax.axvline(hi,  color="#DD8452",  lw=1.5, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bootstrap count")
        ax.legend(fontsize=9)

    plt.suptitle(
        f"Bootstrapped 95% Confidence Intervals  (B={n_bootstrap:,})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    out_png = OUTPUT_FIGURES_DIR / "bootstrap_distributions.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return out_png


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrapped CIs for accuracy and AUC")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for bootstrap RNG (default: 42)")
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print(f"  Bootstrapped 95% CIs  (B={args.n_bootstrap:,}, seed={args.seed})")
    print(sep)

    # ── Accuracy ───────────────────────────────────────────────────────
    print("\n[1/2] Accuracy — HBFMID-derived test split")
    acc, acc_lo, acc_hi, boot_accs, n_acc = bootstrap_accuracy(
        args.n_bootstrap, args.seed
    )
    print(f"  n                 : {n_acc}")
    print(f"  Observed accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  95% CI            : [{acc_lo:.4f}, {acc_hi:.4f}]"
          f"  ([{acc_lo * 100:.1f}%, {acc_hi * 100:.1f}%])")
    print(f"  Bootstrap std     : {boot_accs.std():.4f}")

    # ── AUC ────────────────────────────────────────────────────────────
    print("\n[2/2] AUC — FracAtlas, Triplet (w/o RAD-DINO)")
    y_true, y_scores = load_fracatlas_scores()
    auc, auc_lo, auc_hi, boot_aucs, n_auc = bootstrap_auc(
        y_true, y_scores, args.n_bootstrap, args.seed
    )
    print(f"  n                 : {n_auc}")
    print(f"  Observed AUC      : {auc:.4f}")
    print(f"  95% CI            : [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  Bootstrap std     : {boot_aucs.std():.4f}")

    # ── Save results JSON ──────────────────────────────────────────────
    results = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "accuracy": {
            "dataset": "HBFMID-derived test split (balanced_augmented_dataset/test.csv)",
            "n_samples": int(n_acc),
            "observed": round(acc, 6),
            "ci_95_lo": round(acc_lo, 6),
            "ci_95_hi": round(acc_hi, 6),
            "ci_95_width": round(acc_hi - acc_lo, 6),
            "boot_std": round(float(boot_accs.std()), 6),
        },
        "auc": {
            "dataset": "FracAtlas — Triplet (w/o RAD-DINO): MaxViT + YOLO + HyperColumn-CBAM",
            "n_samples": int(n_auc),
            "n_bootstrap_valid": int(len(boot_aucs)),
            "observed": round(auc, 6),
            "ci_95_lo": round(auc_lo, 6),
            "ci_95_hi": round(auc_hi, 6),
            "ci_95_width": round(auc_hi - auc_lo, 6),
            "boot_std": round(float(boot_aucs.std()), 6),
        },
    }

    out_json = OUTPUT_EVAL_DIR / "bootstrap_ci_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ───────────────────────────────────────────────────────────
    out_png = _plot(
        boot_accs, acc, acc_lo, acc_hi, n_acc,
        boot_aucs, auc, auc_lo, auc_hi, n_auc,
        args.n_bootstrap,
    )

    print(f"\n  Results  → {out_json.relative_to(PROJECT_ROOT)}")
    print(f"  Plot     → {out_png.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
