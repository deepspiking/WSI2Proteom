"""CTransPath + MoAE 5-fold CV 결과 시각화"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src))

from wsi2proteom.data.dataset import PairedSlideDataset, collate_paired_samples
from wsi2proteom.data.normalizer import TargetNormalizer
from wsi2proteom.models.baseline import build_model

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})
sns.set_style("whitegrid")


def load_model_and_data(exp_dir: str, device: torch.device):
    """Load checkpoint, rebuild model, and create test DataLoader."""
    ckpt = torch.load(f"{exp_dir}/best_model.pt", map_location="cpu")
    args = argparse.Namespace(**ckpt["args"])

    test_ds = PairedSlideDataset(
        manifest_path=args.manifest,
        proteome_pickle_path=args.proteome,
        split_path=args.split,
        split_name="test",
        feature_model=args.feature_model,
        match_status="ok",
    )

    # Build normalizer from saved state
    norm_state = pickle.load(open(f"{exp_dir}/normalizer.pkl", "rb"))
    normalizer = TargetNormalizer()
    normalizer.load_state_dict(norm_state)

    # Rebuild model
    cls_dims = args.cls_dims if hasattr(args, "cls_dims") and args.cls_dims else None
    model = build_model(
        args.model, in_dim=args.in_dim, out_dim=test_ds.out_dim,
        hidden_dim=args.hidden_dim, hidden_dims=cls_dims,
        dropout=args.dropout, n_heads=args.n_heads, n_layers=args.n_layers,
        top_k=args.top_k, norm=args.norm, residual=args.residual,
        latent_dim=args.latent_dim, use_skip=args.use_skip,
        n_latents=args.n_latents, n_prototypes=args.n_prototypes,
        temperature=args.temperature, n_slots=args.n_slots,
        n_sinkhorn_iters=args.n_sinkhorn_iters,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        n_experts=args.n_experts,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=collate_paired_samples,
                             num_workers=4, pin_memory=True)

    return model, test_loader, normalizer, test_ds


def run_inference(model, loader, normalizer, device):
    """Run inference and return predictions + targets (original scale)."""
    all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            features = [f.to(device) for f in batch["features"]]
            target = batch["target"].to(device)
            pred_norm = model(features)
            pred = normalizer.inverse_transform(pred_norm)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_ids.extend(batch["slide_id"])
    return torch.cat(all_preds), torch.cat(all_targets), all_ids


def compute_sample_pearson(preds, targets):
    """Per-sample Pearson R."""
    rs = []
    for i in range(preds.shape[0]):
        with np.errstate(invalid="ignore"):
            r, _ = pearsonr(preds[i].numpy(), targets[i].numpy())
            rs.append(r if not np.isnan(r) else 0.0)
    return np.array(rs)


def plot_scatter_grid(preds, targets, ids, sample_rs, out_path, n_samples=10):
    """Scatter: predicted vs ground truth for selected samples."""
    # Pick n_samples spanning the R range
    order = np.argsort(sample_rs)
    idxs = np.linspace(0, len(order) - 1, n_samples).astype(int)
    selected = order[idxs]

    cols = 5
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax, idx in zip(axes, selected):
        t = targets[idx].numpy()
        p = preds[idx].numpy()
        ax.scatter(t, p, s=1, alpha=0.3, c="#2196F3", edgecolors="none")
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "r--", lw=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        label = ids[idx] if len(ids[idx]) <= 20 else ids[idx][:17] + "..."
        ax.set_title(f"R={sample_rs[idx]:.3f}  [{label}]")

    for ax in axes[len(selected):]:
        ax.set_visible(False)

    fig.suptitle("CTransPath + MoAE (h=512): Predicted vs Ground Truth (10 test samples)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(preds, targets, ids, sample_rs, out_path, n_proteins=50, n_samples=10):
    """Interleaved heatmap: columns alternate GT | Prediction per sample."""
    # Rank proteins by mean ground-truth expression (highest expressed)
    gt_mean = targets.mean(dim=0).numpy()
    top_idx = np.argsort(-gt_mean)[:n_proteins]

    # Compute per-protein R for annotation
    top_r = np.array([pearsonr(preds[:, i].numpy(), targets[:, i].numpy())[0]
                       for i in top_idx])

    # Pick samples spanning the R range
    order = np.argsort(sample_rs)
    idxs = np.linspace(0, len(order) - 1, n_samples).astype(int)
    sel_idxs = order[idxs]
    n_prot_show = min(n_proteins, 30)

    # Build interleaved matrix: columns = Sample1_GT | Sample1_Pred | Sample2_GT | ...
    gt_data = targets[sel_idxs][:, top_idx[:n_prot_show]].numpy()  # (n_samples, n_prot)
    pr_data = preds[sel_idxs][:, top_idx[:n_prot_show]].numpy()

    interleaved = np.zeros((n_prot_show, n_samples * 2))
    for j in range(n_samples):
        interleaved[:, j * 2] = gt_data[j, :]      # GT column
        interleaved[:, j * 2 + 1] = pr_data[j, :]  # Pred column

    vmax = max(abs(interleaved).max(), 1e-8)

    # Column labels: Sample1_GT | Sample1_Pred | ...
    col_labels = []
    for i in range(n_samples):
        col_labels.append(f"S{i+1} GT\nR={sample_rs[sel_idxs[i]]:.2f}")
        col_labels.append(f"S{i+1} Pred")

    # Row labels: protein rank + R
    row_labels = [f"#{i+1}  R={top_r[i]:.3f}" for i in range(n_prot_show)]

    fig_height = max(6, n_prot_show * 0.35)
    fig, ax = plt.subplots(figsize=(max(10, n_samples * 1.6), fig_height))

    sns.heatmap(interleaved, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={"shrink": 0.4, "label": "Expression"},
                linewidths=0.3, linecolor="#eeeeee")

    # Light vertical separators between sample pairs
    for j in range(1, n_samples):
        ax.axvline(j * 2, color="#999999", lw=0.8, ls="--")

    ax.set_xlabel("Test Sample (GT | Prediction alternating)")
    ax.set_ylabel("Protein (by GT magnitude)")
    ax.set_title(f"CTransPath + MoAE (h=512): GT | Prediction per Sample\n"
                 f"Top-{n_prot_show} proteins | "
                 f"mean R={np.mean(top_r[:n_prot_show]):.4f}",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_sample_r_barchart(sample_rs, ids, out_path):
    """Bar chart of per-sample Pearson R."""
    order = np.argsort(sample_rs)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in sample_rs[order]]
    bars = ax.bar(range(len(sample_rs)), sample_rs[order], color=colors, width=0.7)
    ax.axhline(y=np.mean(sample_rs), color="red", linestyle="--", lw=1.5,
               label=f"Mean R = {np.mean(sample_rs):.4f}")
    ax.set_xlabel("Test Sample (sorted by R)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"CTransPath + MoAE (h=512): Per-Sample Pearson R\n"
                 f"n={len(sample_rs)}, mean={np.mean(sample_rs):.4f}, std={np.std(sample_rs):.4f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str,
                        default="experiments/cv_ctranspath_moae_h512_fold1")
    parser.add_argument("--output", type=str, default="reports/figures")
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)
    exp_name = os.path.basename(args.exp_dir)

    print(f"Loading model from {args.exp_dir} ...")
    model, loader, normalizer, ds = load_model_and_data(args.exp_dir, device)

    print(f"Running inference on {len(ds)} test samples ...")
    preds, targets, ids = run_inference(model, loader, normalizer, device)
    sample_rs = compute_sample_pearson(preds, targets)
    print(f"  Test sample R: mean={sample_rs.mean():.4f} ± {sample_rs.std():.4f}")

    # 1. Scatter grid
    print("\n[1/3] Scatter plot grid ...")
    plot_scatter_grid(preds, targets, ids, sample_rs,
                      f"{args.output}/scatter_{exp_name}.png",
                      n_samples=args.n_samples)

    # 2. Heatmap
    print("[2/3] Heatmap ...")
    plot_heatmap(preds, targets, ids, sample_rs,
                 f"{args.output}/heatmap_{exp_name}.png")

    # 3. Bar chart
    print("[3/3] Per-sample R bar chart ...")
    plot_sample_r_barchart(sample_rs, ids,
                           f"{args.output}/barchart_{exp_name}.png")

    print(f"\nAll figures saved to {args.output}/")


if __name__ == "__main__":
    main()
