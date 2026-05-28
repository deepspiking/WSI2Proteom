from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src))

from wsi2proteom.data.dataset import PairedSlideDataset, collate_paired_samples
from wsi2proteom.data.normalizer import TargetNormalizer
from wsi2proteom.models.baseline import build_model, FactorizedProteomeHead


def pearson_r_per_protein(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    t_mean = y_true.mean(dim=0, keepdim=True)
    p_mean = y_pred.mean(dim=0, keepdim=True)
    t_centered = y_true - t_mean
    p_centered = y_pred - p_mean
    cov = (t_centered * p_centered).sum(dim=0)
    t_var = (t_centered ** 2).sum(dim=0)
    p_var = (p_centered ** 2).sum(dim=0)
    denom = (t_var * p_var).clamp(min=1e-8).sqrt()
    return cov / denom


def spearman_r_per_protein(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    n = y_true.shape[0]
    true_rank = y_true.argsort(dim=0).argsort(dim=0).float()
    pred_rank = y_pred.argsort(dim=0).argsort(dim=0).float()
    d = true_rank - pred_rank
    return 1 - (6 * (d ** 2).sum(dim=0)) / (n * (n ** 2 - 1) + 1e-8)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


def train_one_epoch(model, loader, optimizer, normalizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    pearson_meter = AverageMeter()
    spearman_meter = AverageMeter()

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        features = [f.to(device) for f in batch["features"]]
        target = batch["target"].to(device)
        target_norm = normalizer.transform(target).to(device)

        pred = model(features)
        loss = F.smooth_l1_loss(pred, target_norm)

        with torch.no_grad():
            r = pearson_r_per_protein(target_norm, pred)
            r = r[~torch.isnan(r)]
            pearson = r.mean().item() if r.numel() > 0 else 0.0
            s = spearman_r_per_protein(target_norm, pred)
            s = s[~torch.isnan(s)]
            spearman = s.mean().item() if s.numel() > 0 else 0.0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_meter.update(loss.item(), 1)
        pearson_meter.update(pearson, 1)
        spearman_meter.update(spearman, 1)

    return loss_meter.avg, pearson_meter.avg, spearman_meter.avg


@torch.no_grad()
def evaluate(model, loader, normalizer, device, desc="Eval"):
    model.eval()
    loss_meter = AverageMeter()
    pearson_meter = AverageMeter()
    spearman_meter = AverageMeter()
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc=desc, leave=False):
        features = [f.to(device) for f in batch["features"]]
        target = batch["target"].to(device)
        target_norm = normalizer.transform(target).to(device)

        pred = model(features)
        loss = F.smooth_l1_loss(pred, target_norm)

        r = pearson_r_per_protein(target_norm, pred)
        r = r[~torch.isnan(r)]
        pearson = r.mean().item() if r.numel() > 0 else 0.0
        s = spearman_r_per_protein(target_norm, pred)
        s = s[~torch.isnan(s)]
        spearman = s.mean().item() if s.numel() > 0 else 0.0

        loss_meter.update(loss.item(), 1)
        pearson_meter.update(pearson, 1)
        spearman_meter.update(spearman, 1)

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    return loss_meter.avg, pearson_meter.avg, spearman_meter.avg, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/paired_manifest.csv")
    parser.add_argument("--proteome", type=str,
                        default="/data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl")
    parser.add_argument("--split", type=str, default="data/splits/split_v1.csv")
    parser.add_argument("--feature_model", type=str, default="ctranspath")
    parser.add_argument("--in_dim", type=int, default=0,
                        help="Feature dimension (0=auto-detect)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Attention hidden dim (auto-scaled for retccl)")
    parser.add_argument("--model", type=str, default="meanpool_linear",
                        choices=["meanpool_linear", "meanpool", "attention"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="experiments/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--factorized", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--cls_dims", type=str, default=None,
                        help="Comma-separated classifier hidden dims, e.g. '1024,512'. Default = [hidden_dim, hidden_dim//2]")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in classifier (default: 0.3)")
    args = parser.parse_args()

    if args.cls_dims is not None:
        args.cls_dims = [int(x) for x in args.cls_dims.split(",")] if args.cls_dims else []

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairedSlideDataset(
        manifest_path=args.manifest,
        proteome_pickle_path=args.proteome,
        split_path=args.split,
        split_name="train",
        feature_model=args.feature_model,
        match_status="ok",
    )
    val_ds = PairedSlideDataset(
        manifest_path=args.manifest,
        proteome_pickle_path=args.proteome,
        split_path=args.split,
        split_name="val",
        feature_model=args.feature_model,
        match_status="ok",
    )
    test_ds = PairedSlideDataset(
        manifest_path=args.manifest,
        proteome_pickle_path=args.proteome,
        split_path=args.split,
        split_name="test",
        feature_model=args.feature_model,
        match_status="ok",
    )

    out_dim = train_ds.out_dim
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Output dimension: {out_dim}")

    if args.in_dim == 0:
        first_row = train_ds.manifest.iloc[0]
        feat = torch.load(first_row["feature_path"], map_location="cpu")
        args.in_dim = feat["features"].shape[-1]
        print(f"Auto-detected feature dim: {args.in_dim}")

    hidden_dim = args.hidden_dim
    if args.model == "attention" and args.in_dim >= 1024:
        hidden_dim = max(args.hidden_dim, args.in_dim // 4)
        print(f"Scaled attention hidden_dim to {hidden_dim} (in_dim={args.in_dim})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_paired_samples,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_paired_samples,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_paired_samples,
                             num_workers=4, pin_memory=True)

    with open(args.proteome, "rb") as f:
        target_dict = pickle.load(f)
    train_aliquots = train_ds.manifest["aliquot_submitter_id"].tolist()
    train_targets = np.stack([target_dict[a].astype(np.float32, copy=False) for a in train_aliquots])
    normalizer = TargetNormalizer()
    normalizer.fit(train_targets)
    with open(out_dir / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer.state_dict(), f)

    if args.factorized and args.model == "attention":
        n_train = len(train_targets)
        latent_dim = min(args.latent_dim, n_train - 1)
        if latent_dim < args.latent_dim:
            print(f"Clamping latent_dim from {args.latent_dim} to {latent_dim} (n_train={n_train})")
        pca = PCA(n_components=latent_dim, svd_solver="randomized")
        train_targets_norm = normalizer.transform(train_targets).numpy()
        pca.fit(train_targets_norm)
        decoder_weight = torch.from_numpy(pca.components_.T).float()
        decoder_bias = torch.from_numpy(pca.mean_).float()

        slide_encoder = build_model("attention", in_dim=args.in_dim, out_dim=out_dim,
                                    hidden_dim=hidden_dim, hidden_dims=[hidden_dim, hidden_dim // 2])
        model = FactorizedProteomeHead(slide_encoder, hidden_dim=hidden_dim,
                                       latent_dim=latent_dim,
                                       decoder_weight=decoder_weight,
                                       decoder_bias=decoder_bias)
        print(f"Factorized head: latent_dim={latent_dim}")
        args.latent_dim = latent_dim
    else:
        cls_dims = args.cls_dims if args.cls_dims is not None else [hidden_dim, hidden_dim // 2]
        model = build_model(args.model, in_dim=args.in_dim, out_dim=out_dim,
                            hidden_dim=hidden_dim, hidden_dims=cls_dims,
                            dropout=args.dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_pearson": [], "val_pearson": [],
               "train_spearman": [], "val_spearman": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_r, train_s = train_one_epoch(
            model, train_loader, optimizer, normalizer, device, epoch)
        val_loss, val_r, val_s, _, _ = evaluate(
            model, val_loader, normalizer, device, desc=f"Val E{epoch}")

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pearson"].append(train_r)
        history["val_pearson"].append(val_r)
        history["train_spearman"].append(train_s)
        history["val_spearman"].append(val_s)

        elapsed = time.time() - start_time
        print(f"E{epoch:3d} | train_loss={train_loss:.4f} train_r={train_r:.4f} "
              f"train_s={train_s:.4f} | val_loss={val_loss:.4f} val_r={val_r:.4f} "
              f"val_s={val_s:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_pearson": val_r,
                "val_spearman": val_s,
                "args": vars(args),
            }, out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n--- Testing best model ---")
    ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    test_loss, test_r, test_s, test_preds, test_targets = evaluate(
        model, test_loader, normalizer, device, desc="Test"
    )
    print(f"Test loss={test_loss:.4f} pearson_r={test_r:.4f} spearman_r={test_s:.4f}")

    all_preds = torch.cat(test_preds, dim=0)
    all_targets = torch.cat(test_targets, dim=0)

    sample_rs = []
    for i in range(all_preds.shape[0]):
        r = torch.corrcoef(torch.stack([all_preds[i], all_targets[i]]))[0, 1].item()
        if not np.isnan(r):
            sample_rs.append(r)
    mean_sample_r = np.mean(sample_rs) if sample_rs else 0.0
    print(f"Per-sample Pearson R: mean={mean_sample_r:.4f} n={len(sample_rs)}")

    protein_rs = pearson_r_per_protein(all_targets, all_preds)
    protein_rs = protein_rs[~torch.isnan(protein_rs)].sort(descending=True).values
    n_prot = len(protein_rs)
    top1k = protein_rs[:min(1000, n_prot)].mean().item()
    median_r = protein_rs[n_prot // 2].item() if n_prot > 0 else 0.0

    protein_ss = spearman_r_per_protein(all_targets, all_preds)
    protein_ss = protein_ss[~torch.isnan(protein_ss)].sort(descending=True).values
    top1k_s = protein_ss[:min(1000, n_prot)].mean().item()
    median_s = protein_ss[n_prot // 2].item() if n_prot > 0 else 0.0

    print(f"Top-1000 protein Pearson R: {top1k:.4f}")
    print(f"Median protein Pearson R: {median_r:.4f}")
    print(f"Top-1000 protein Spearman R: {top1k_s:.4f}")
    print(f"Median protein Spearman R: {median_s:.4f}")

    results = {
        "args": vars(args),
        "out_dim": out_dim,
        "best_epoch": ckpt["epoch"],
        "best_val_loss": best_val_loss,
        "best_val_pearson": ckpt["val_pearson"],
        "best_val_spearman": ckpt["val_spearman"],
        "test_loss": test_loss,
        "test_pearson_r": test_r,
        "test_spearman_r": test_s,
        "test_sample_pearson_mean": mean_sample_r,
        "test_top1k_protein_pearson": top1k,
        "test_median_protein_pearson": median_r,
        "test_top1k_protein_spearman": top1k_s,
        "test_median_protein_spearman": median_s,
        "total_params": total_params,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
