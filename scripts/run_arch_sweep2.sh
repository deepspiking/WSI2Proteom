#!/usr/bin/env bash
# Sequential experiment runner for architecture sweep (removed set -e for robustness)
BASE_DIR="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE_DIR"

CONDA_ENV="ai2bio_genesis"
export PYTHONUNBUFFERED=1
CONDA_RUN="conda run --no-capture-output -n ${CONDA_ENV}"

PROTEOME="/data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
MANIFEST="data/paired_manifest.csv"
SPLIT="data/splits/split_v1.csv"
FEATURE="ctranspath"
BATCH=32
EPOCHS=200
PATIENCE=30
WD=1e-5

echo "=========================================="
echo "Resuming architecture sweep at $(date)"
echo "=========================================="

# === Exp B: Dual-Path MIL ===
OUTDIR="experiments/arch_dualpath_h1024"
echo "[$(date)] Exp B: Dual-Path MIL (attention+mean, h=1024, ln+residual)"
$CONDA_RUN python3 scripts/train_baseline.py \
  --manifest "$MANIFEST" \
  --proteome "$PROTEOME" \
  --split "$SPLIT" \
  --feature_model "$FEATURE" \
  --in_dim 768 \
  --model dualpath \
  --hidden_dim 1024 \
  --cls_dims "1024,1024,512" \
  --dropout 0.3 \
  --norm ln \
  --residual \
  --lr 3e-4 \
  --wd "$WD" \
  --batch_size "$BATCH" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --output_dir "$OUTDIR" \
  --seed 42
echo "[$(date)] Exp B complete"

# === Exp C: Low-Rank Decoder GatedAttention ===
OUTDIR="experiments/arch_lowrank_attn_latent64"
echo "[$(date)] Exp C: Low-Rank Decoder GatedAttention (h=1024, latent=64, skip)"
$CONDA_RUN python3 scripts/train_baseline.py \
  --manifest "$MANIFEST" \
  --proteome "$PROTEOME" \
  --split "$SPLIT" \
  --feature_model "$FEATURE" \
  --in_dim 768 \
  --model lowrank_attention \
  --hidden_dim 1024 \
  --latent_dim 64 \
  --use_skip \
  --dropout 0.3 \
  --lr 3e-4 \
  --wd "$WD" \
  --batch_size "$BATCH" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --output_dir "$OUTDIR" \
  --seed 42
echo "[$(date)] Exp C complete"

echo "=========================================="
echo "All experiments complete at $(date)"
echo "=========================================="

echo ""
echo "=== FINAL RESULTS ==="
for f in experiments/arch_*/results.json; do
  name=$(echo "$f" | cut -d/ -f2)
  R=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d[\"test_pearson_r\"]:.4f}')" 2>/dev/null || echo "FAIL")
  echo "  $name: Test R=$R"
done
