#!/usr/bin/env bash
# Fusion architecture sweep: 4 new models, ~10M each, sequential
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

echo "=== Fusion architecture sweep ==="
date

run_exp() {
    local name="$1"; shift
    local outdir="experiments/${name}"
    if [ -f "${outdir}/results.json" ]; then
        echo "SKIP $name"; return 0
    fi
    echo "START $name"
    $CONDA_RUN python3 scripts/train_baseline.py \
        --manifest "$MANIFEST" --proteome "$PROTEOME" --split "$SPLIT" \
        --feature_model "$FEATURE" --in_dim 768 "$@" \
        --lr 3e-4 --wd "$WD" --batch_size "$BATCH" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --output_dir "$outdir" --seed 42
    echo "DONE $name"
}

# 1. DualMoE (MoAE attn + Dual-Path mean) ~9.76M
run_exp "arch_dualmoe_h512" \
    --model dualmoe --hidden_dim 512 --n_experts 4 \
    --cls_dims "2048,1024,512" --norm ln --residual --dropout 0.3

# 2. CrossAttnFusion (Dual encoder fusion) ~8.45M
run_exp "arch_crossattn_h512" \
    --model crossattn_fusion --hidden_dim 512 --n_heads 4 \
    --cls_dims "1024,512" --norm ln --residual --dropout 0.3

# 3. ProgressiveExpert (Boosting-style) ~11.63M
run_exp "arch_progressive_h512" \
    --model progressive --hidden_dim 512 --n_experts 4 \
    --cls_dims "2048,1024" --norm ln --residual --dropout 0.3

# 4. SparseTop2MoAE (Token-choice routing) ~11.70M
run_exp "arch_sparse_top2_h512" \
    --model sparse_top2 --hidden_dim 512 --n_experts 4 --top_k 2 \
    --cls_dims "2048,1024" --norm ln --residual --dropout 0.3

echo ""
echo "=== ALL DONE ==="
date
echo ""
for f in experiments/arch_*/results.json; do
    name=$(echo "$f" | cut -d/ -f2)
    R=$(python3 -c "
import json; d=json.load(open('$f'))
print(f'{d[\"test_pearson_r\"]:.4f} (val={d[\"best_val_pearson\"]:.4f}, params={d[\"total_params\"]:,})')
" 2>/dev/null || echo "FAIL")
    echo "  $name: R=$R"
done
