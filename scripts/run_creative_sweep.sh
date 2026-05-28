#!/usr/bin/env bash
# Creative architecture sweep: 7 models, sequential
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
echo "Creative architecture sweep at $(date)"
echo "=========================================="

run_exp() {
    local name="$1"
    shift
    local outdir="experiments/${name}"
    if [ -f "${outdir}/results.json" ]; then
        echo "[$(date)] SKIP $name — results.json exists"
        return 0
    fi
    echo "[$(date)] START $name"
    $CONDA_RUN python3 scripts/train_baseline.py \
        --manifest "$MANIFEST" \
        --proteome "$PROTEOME" \
        --split "$SPLIT" \
        --feature_model "$FEATURE" \
        --in_dim 768 \
        "$@" \
        --lr 3e-4 \
        --wd "$WD" \
        --batch_size "$BATCH" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --output_dir "$outdir" \
        --seed 42
    echo "[$(date)] DONE $name"
}

# ── Already implemented, now running ──
run_exp "arch_transformer_h256" \
    --model transformer --hidden_dim 256 --n_layers 2 --n_heads 4 --cls_dims "256,128" \
    --norm ln --residual --dropout 0.3

run_exp "arch_topk_h512_k64" \
    --model topk_attention --hidden_dim 512 --top_k 64 --cls_dims "512,256" \
    --norm bn --dropout 0.3

run_exp "arch_deepsets_h256" \
    --model deepsets --hidden_dim 256 --cls_dims "256,128" \
    --norm ln --residual --dropout 0.3

# ── New creative architectures ──
run_exp "arch_perceiver_lat128_k16" \
    --model perceiver --hidden_dim 128 --n_latents 16 --n_layers 2 --n_heads 4 --cls_dims "128,64" \
    --norm ln --residual --dropout 0.3

run_exp "arch_prototype_h128_k32" \
    --model prototype --hidden_dim 128 --n_prototypes 32 --temperature 1.0 --cls_dims "160,80" \
    --norm ln --residual --dropout 0.3

run_exp "arch_sinkhorn_h128_s16" \
    --model sinkhorn --hidden_dim 128 --n_slots 16 --n_sinkhorn_iters 3 \
    --sinkhorn_epsilon 0.05 --cls_dims "512,256" \
    --norm ln --residual --dropout 0.3

run_exp "arch_moae_h128_e4" \
    --model moae --hidden_dim 128 --n_experts 4 --cls_dims "128,64" \
    --norm ln --residual --dropout 0.3

echo "=========================================="
echo "All experiments complete at $(date)"
echo "=========================================="

echo ""
echo "=== FINAL RESULTS ==="
for f in experiments/arch_*/results.json; do
    name=$(echo "$f" | cut -d/ -f2)
    R=$(python3 -c "
import json
d = json.load(open('$f'))
print(f'{d[\"test_pearson_r\"]:.4f} (val={d[\"best_val_pearson\"]:.4f}, params={d[\"total_params\"]:,}, ep={d[\"best_epoch\"]})')
" 2>/dev/null || echo "FAILED")
    echo "  $name: Test R=$R"
done
