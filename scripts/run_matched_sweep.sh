#!/usr/bin/env bash
# Fair comparison: all models ~10M params, dualpath ref=10.87M
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

echo "=== Matched-size architecture sweep ==="
date

run_exp() {
    local name="$1"; shift
    local outdir="experiments/${name}"
    if [ -f "${outdir}/results.json" ]; then
        echo "SKIP $name — exists"; return 0
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

# 1. PerceiverMIL (h=512, k=16, L=2, heads=8) ~10.97M
run_exp "arch_perceiver_h512" \
    --model perceiver --hidden_dim 512 --n_latents 16 --n_layers 2 --n_heads 8 \
    --cls_dims "512,256" --norm ln --residual --dropout 0.3

# 2. PrototypeMIL (h=512, K=32) ~10.66M  
run_exp "arch_prototype_h512" \
    --model prototype --hidden_dim 512 --n_prototypes 32 \
    --cls_dims "2048,1024" --norm ln --residual --dropout 0.3

# 3. SinkhornMIL (h=192, S=16) ~11.33M
run_exp "arch_sinkhorn_h192" \
    --model sinkhorn --hidden_dim 192 --n_slots 16 --n_sinkhorn_iters 3 \
    --sinkhorn_epsilon 0.05 --cls_dims "1536,768" --norm ln --residual --dropout 0.3

# 4. MoAE (h=512, E=4) ~11.69M
run_exp "arch_moae_h512" \
    --model moae --hidden_dim 512 --n_experts 4 \
    --cls_dims "2048,1024" --norm ln --residual --dropout 0.3

# 5. TransformerMIL (h=512, L=2, heads=8) ~11.27M
run_exp "arch_transformer_h512" \
    --model transformer --hidden_dim 512 --n_layers 2 --n_heads 8 \
    --cls_dims "1024,512" --norm ln --residual --dropout 0.3

# 6. TopK GA (h=1024, k=64) ~13.07M
run_exp "arch_topk_h1024_v2" \
    --model topk_attention --hidden_dim 1024 --top_k 64 \
    --cls_dims "1536,1024" --norm bn --dropout 0.3

# 7. DeepSets (h=512) ~10.84M
run_exp "arch_deepsets_h512" \
    --model deepsets --hidden_dim 512 \
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
