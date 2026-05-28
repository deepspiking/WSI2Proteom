#!/usr/bin/env bash
# Autonomous experiment orchestrator — runs 3 in parallel in batches
BASE_DIR="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE_DIR" || exit 1
export PYTHONUNBUFFERED=1

CONDA_PY=$(conda run --no-capture-output -n ai2bio_genesis which python3 2>/dev/null)
echo "Using: $CONDA_PY"
echo "Start: $(date)" | tee experiments/auto_orch.log
echo "" | tee -a experiments/auto_orch.log

PROTEOME="/data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
MANIFEST="data/paired_manifest.csv"
SPLIT="data/splits/split_v1.csv"
FEATURE="ctranspath"
BATCH=32
EPOCHS=300
PATIENCE=20
WD=1e-5

# Each experiment = 2 array entries: name, then args
# Batch 1: Scale champions
EXPS=(
auto_moae_h1024_e6
"--model moae --hidden_dim 1024 --n_experts 6 --cls_dims 2048,1024 --dropout 0.4 --norm ln --residual"
auto_dualpath_h2048
"--model dualpath --hidden_dim 2048 --cls_dims 4096,2048,1024 --dropout 0.4 --norm ln --residual"
auto_moae_h768_e8
"--model moae --hidden_dim 768 --n_experts 8 --cls_dims 3072,1536,768 --dropout 0.4 --norm ln --residual"
# Batch 2: Deep representation + hybrids
auto_moae_h512_deep
"--model moae --hidden_dim 512 --n_experts 4 --cls_dims 4096,2048,1024 --dropout 0.5 --norm ln --residual"
auto_dualmoe_h1024
"--model dualmoe --hidden_dim 1024 --n_experts 4 --cls_dims 2048,1024,512 --dropout 0.4 --norm ln --residual"
auto_crossattn_h768
"--model crossattn_fusion --hidden_dim 768 --n_heads 8 --cls_dims 2048,1024,512 --dropout 0.4 --norm ln --residual"
# Batch 3: Regularized variants
auto_moae_h1024_dp05
"--model moae --hidden_dim 1024 --n_experts 6 --cls_dims 2048,1024 --dropout 0.5 --norm ln --residual"
auto_dualpath_h1536
"--model dualpath --hidden_dim 1536 --cls_dims 3072,1536,768 --dropout 0.4 --norm ln --residual"
auto_moae_h768_dp05
"--model moae --hidden_dim 768 --n_experts 8 --cls_dims 3072,1536,768 --dropout 0.5 --norm ln --residual"
# Batch 4: Architectural diversity
auto_dualpath_h1024_deep
"--model dualpath --hidden_dim 1024 --cls_dims 4096,2048,1024,512 --dropout 0.4 --norm ln --residual"
auto_sparse_top2_h1024
"--model sparse_top2 --hidden_dim 1024 --n_experts 8 --top_k 2 --cls_dims 2048,1024 --dropout 0.4 --norm ln --residual"
auto_moae_h1024_e8
"--model moae --hidden_dim 1024 --n_experts 8 --cls_dims 2048,1024,512 --dropout 0.4 --norm ln --residual"
# Batch 5: Output & scale
auto_lowrank_h1024
"--model lowrank_attention --hidden_dim 1024 --latent_dim 256 --use_skip --dropout 0.4"
auto_dualpath_h1792
"--model dualpath --hidden_dim 1792 --cls_dims 3584,1792,896 --dropout 0.4 --norm ln --residual"
auto_moae_h1024_clsdeep
"--model moae --hidden_dim 1024 --n_experts 4 --cls_dims 4096,2048,1024 --dropout 0.4 --norm ln --residual"
)

N_EXPS=${#EXPS[@]}
echo "Total: $((N_EXPS / 2)) experiments" | tee -a experiments/auto_orch.log
echo "" | tee -a experiments/auto_orch.log

# Batch settings: 3 experiments per batch = 6 array entries
BATCH_STEP=6
for ((i=0; i<N_EXPS; i+=BATCH_STEP)); do
    batch_num=$((i / 6 + 1))
    echo "========== BATCH $batch_num at $(date) ==========" | tee -a experiments/auto_orch.log

    PIDS=(); NAMES=(); OUTDIRS=()

    for ((j=0; j<BATCH_STEP; j+=2)); do
        idx=$((i + j))
        name="${EXPS[$idx]}"
        args="${EXPS[$idx+1]}"
        outdir="experiments/$name"
        
        if [ -f "${outdir}/results.json" ]; then
            echo "  SKIP $name" | tee -a experiments/auto_orch.log
            continue
        fi
        
        echo "  START $name" | tee -a experiments/auto_orch.log
        
        ($CONDA_PY scripts/train_baseline.py \
            --manifest "$MANIFEST" --proteome "$PROTEOME" --split "$SPLIT" \
            --feature_model "$FEATURE" --in_dim 768 \
            $args \
            --lr 3e-4 --wd "$WD" --batch_size "$BATCH" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --output_dir "$outdir" --seed 42 \
            &> "experiments/${name}.log") &
        
        PIDS+=($!); NAMES+=("$name"); OUTDIRS+=("$outdir")
    done

    if [ ${#PIDS[@]} -eq 0 ]; then
        echo "  All skipped" | tee -a experiments/auto_orch.log
        continue
    fi

    # Monitor every 60s
    while true; do
        running=0; running_names=""
        for i in "${!PIDS[@]}"; do
            kill -0 "${PIDS[$i]}" 2>/dev/null && running=$((running + 1)) && running_names+=" ${NAMES[$i]}"
        done
        [ $running -eq 0 ] && break
        gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        echo "  [$(date)] ${running} running:${running_names}, GPU=${gpu}MiB" | tee -a experiments/auto_orch.log
        sleep 60
    done

    echo "  Batch $batch_num complete at $(date)" | tee -a experiments/auto_orch.log
    for od in "${OUTDIRS[@]}"; do
        if [ -f "${od}/results.json" ]; then
            R=$($CONDA_PY -c "import json; d=json.load(open('${od}/results.json')); print(f'{d[\"test_pearson_r\"]:.4f}')" 2>/dev/null)
            echo "  RESULT $(basename $od): R=$R" | tee -a experiments/auto_orch.log
        fi
    done
done

# Final report
echo "" | tee -a experiments/auto_orch.log
echo "==========================================" | tee -a experiments/auto_orch.log
echo "ALL DONE at $(date)" | tee -a experiments/auto_orch.log
echo "==========================================" | tee -a experiments/auto_orch.log
echo "" | tee -a experiments/auto_orch.log
echo "LEADERBOARD" | tee -a experiments/auto_orch.log
echo "----------" | tee -a experiments/auto_orch.log
for f in experiments/auto_*/results.json; do
    [ -f "$f" ] || continue
    name=$(basename "$(dirname "$f")")
    $CONDA_PY -c "
import json; d=json.load(open('$f'))
print(f'{d[\"test_pearson_r\"]:.4f} | {name:35s} | v={d[\"best_val_pearson\"]:.4f} p={d[\"total_params\"]:,} ep={d[\"best_epoch\"]}')
" 2>/dev/null
done | sort -rn | tee -a experiments/auto_orch.log
