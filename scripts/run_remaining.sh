#!/usr/bin/env bash
# Direct batch runner — explicitly list each experiment, no complex indexing
set -e

BASE="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE"
export PYTHONUNBUFFERED=1

CONDA="conda run --no-capture-output -n ai2bio_genesis python3"
MAN="--manifest data/paired_manifest.csv"
PRO="--proteome /data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
SPL="--split data/splits/split_v1.csv"
BASE_ARGS="$MAN $PRO $SPL --feature_model ctranspath --in_dim 768 --lr 3e-4 --wd 1e-5 --batch_size 32 --epochs 300 --patience 20 --seed 42"

PIDS=()
run_exp() {
    local name="$1"
    local args="$2"
    local outdir="experiments/$name"
    if [ -f "${outdir}/results.json" ]; then
        echo "[SKIP] $name — already done" | tee -a experiments/orch_direct.log
        PIDS+=("skip")
        return
    fi
    echo "[START] $name at $(date)" | tee -a experiments/orch_direct.log
    $CONDA scripts/train_baseline.py $BASE_ARGS $args --output_dir "$outdir" &> "experiments/${name}.log" &
    local pid=$!
    echo "[PID] $name = $pid" | tee -a experiments/orch_direct.log
    PIDS+=("$pid")
}

wait_batch() {
    local batch=$1
    local n_total=0 n_running=0
    for p in "${PIDS[@]}"; do [ "$p" != "skip" ] && n_total=$((n_total + 1)); done
    echo "[BATCH $batch] Waiting for $n_total experiment(s) at $(date)" | tee -a experiments/orch_direct.log
    while true; do
        n_running=0
        for p in "${PIDS[@]}"; do
            [ "$p" != "skip" ] && kill -0 "$p" 2>/dev/null && n_running=$((n_running + 1))
        done
        [ $n_running -eq 0 ] && break
        gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        echo "  [$(date)] ${n_running} running, GPU=${gpu}MiB" | tee -a experiments/orch_direct.log
        sleep 60
    done
    echo "[BATCH $batch] Complete at $(date)" | tee -a experiments/orch_direct.log
    echo "" | tee -a experiments/orch_direct.log
    PIDS=()
}

echo "=== DIRECT BATCH RUNNER at $(date) ===" | tee experiments/orch_direct.log
echo "" | tee -a experiments/orch_direct.log

# --- BATCH 3: Regularized variants ---
echo "--- BATCH 3: Regularized variants ---" | tee -a experiments/orch_direct.log
run_exp auto_moae_h1024_dp05 "--model moae --hidden_dim 1024 --n_experts 6 --cls_dims 2048,1024 --dropout 0.5 --norm ln --residual"
run_exp auto_dualpath_h1536   "--model dualpath --hidden_dim 1536 --cls_dims 3072,1536,768 --dropout 0.4 --norm ln --residual"
run_exp auto_moae_h768_dp05 "--model moae --hidden_dim 768 --n_experts 8 --cls_dims 3072,1536,768 --dropout 0.5 --norm ln --residual"
wait_batch 3

echo "--- BATCH 4: Architectural diversity ---" | tee -a experiments/orch_direct.log
run_exp auto_dualpath_h1024_deep "--model dualpath --hidden_dim 1024 --cls_dims 4096,2048,1024,512 --dropout 0.4 --norm ln --residual"
run_exp auto_sparse_top2_h1024   "--model sparse_top2 --hidden_dim 1024 --n_experts 8 --top_k 2 --cls_dims 2048,1024 --dropout 0.4 --norm ln --residual"
run_exp auto_moae_h1024_e8       "--model moae --hidden_dim 1024 --n_experts 8 --cls_dims 2048,1024,512 --dropout 0.4 --norm ln --residual"
wait_batch 4

echo "--- BATCH 5: Output & scale ---" | tee -a experiments/orch_direct.log
run_exp auto_lowrank_h1024       "--model lowrank_attention --hidden_dim 1024 --latent_dim 256 --use_skip --dropout 0.4"
run_exp auto_dualpath_h1792      "--model dualpath --hidden_dim 1792 --cls_dims 3584,1792,896 --dropout 0.4 --norm ln --residual"
run_exp auto_moae_h1024_clsdeep  "--model moae --hidden_dim 1024 --n_experts 4 --cls_dims 4096,2048,1024 --dropout 0.4 --norm ln --residual"
wait_batch 5

# --- FINAL LEADERBOARD ---
echo "" | tee -a experiments/orch_direct.log
echo "==========================================" | tee -a experiments/orch_direct.log
echo "ALL DONE at $(date)" | tee -a experiments/orch_direct.log
echo "==========================================" | tee -a experiments/orch_direct.log
echo "" | tee -a experiments/orch_direct.log
echo "LEADERBOARD" | tee -a experiments/orch_direct.log
echo "----------" | tee -a experiments/orch_direct.log
python3 -c "
import json, pathlib
for d in sorted(pathlib.Path('experiments').glob('auto_*/results.json')):
    r = json.loads(d.read_text())
    n = d.parent.name
    print(f'{r[\"test_pearson_r\"]:.4f} | {n:35s} | p={r[\"total_params\"]:>8,} ep={r[\"best_epoch\"]:2d} val_R={r[\"best_val_pearson\"]:.4f}')
" | sort -rn | tee -a experiments/orch_direct.log
