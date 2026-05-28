#!/usr/bin/env bash
# 5-fold cross-validation for top 3 models — runs 3 at a time
set -e
BASE="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE"
export PYTHONUNBUFFERED=1

CONDA="conda run --no-capture-output -n ai2bio_genesis python3"
MAN="--manifest data/paired_manifest.csv"
PRO="--proteome /data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
FEAT="--feature_model ctranspath --in_dim 768"
HP="--lr 3e-4 --wd 1e-5 --batch_size 32 --epochs 300 --patience 20 --seed 42"

# Top 3 models (index-aligned with MODEL_ARGS)
MODELS=(moae_h512 dualpath_h1536 moae_h1024_dp05)
MODEL_ARGS=(
  "--model moae --hidden_dim 512"
  "--model dualpath --hidden_dim 1536 --cls_dims 3072,1536,768 --dropout 0.4 --norm ln --residual"
  "--model moae --hidden_dim 1024 --n_experts 6 --cls_dims 2048,1024 --dropout 0.5 --norm ln --residual"
)

FOLDS=5
RESULTS_CSV="experiments/cv_results.csv"
echo "model,fold,test_pearson_r,best_val_pearson,total_params,best_epoch,top1k_pearson" > "$RESULTS_CSV"

echo "=== 5-FOLD CV at $(date) ===" | tee -a experiments/cv.log

# Collect all (model_index, fold) pairs that need running
declare -a RUN_QUEUE=()
for ((m=0; m<${#MODELS[@]}; m++)); do
  for fold in $(seq 1 $FOLDS); do
    ODIR="experiments/cv_${MODELS[$m]}_fold${fold}"
    [ ! -f "${ODIR}/results.json" ] && RUN_QUEUE+=("$m:$fold")
  done
done

TOTAL=${#RUN_QUEUE[@]}
echo "Remaining runs: $TOTAL" | tee -a experiments/cv.log

# Process in batches of 3
BATCH=3
for ((b=0; b<TOTAL; b+=BATCH)); do
  echo "--- Batch $((b / BATCH + 1)) at $(date) ---" | tee -a experiments/cv.log
  BGPIDS=()
  for ((j=0; j<BATCH && b+j<TOTAL; j++)); do
    IFS=':' read -r m fold <<< "${RUN_QUEUE[$((b+j))]}"
    SPLIT="data/splits/split_v1_fold${fold}.csv"
    ODIR="experiments/cv_${MODELS[$m]}_fold${fold}"
    echo "[START] ${MODELS[$m]} fold${fold}" | tee -a experiments/cv.log
    $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" $FEAT $HP \
      ${MODEL_ARGS[$m]} --output_dir "$ODIR" \
      &> "experiments/cv_${MODELS[$m]}_fold${fold}.log" &
    BGPIDS+=($!)
  done
  for pid in "${BGPIDS[@]}"; do wait "$pid"; done
  echo "Batch done at $(date)" | tee -a experiments/cv.log
done

echo "All training done at $(date)" | tee -a experiments/cv.log

# Collect results into CSV and print summary
echo "" >> "$RESULTS_CSV"
for ((m=0; m<${#MODELS[@]}; m++)); do
  echo "--- ${MODELS[$m]} ---" | tee -a experiments/cv.log
  for fold in $(seq 1 $FOLDS); do
    RJSON="experiments/cv_${MODELS[$m]}_fold${fold}/results.json"
    if [ -f "$RJSON" ]; then
      python3 -c "import json; d=json.load(open('$RJSON')); print(f'  Fold {fold}: R={d[\"test_pearson_r\"]:.4f} val_R={d[\"best_val_pearson\"]:.4f} p={d[\"total_params\"]:,} ep={d[\"best_epoch\"]} top1k={d[\"test_top1k_protein_pearson\"]:.4f}')" 2>/dev/null
      python3 -c "import json; d=json.load(open('$RJSON')); open('$RESULTS_CSV','a').write(f'${MODELS[$m]},$fold,{d[\"test_pearson_r\"]:.4f},{d[\"best_val_pearson\"]:.4f},{d[\"total_params\"]},{d[\"best_epoch\"]},{d[\"test_top1k_protein_pearson\"]:.4f}\n')" 2>/dev/null
    else
      echo "  Fold $fold: NO RESULT"
    fi
  done
done

echo "" | tee -a experiments/cv.log
echo "=== CV SUMMARY (mean ± std) ===" | tee -a experiments/cv.log
python3 -c "
import csv, math
results = {}
with open('$RESULTS_CSV') as f:
    for row in csv.DictReader(f):
        if row['fold'] == str(row['fold']):  # skip non-numeric fold rows
            try:
                m = row['model']; v = float(row['test_pearson_r'])
                results.setdefault(m, []).append(v)
            except: pass
for m in sorted(results, key=lambda k: -sum(results[k])/len(results[k])):
    v = results[m]; mu = sum(v)/len(v); s = math.sqrt(sum((x-mu)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0
    print(f'  {m:25s} R={mu:.4f} ± {s:.4f} (n={len(v)})')
" 2>/dev/null | tee -a experiments/cv.log
