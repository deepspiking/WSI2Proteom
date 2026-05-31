#!/usr/bin/env bash
# 5-fold CV for all feature models — 3 model configs × 5 folds per model
set -o pipefail  # fail on pipe errors but don't exit on individual command failures
BASE="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA="conda run --no-capture-output -n ai2bio_genesis python3"
MAN="--manifest data/paired_manifest_v2.csv"
PRO="--proteome /data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
HP="--lr 3e-4 --wd 1e-5 --batch_size 32 --epochs 300 --patience 20 --seed 42"

# Feature models with their dimensions
declare -A FEAT_DIMS=(
  [ctranspath]=768
  [exaone2]=768
  [uni2-h]=1536
  [conch]=512
  [retccl]=2048
)

# Model configs (index-aligned)
MODEL_LABELS=(moae_h512 dualpath_h1536 moae_h1024_dp05)
MODEL_ARGS=(
  "--model moae --hidden_dim 512"
  "--model dualpath --hidden_dim 1536 --cls_dims 3072,1536,768 --dropout 0.4 --norm ln --residual"
  "--model moae --hidden_dim 1024 --n_experts 6 --cls_dims 2048,1024 --dropout 0.5 --norm ln --residual"
)

FOLDS=5
RESULTS_CSV="experiments/cv_all_results.csv"
echo "model,feature_model,fold,test_pearson_r,best_val_pearson,test_spearman_r,best_epoch,total_params,top1k_pearson" > "$RESULTS_CSV"

BATCH=1  # 1 job at a time — dualpath_h1536 uses ~16GB, 2× exceeds 32GB VRAM
START_MODEL="${1:-ctranspath}"  # Optional: start from specific model

echo "============================================"
echo "  Full 5-fold CV across all feature models"
echo "  Start time: $(date)"
echo "  Start from: ${START_MODEL}"
echo "============================================"

STARTED=0
for FEAT_MODEL in ctranspath exaone2 uni2-h conch retccl; do
  IN_DIM=${FEAT_DIMS[$FEAT_MODEL]}
  echo ""
  echo "========== ${FEAT_MODEL} (${IN_DIM}-dim) =========="

  if [ "$STARTED" -eq 0 ] && [ "$FEAT_MODEL" != "$START_MODEL" ]; then
    echo "  Skipping (not yet at ${START_MODEL})"
    continue
  fi
  STARTED=1

  # Build queue of (model_idx, fold) pairs
  declare -a RUN_QUEUE=()
  for ((m=0; m<${#MODEL_LABELS[@]}; m++)); do
    for fold in $(seq 1 $FOLDS); do
      ODIR="experiments/cv_${FEAT_MODEL}_${MODEL_LABELS[$m]}_fold${fold}"
      if [ ! -f "${ODIR}/results.json" ]; then
        RUN_QUEUE+=("$m:$fold")
      fi
    done
  done

  TOTAL=${#RUN_QUEUE[@]}
  echo "  Remaining runs: ${TOTAL}"

  if [ "$TOTAL" -eq 0 ]; then
    echo "  All runs completed, skipping."
    continue
  fi

  # Process in batches
  for ((b=0; b<TOTAL; b+=BATCH)); do
    echo "  --- Batch $((b / BATCH + 1)) at $(date) ---"
    BGPIDS=()
    for ((j=0; j<BATCH && b+j<TOTAL; j++)); do
      IFS=':' read -r m fold <<< "${RUN_QUEUE[$((b+j))]}"
      M_LABEL=${MODEL_LABELS[$m]}
      M_ARGS=${MODEL_ARGS[$m]}
      SPLIT="data/splits/split_v1_fold${fold}.csv"
      ODIR="experiments/cv_${FEAT_MODEL}_${M_LABEL}_fold${fold}"
      LOG="experiments/cv_${FEAT_MODEL}_${M_LABEL}_fold${fold}.log"
      echo "    [START] ${FEAT_MODEL}/${M_LABEL} fold${fold} -> ${LOG}"
      $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" \
        --feature_model "$FEAT_MODEL" --in_dim "$IN_DIM" \
        $HP ${M_ARGS} --output_dir "$ODIR" \
        &> "$LOG" &
      BGPIDS+=($!)
    done
    for pid in "${BGPIDS[@]}"; do wait "$pid"; done
    echo "  Batch done at $(date)"
  done

  # Collect results for this model
  for ((m=0; m<${#MODEL_LABELS[@]}; m++)); do
    M_LABEL=${MODEL_LABELS[$m]}
    echo "  --- ${FEAT_MODEL}/${M_LABEL} ---"
    for fold in $(seq 1 $FOLDS); do
      RJSON="experiments/cv_${FEAT_MODEL}_${M_LABEL}_fold${fold}/results.json"
      if [ -f "$RJSON" ]; then
        $CONDA -c "
import json
d=json.load(open('$RJSON'))
print(f'    Fold $fold: R={d[\"test_pearson_r\"]:.4f} val_R={d[\"best_val_pearson\"]:.4f} '
      f'p={d[\"total_params\"]:,} ep={d[\"best_epoch\"]} top1k={d.get(\"test_top1k_protein_pearson\",0):.4f}')
" 2>/dev/null || echo "    Fold $fold: parse error"
        # write to CSV (use python within conda to ensure json module available)
        $CONDA -c "
import json
d=json.load(open('$RJSON'))
open('$RESULTS_CSV','a').write(f'${M_LABEL},${FEAT_MODEL},$fold,{d[\"test_pearson_r\"]:.4f},{d[\"best_val_pearson\"]:.4f},{d[\"test_spearman_r\"]:.4f},{d[\"best_epoch\"]},{d[\"total_params\"]},{d.get(\"test_top1k_protein_pearson\",0):.4f}\n')
" 2>/dev/null || true
      else
        echo "    Fold $fold: NO RESULT"
      fi
    done
  done
done

echo ""
echo "============================================"
echo "  All training done at $(date)"
echo "============================================"

# Print final summary
echo ""
echo "=== CV SUMMARY (mean ± std) ==="
$CONDA -c "
import csv, math
results = {}
with open('$RESULTS_CSV') as f:
    for row in csv.DictReader(f):
        try:
            key = f\"{row['feature_model']}/{row['model']}\"
            v = float(row['test_pearson_r'])
            results.setdefault(key, []).append(v)
        except: pass
for key in sorted(results, key=lambda k: -sum(results[k])/len(results[k])):
    v = results[key]; mu = sum(v)/len(v); s = math.sqrt(sum((x-mu)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0
    print(f'  {key:30s} R={mu:.4f} ± {s:.4f} (n={len(v)})')
" 2>/dev/null
