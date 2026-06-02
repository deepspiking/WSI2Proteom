#!/usr/bin/env bash
# Loss function sweep — Wave 1
# Runs all loss configs on a single fold (fold 1) of CTransPath + MoAE h=512
set -e
BASE="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE"
export PYTHONUNBUFFERED=1

CONDA="conda run --no-capture-output -n ai2bio_genesis python3"
MAN="--manifest data/paired_manifest_v2.csv"
PRO="--proteome /data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
SPLIT="data/splits/split_v1_fold1.csv"
FEAT="--feature_model ctranspath --in_dim 768"
MODEL="--model moae --hidden_dim 512"
HP="--lr 3e-4 --wd 1e-5 --batch_size 32 --epochs 300 --patience 30 --seed 42"

# Loss configs to sweep
declare -A LOSS_ARGS
LOSS_ARGS["smoothl1_b1.0"]="--loss smoothl1 --smoothl1-beta 1.0"
LOSS_ARGS["smoothl1_b0.1"]="--loss smoothl1 --smoothl1-beta 0.1"
LOSS_ARGS["smoothl1_b0.5"]="--loss smoothl1 --smoothl1-beta 0.5"
LOSS_ARGS["smoothl1_b2.0"]="--loss smoothl1 --smoothl1-beta 2.0"
LOSS_ARGS["mse"]="--loss mse"
LOSS_ARGS["huber_d0.5"]="--loss huber --huber-delta 0.5"
LOSS_ARGS["huber_d1.0"]="--loss huber --huber-delta 1.0"
LOSS_ARGS["huber_d2.0"]="--loss huber --huber-delta 2.0"

RESULTS_CSV="experiments/loss_sweep_results.csv"
echo "config,fold,test_pearson_r,best_val_pearson,best_epoch,total_params,test_loss,test_top1k_pearson" > "$RESULTS_CSV"

echo "=== LOSS SWEEP at $(date) ===" | tee -a experiments/loss_sweep.log

for config in "${!LOSS_ARGS[@]}"; do
    ODIR="experiments/loss_sweep_${config}_fold1"
    if [ -f "${ODIR}/results.json" ]; then
        echo "[SKIP] ${config} — already exists" | tee -a experiments/loss_sweep.log
    else
        echo "[START] ${config} at $(date)" | tee -a experiments/loss_sweep.log
        $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" $FEAT $MODEL $HP \
            ${LOSS_ARGS[$config]} --output_dir "$ODIR" \
            &> "experiments/loss_sweep_${config}_fold1.log"
        echo "[DONE]  ${config} at $(date)" | tee -a experiments/loss_sweep.log
    fi

    # Collect result
    RJSON="${ODIR}/results.json"
    if [ -f "$RJSON" ]; then
        python3 -c "
import json
d = json.load(open('$RJSON'))
print(f'  test_R={d[\"test_pearson_r\"]:.4f} val_R={d[\"best_val_pearson\"]:.4f} ep={d[\"best_epoch\"]} loss={d[\"test_loss\"]:.4f} top1k={d[\"test_top1k_protein_pearson\"]:.4f}')
open('$RESULTS_CSV','a').write(f'$config,1,{d[\"test_pearson_r\"]:.4f},{d[\"best_val_pearson\"]:.4f},{d[\"best_epoch\"]},{d[\"total_params\"]},{d[\"test_loss\"]:.4f},{d[\"test_top1k_protein_pearson\"]:.4f}\n')
" 2>/dev/null
    fi
done

echo "" | tee -a experiments/loss_sweep.log
echo "=== LOSS SWEEP SUMMARY ===" | tee -a experiments/loss_sweep.log
python3 -c "
import csv
rows = []
with open('$RESULTS_CSV') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
rows.sort(key=lambda r: -float(r['test_pearson_r']))
print(f'{\"Config\":25s} {\"Fold\":4s} {\"test_R\":>8s} {\"val_R\":>8s} {\"top1k\":>7s} {\"epoch\":>5s}')
print('-' * 60)
for r in rows:
    print(f'{r[\"config\"]:25s} {r[\"fold\"]:4s} {r[\"test_pearson_r\"]:>8s} {r[\"best_val_pearson\"]:>8s} {r[\"test_top1k_pearson\"]:>7s} {r[\"best_epoch\"]:>5s}')
" 2>/dev/null | tee -a experiments/loss_sweep.log

echo "Done at $(date)" | tee -a experiments/loss_sweep.log
