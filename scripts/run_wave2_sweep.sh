#!/usr/bin/env bash
# Wave 2: Monitor fix campaign — train with --monitor val_pearson
# Runs top 3 loss configs from Wave 1 on fold 1
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

declare -A LOSS_ARGS
LOSS_ARGS["smoothl1_b2.0"]="--loss smoothl1 --smoothl1-beta 2.0"
LOSS_ARGS["mse"]="--loss mse"
LOSS_ARGS["huber_d2.0"]="--loss huber --huber-delta 2.0"

RESULTS_CSV="experiments/wave2_results.csv"
echo "config,fold,test_pearson_r,best_val_pearson,best_epoch,total_params,test_loss,test_top1k_pearson,monitor" > "$RESULTS_CSV"

echo "=== WAVE 2 (monitor=val_pearson) at $(date) ===" | tee -a experiments/wave2.log

for config in "${!LOSS_ARGS[@]}"; do
    ODIR="experiments/wave2_${config}_fold1"
    if [ -f "${ODIR}/results.json" ]; then
        echo "[SKIP] ${config} — already exists" | tee -a experiments/wave2.log
    else
        echo "[START] ${config} at $(date)" | tee -a experiments/wave2.log
        $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" $FEAT $MODEL $HP \
            ${LOSS_ARGS[$config]} --monitor val_pearson --output_dir "$ODIR" \
            &> "experiments/wave2_${config}_fold1.log"
        echo "[DONE]  ${config} at $(date)" | tee -a experiments/wave2.log
    fi

    RJSON="${ODIR}/results.json"
    if [ -f "$RJSON" ]; then
        python3 -c "
import json
d = json.load(open('$RJSON'))
t = d['test_pearson_r']
print(f'  test_R={t:.4f} val_R={d[\"best_val_pearson\"]:.4f} ep={d[\"best_epoch\"]} loss={d[\"test_loss\"]:.4f} top1k={d[\"test_top1k_protein_pearson\"]:.4f}')
open('$RESULTS_CSV','a').write(f'$config,1,{t:.4f},{d[\"best_val_pearson\"]:.4f},{d[\"best_epoch\"]},{d[\"total_params\"]},{d[\"test_loss\"]:.4f},{d[\"test_top1k_protein_pearson\"]:.4f},val_pearson\n')
" 2>/dev/null
    fi
done

echo "" | tee -a experiments/wave2.log
echo "=== WAVE 2 SUMMARY (with --monitor val_pearson) ===" | tee -a experiments/wave2.log
echo "For comparison, Wave 1 (monitor=val_loss) results on fold 1:" | tee -a experiments/wave2.log
python3 -c "
import csv
rows = []
with open('experiments/loss_sweep_results.csv') as f:
    for r in csv.DictReader(f):
        if r['config'] in ['smoothl1_b2.0','mse','huber_d2.0']:
            rows.append(r)
print(f'{\"Config\":20s} {\"Wave1(monitor=loss)\":20s} {\"Wave2(monitor=R)\":20s} {\"Delta\":>8s}')
print('-' * 70)
for r in sorted(rows, key=lambda x: -float(x['test_pearson_r'])):
    cfg = r['config']
    w1_r = float(r['test_pearson_r'])
    w2_r = None
    with open('experiments/wave2_results.csv') as f2:
        for r2 in csv.DictReader(f2):
            if r2['config'] == cfg:
                w2_r = float(r2['test_pearson_r'])
    if w2_r is not None:
        delta = w2_r - w1_r
        print(f'{cfg:20s} {w1_r:>8.4f} (val_loss)  {w2_r:>8.4f} (val_R)   {delta:>+8.4f}')
    else:
        print(f'{cfg:20s} {w1_r:>8.4f} (val_loss)  {\"pending\":>20s}')
" 2>/dev/null | tee -a experiments/wave2.log

echo "Done at $(date)" | tee -a experiments/wave2.log
