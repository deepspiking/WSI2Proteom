#!/usr/bin/env bash
# Wave 3: Pearson-based losses — pure Pearson + auxiliary combos
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

declare -A CFG
CFG["pearson"]="--loss pearson"
CFG["smoothl1_b2.0_aux0.1"]="--loss smoothl1 --smoothl1-beta 2.0 --pearson-aux-weight 0.1"
CFG["smoothl1_b2.0_aux0.5"]="--loss smoothl1 --smoothl1-beta 2.0 --pearson-aux-weight 0.5"
CFG["smoothl1_b2.0_aux1.0"]="--loss smoothl1 --smoothl1-beta 2.0 --pearson-aux-weight 1.0"
CFG["huber_d2.0_aux0.1"]="--loss huber --huber-delta 2.0 --pearson-aux-weight 0.1"
CFG["mse_aux0.1"]="--loss mse --pearson-aux-weight 0.1"

RESULTS_CSV="experiments/wave3_results.csv"
echo "config,fold,test_pearson_r,best_val_pearson,best_epoch,total_params,test_loss,test_top1k_pearson" > "$RESULTS_CSV"

echo "=== WAVE 3 (Pearson-based losses) at $(date) ===" | tee -a experiments/wave3.log

for config in "${!CFG[@]}"; do
    ODIR="experiments/wave3_${config}_fold1"
    if [ -f "${ODIR}/results.json" ]; then
        echo "[SKIP] ${config} — already exists" | tee -a experiments/wave3.log
    else
        echo "[START] ${config} at $(date)" | tee -a experiments/wave3.log
        $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" $FEAT $MODEL $HP \
            ${CFG[$config]} --output_dir "$ODIR" \
            &> "experiments/wave3_${config}_fold1.log"
        echo "[DONE]  ${config} at $(date)" | tee -a experiments/wave3.log
    fi

    RJSON="${ODIR}/results.json"
    if [ -f "$RJSON" ]; then
        python3 -c "
import json
d = json.load(open('$RJSON'))
t = d['test_pearson_r']
print(f'  test_R={t:.4f} val_R={d[\"best_val_pearson\"]:.4f} ep={d[\"best_epoch\"]}')
open('$RESULTS_CSV','a').write(f'$config,1,{t:.4f},{d[\"best_val_pearson\"]:.4f},{d[\"best_epoch\"]},{d[\"total_params\"]},{d[\"test_loss\"]:.4f},{d[\"test_top1k_protein_pearson\"]:.4f}\n')
" 2>/dev/null
    fi
done

echo "" | tee -a experiments/wave3.log
echo "=== WAVE 3 SUMMARY ===" | tee -a experiments/wave3.log
python3 -c "
import csv
rows = []
for fn in ['experiments/loss_sweep_results.csv','experiments/wave2_results.csv','experiments/wave3_results.csv']:
    try:
        with open(fn) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except: pass
rows.sort(key=lambda r: -float(r['test_pearson_r']))
print(f'{\"Config\":35s} {\"test_R\":>8s} {\"val_R\":>8s} {\"ep\":>4s}')
print('-' * 60)
for r in rows[:15]:
    print(f'{r[\"config\"]:35s} {r[\"test_pearson_r\"]:>8s} {r[\"best_val_pearson\"]:>8s} {r[\"best_epoch\"]:>4s}')
" 2>/dev/null | tee -a experiments/wave3.log

echo "Done at $(date)" | tee -a experiments/wave3.log
