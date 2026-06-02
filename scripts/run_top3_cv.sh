#!/usr/bin/env bash
# 5-fold CV for top 3 configs -- sequential (single GPU)
set -e
BASE="/data/workspace/ai2bio/ai2bio_genesis/WSI2Proteom"
cd "$BASE"
export PYTHONUNBUFFERED=1

CONDA="conda run --no-capture-output -n ai2bio_genesis python3"
MAN="--manifest data/paired_manifest_v2.csv"
PRO="--proteome /data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors_nonan.pkl"
FEAT="--feature_model ctranspath --in_dim 768"
MODEL="--model moae --hidden_dim 512"
HP="--lr 3e-4 --wd 1e-5 --batch_size 32 --epochs 300 --patience 30 --seed 42"

declare -A CFG
CFG["smoothl1_b2.0_aux0.1"]="--loss smoothl1 --smoothl1-beta 2.0 --pearson-aux-weight 0.1"
CFG["huber_d2.0_monR"]="--loss huber --huber-delta 2.0 --monitor val_pearson"
CFG["smoothl1_b2.0"]="--loss smoothl1 --smoothl1-beta 2.0"

FOLDS=5
RESULTS_CSV="experiments/top3_cv_results.csv"
COLLECT="python3 scripts/_collect_result.py"

echo "config,fold,test_pearson_r,best_val_pearson,best_epoch,total_params,test_loss,test_top1k_pearson" > "$RESULTS_CSV"

echo "=== TOP 3 -- 5-FOLD CV at $(date) ===" | tee -a experiments/top3_cv.log

for config in "${!CFG[@]}"; do
    echo "" | tee -a experiments/top3_cv.log
    echo "--- Config: $config ---" | tee -a experiments/top3_cv.log
    for fold in $(seq 1 $FOLDS); do
        SPLIT="data/splits/split_v1_fold${fold}.csv"
        ODIR="experiments/cv_${config}_fold${fold}"
        if [ -f "${ODIR}/results.json" ]; then
            echo "[SKIP] ${config} fold${fold} -- exists" | tee -a experiments/top3_cv.log
        else
            echo "[START] ${config} fold${fold} at $(date)" | tee -a experiments/top3_cv.log
            $CONDA scripts/train_baseline.py $MAN $PRO --split "$SPLIT" $FEAT $MODEL $HP \
                ${CFG[$config]} --output_dir "$ODIR" \
                &> "experiments/cv_${config}_fold${fold}.log"
            echo "[DONE]  ${config} fold${fold} at $(date)" | tee -a experiments/top3_cv.log
        fi

        RJSON="${ODIR}/results.json"
        if [ -f "$RJSON" ]; then
            $COLLECT "$RJSON" "$RESULTS_CSV" "$config" "$fold" | tee -a experiments/top3_cv.log
        fi
    done
done

echo "" | tee -a experiments/top3_cv.log
echo "=== TOP 3 -- 5-FOLD CV SUMMARY ===" | tee -a experiments/top3_cv.log
python3 <<-PYEOF | tee -a experiments/top3_cv.log
import csv, math
results = {}
with open("experiments/top3_cv_results.csv") as f:
    for r in csv.DictReader(f):
        try:
            c = r["config"]
            v = float(r["test_pearson_r"])
            results.setdefault(c, []).append(v)
        except:
            pass
print(f'{"Config":30s} {"mean_R":>8s} {"std":>6s} {"n":>3s}')
print("-" * 50)
for c in sorted(results, key=lambda k: -sum(results[k])/len(results[k])):
    v = results[c]
    mu = sum(v)/len(v)
    s = math.sqrt(sum((x-mu)**2 for x in v)/(len(v)-1)) if len(v)>1 else 0
    print(f"{c:30s} {mu:.4f} ± {s:.4f}  (n={len(v)})")
PYEOF

echo "Done at $(date)" | tee -a experiments/top3_cv.log
