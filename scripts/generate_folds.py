"""Generate k-fold splits with train/val/test for each fold."""
import csv
import random
import argparse
from pathlib import Path

random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/paired_manifest.csv")
    parser.add_argument("--split", default="data/splits/split_v1.csv")
    parser.add_argument("--outdir", default="data/splits")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    cases = []
    with open(args.split) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row["split_group_id"])

    print(f"Total cases: {len(cases)}")

    random.shuffle(cases)
    fold_size = len(cases) // args.n_folds
    folds = []
    for i in range(args.n_folds):
        start = i * fold_size
        end = start + fold_size if i < args.n_folds - 1 else len(cases)
        folds.append(cases[start:end])

    all_cases = set(cases)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_folds):
        test_cases = set(folds[i])
        train_pool = list(all_cases - test_cases)
        random.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * args.val_ratio))
        val_cases = set(train_pool[:n_val])
        train_cases = set(train_pool[n_val:])

        outpath = outdir / f"split_v1_fold{i+1}.csv"
        with open(outpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split_group_id", "split"])
            for c in sorted(train_cases):
                writer.writerow([c, "train"])
            for c in sorted(val_cases):
                writer.writerow([c, "val"])
            for c in sorted(test_cases):
                writer.writerow([c, "test"])

        print(f"  fold{i+1}: train={len(train_cases)} val={len(val_cases)} test={len(test_cases)} -> {outpath.name}")

if __name__ == "__main__":
    main()
