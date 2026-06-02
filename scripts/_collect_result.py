import json, sys

rjson = sys.argv[1]
csv_path = sys.argv[2]
config = sys.argv[3]
fold = sys.argv[4]

d = json.load(open(rjson))
t = d["test_pearson_r"]
line = f'{config},{fold},{t:.4f},{d["best_val_pearson"]:.4f},{d["best_epoch"]},{d["total_params"]},{d["test_loss"]:.4f},{d["test_top1k_protein_pearson"]:.4f}\n'
print(f'  Fold {fold}: test_R={t:.4f} val_R={d["best_val_pearson"]:.4f} ep={d["best_epoch"]} top1k={d["test_top1k_protein_pearson"]:.4f}')
open(csv_path, "a").write(line)
