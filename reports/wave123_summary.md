# Loss Function & Monitor Sweep Summary

**Date:** 2026-06-01  
**Base model:** CTransPath + MoAE (h=512)  
**Data:** CPTAC-BRCA, 5-fold case-level split  
**GPU:** RTX 5090 (33.7 GB)

---

## Wave 1: Loss Function Sweep

Tested 8 loss configurations on fold 1 (monitor=val_loss):

| Config | test_R | val_R | Epoch |
|---|---|---|---|
| smoothl1(b=2.0) | **0.4542** | 0.4849 | 96 |
| mse | 0.4525 | 0.4642 | 56 |
| huber(d=2.0) | 0.4521 | 0.4630 | 55 |
| smoothl1(b=0.5) | 0.4483 | 0.4702 | 67 |
| smoothl1(b=1.0) | 0.4481 | 0.4602 | 55 |
| huber(d=1.0) | 0.4481 | 0.4602 | 55 |
| smoothl1(b=0.1) | 0.4446 | 0.4711 | 68 |
| huber(d=0.5) | 0.4432 | 0.4609 | 55 |

**Finding:** Larger beta/delta values perform better (smoothl1 b=2.0 ~ MSE regime).

---

## Wave 2: Monitor Fix Campaign

Top 3 configs from Wave 1 retrained with `--monitor val_pearson`:

| Config | Wave 1 (val_loss) | Wave 2 (val_R) | Delta |
|---|---|---|---|
| smoothl1(b=2.0) | 0.4542 | 0.4493 | -0.0049 |
| mse | 0.4525 | 0.4475 | -0.0050 |
| huber(d=2.0) | 0.4521 | **0.4553** | **+0.0032** |

**Finding:** val_pearson monitoring helped only huber(d=2.0); hurt smoothl1 and MSE.

---

## Wave 3: Pearson-Based Losses

Tested pure differentiable Pearson loss + auxiliary Pearson term (`--pearson-aux-weight`).

| Config | test_R | val_R | Epoch |
|---|---|---|---|
| smoothl1(b=2.0) + aux 0.1 | **0.4564** | 0.4826 | 70 |
| huber(d=2.0) + aux 0.1 | 0.4514 | 0.4841 | 68 |
| smoothl1(b=2.0) + aux 0.5 | 0.4498 | 0.4961 | 68 |
| mse + aux 0.1 | 0.4486 | 0.4822 | 68 |
| smoothl1(b=2.0) + aux 1.0 | 0.4332 | 0.4939 | 78 |
| pure pearson (no aux) | 0.4075 | 0.4945 | 57 |

**Finding:** SmoothL1(b=2.0) + aux 0.1 was best on fold 1 (0.4564). Pure Pearson overfits (high val_R, low test_R). Aux weight 0.1 is sweet spot.

---

## 5-Fold CV (Top 3 Configs)

| Config | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | **Mean ± std** |
|---|---|---|---|---|---|---|
| huber(d=2.0) + monR | 0.4553 | 0.4830 | 0.4873 | 0.4249 | 0.4254 | **0.4552 ± 0.0300** |
| smoothl1(b=2.0) | 0.4542 | 0.4880 | 0.4853 | 0.4160 | 0.4318 | 0.4551 ± 0.0319 |
| smoothl1(b=2.0) + aux 0.1 | 0.4564 | 0.4904 | 0.4812 | 0.4102 | 0.4331 | 0.4543 ± 0.0333 |
| *Baseline* | *0.4481* | *0.4895* | *0.4940* | *0.4180* | *0.4233* | ***0.4546*** |

---

## Key Takeaways

1. **Marginal improvements** — All configs are within ±0.001 of baseline (0.4546). No statistically significant gain.
2. **Fold 4 curse** — Consistently the hardest fold (R~0.42) regardless of loss/monitor config.
3. **Huber(d=2.0) + val_pearson monitor** — Best overall at 0.4552, but gain is negligible.
4. **Pearson auxiliary** — Works best at low weight (0.1), degrades at higher weights. Pure Pearson loss causes overfitting.
5. **Baseline robustness** — The default smoothl1(b=1.0) + val_loss monitor is already near-optimal. Loss function and monitor choice have limited impact on this architecture/data combination.
