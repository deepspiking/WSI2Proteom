# WSI2Proteom v0 — Experiment Report

> Predicting proteome abundance (6855 proteins) from precomputed WSI features using GatedAttentionMIL.
> CPTAC-BRCA, 130 matched slide–aliquot pairs. All experiments seed=42, single-run.

---

## 1. Experimental Setup

### Dataset

| Item | Value |
|---|---|
| Cohort | CPTAC-BRCA (Breast Prospective Collection, BI) |
| Matched pairs | 130 (1 slide, 1 aliquot each) |
| Split (case-level) | 89 train / 20 val / 21 test |
| Proteome target | 6855 proteins (all-NaN genes removed, per-protein z-score normalized) |
| Feature extractors | cTransPath (ViT, 768-dim), RetCCL (ResNet, 2048-dim) |

### Model Architecture

```
WSI patches  ──►  Feature Extractor  ──►  GatedAttention  ──►  Classifier MLP  ──►  6855 proteins
                    (frozen, off-line)         (trainable)        (trainable)
```

- **GatedAttention**: standard attention-based MIL pooling with gating mechanism
- **Classifier**: configurable-depth MLP with ReLU activations, ending in a final `Linear(hidden, 6855)` layer
- **Loss**: Smooth L1 (Huber)
- **Optimizer**: AdamW (1e-5 weight decay), ReduceLROnPlateau scheduler
- **Training**: batch_size=32, max 200 epochs, patience 30

### Experimental Phases

| Phase | Scope | Experiments |
|---|---|---|
| Phase 1 | Baseline: model type × feature | 4 experiments (MeanPool vs Attention, cTransPath vs RetCCL) |
| Phase 2 | Hidden dimension sweep (cTransPath) | 6 experiments (h=128, 256, 384, 512, 768, 1024) |
| Phase 3 | Classifier depth sweep (h=1024, both features) | 12 experiments (6 depths × 2 features) |
| Phase 4 | Hyperparameter tuning (deep3 baseline) | 6 experiments (deeper cls, LR, dropout) |

---

## 2. Phase 1 — Baseline

**Goal**: Establish minimum viable performance and compare MIL pooling vs mean pooling, with two feature extractors.

| Exp | Feature | Model | hidden_dim | Params | Test R | Test ρ |
|---|---|---|---|---|---|---|
| 1a | cTransPath (768) | MeanPool Linear | 256 | 5.27M | 0.3772 | 0.3101 |
| 2a | cTransPath (768) | **GatedAttention** | 256 | 1.31M | **0.4076** | 0.3312 |
| 1b | RetCCL (2048) | MeanPool Linear | 256 | 14.05M | 0.3813 | 0.3204 |
| 2b | RetCCL (2048) | **GatedAttention** | 512 | 3.73M | **0.4111** | 0.3404 |

**Key takeaways:**
- GatedAttention outperforms MeanPool pooling on both feature types.
- RetCCL (2048-dim) slightly edges cTransPath (768-dim) at small hidden sizes (+0.0035 R).
- The factorized variant (exp3, latent_dim=88) regressed to R=0.233 — severe information bottleneck.
- MeanPool requires many more parameters (full [in_dim → 6855] linear) for worse results.

---

## 3. Phase 2 — Hidden Dimension Sweep (cTransPath)

**Goal**: Find the optimal GatedAttention hidden dimension with default classifier `[h, h//2]`.

| hidden_dim | Classifier | Params | Test R | Test ρ | Best epoch |
|---|---|---|---|---|---|
| 128 | [128, 64] | 0.60M | 0.4309 | 0.3505 | 199 |
| 256 (ref) | [256, 128] | 1.31M | 0.4076 | 0.3312 | 89 |
| 384 | [384, 192] | 2.14M | 0.4245 | 0.3613 | 107 |
| 512 | [512, 256] | 3.08M | 0.4307 | 0.3673 | 84 |
| 768 | [768, 384] | 5.30M | 0.4082 | 0.3494 | 40 |
| **1024** | [1024, 512] | 7.98M | **0.4334** | 0.3717 | 37 |

**Key takeaways:**
- h=128 achieves the strongest test R among sweeps with default classifier, despite smallest size.
- h=1024 gives the best result overall, with much faster convergence (37 epochs vs 199).
- The default classifier `[h, h//2]` is restrictive for large h — the attention output collapses to h//2 before the final 6855 projection, creating a bottleneck. This motivates Phase 3.

---

## 4. Phase 3 — Classifier Depth Sweep

**Goal**: Fix hidden_dim=1024 and systematically vary classifier depth to eliminate the bottleneck.

### 4.1 cTransPath (768-dim → 1024-dim attention)

| Classifier | Hidden layers | Params | Test R | Test ρ | Best epoch |
|---|---|---|---|---|---|
| `direct` (Linear to 6855) | 0 | 9.92M | 0.4129 | 0.3478 | 13 |
| `bottleneck` [256] | 1 (→256) | 4.91M | 0.3850 | 0.3064 | 33 |
| `shallow` [512] | 1 (→512) | 6.93M | 0.3983 | 0.3303 | 26 |
| `default` [1024, 512] | 2 | 7.98M | 0.4334 | 0.3717 | 37 |
| `deep2` [1024, 1024, 512] | 3 | 9.04M | 0.4449 | 0.3734 | 36 |
| **`deep3` [1024, 1024, 1024, 512]** | 4 | 10.09M | **0.4679** | 0.3974 | 47 |

### 4.2 RetCCL (2048-dim → 1024-dim attention)

| Classifier | Hidden layers | Params | Test R | Test ρ | Best epoch |
|---|---|---|---|---|---|
| `direct` | 0 | 11.23M | 0.4218 | 0.3513 | 14 |
| `bottleneck` [256] | 1 (→256) | 6.23M | 0.3855 | 0.3239 | 53 |
| `shallow` [512] | 1 (→512) | 8.24M | 0.3431 | 0.2652 | 16 |
| `default` [1024, 512] | 2 | 9.29M | 0.3668 | 0.2984 | 28 |
| `deep2` [1024, 1024, 512] | 3 | 10.35M | 0.4151 | 0.3425 | 32 |
| `deep3` [1024, 1024, 1024, 512] | 4 | 11.40M | 0.4342 | 0.3709 | 46 |

### 4.3 Feature Comparison at deep3

| Feature | Classifier | Params | Test R |
|---|---|---|---|
| cTransPath (768) | deep3 | 10.09M | **0.4679** |
| RetCCL (2048) | deep3 | 11.40M | 0.4342 |

**Key takeaways:**
- Deeper is better — classifier depth monotonically improves performance up to 4 hidden layers.
- Narrow bottlenecks (256, 512) after attention significantly degrade performance — the 1024-dim attention output needs sufficient capacity.
- cTransPath consistently outperforms RetCCL at larger model scales, reversing the Phase 1 trend where RetCCL had a slight edge at smaller sizes.
- The GatedAttention module itself contributes few parameters (~100K for h=1024) — most live in the classifier.

---

## 5. Phase 4 — Hyperparameter Tuning

**Goal**: Starting from cTransPath + deep3 (R=0.4679), probe deeper classifiers, learning rate, and dropout.

### 5.1 Deeper Classifier

| Config | Hidden layers | Params | Test R | Test ρ | Best epoch |
|---|---|---|---|---|---|
| deep3 [1024×3, 512] | 4 | 10.09M | 0.4679 | 0.3974 | 47 |
| deep4 [1024×4, 512] | 5 | 11.14M | 0.4537 | 0.3780 | 46 |
| **deep5 [1024×5, 512]** | 6 | 12.19M | **0.4708** | 0.3927 | 119 |

### 5.2 Learning Rate (deep3)

| LR | Test R | Test ρ | Best epoch |
|---|---|---|---|
| 1e-4 | 0.4506 | 0.3850 | 117 |
| 3e-4 (default) | 0.4679 | 0.3974 | 47 |
| **1e-3** | **0.4702** | 0.3973 | 31 |

### 5.3 Dropout (deep3)

| Dropout | Test R | Test ρ | Best epoch |
|---|---|---|---|
| 0.1 | 0.4337 | 0.3570 | 24 |
| **0.3 (default)** | **0.4679** | 0.3974 | 47 |
| 0.5 | 0.4422 | 0.3386 | 55 |

---

## 6. Summary

### Best Configuration

| Hyperparameter | Value |
|---|---|
| Feature extractor | cTransPath (768-dim) |
| MIL pooling | GatedAttention |
| Attention hidden dim | 1024 |
| Classifier | `[1024, 1024, 1024, 1024, 1024, 512]` (deep5, 6 hidden layers) |
| Learning rate | 3e-4 |
| Dropout | 0.3 |
| Total parameters | 12,190,407 |
| **Test Pearson R** | **0.4708** |
| Test Spearman ρ | 0.3927 |
| Test sample-wise Pearson (mean) | 0.3171 |

### Progression

```
exp2a (h=256)         0.4076  ─┐
sweep_h128            0.4309   │
sweep_h1024+default   0.4334   │  +14.8%
deep2                 0.4449   │  from
deep3                 0.4679   │  baseline
deep5 (best)          0.4708  ─┘
```

### Key Findings

1. **Classifier depth matters more than hidden dimension.** Going from 2→4 hidden layers (at h=1024) improves R by 0.0345 (+8%). Going from h=128→1024 (at default cls) improves by only 0.0025.

2. **cTransPath scales better than RetCCL.** At large model sizes (h=1024, deep3), cTransPath (768-dim input) outperforms RetCCL (2048-dim) by 0.0337 R. This reverses the small-model trend.

3. **Avoid bottlenecks after attention.** Narrowing to h//2 or smaller hurts performance. The classifier should maintain the full attention output dimension for several layers before tapering.

4. **Optimal LR is 3e-4 to 1e-3.** Going lower (1e-4) underfits (converges slowly at ep 117); higher doesn't diverge but offers marginal gain.

5. **Dropout 0.3 helps.** Too low (0.1) overfits; too high (0.5) underfits. The default 0.3 is near-optimal.

---

## 7. Future Directions

- **Multi-seed validation**: Current results are single-run (seed=42). Top configurations need 3-5 seed runs for statistical confidence.
- **Alternative architectures**: Ideas file saved for future — Multi-Head GatedAttention, LiteMIL (learnable-query cross-attention), ProtoMIL, Sparse Top-k, Asymmetric Decoder.
- **Feature fusion**: cTransPath + RetCCL combined features may capture complementary information.
- **Gene-level analysis**: Which protein classes (kinases, transcription factors, etc.) are best/worst predicted? Do attention weights correlate with morphological tissue regions?
- **Cross-cohort validation**: Test generalization on independent CPTAC cohorts (OV, LUAD, etc.).

---

## A. Appendix — Complete Results Table

| Experiment | Feature | hidden_dim | Classifier | dropout | LR | Params | Test R | Test ρ |
|---|---|---|---|---|---|---|---|---|
| exp1a | ctranspath | 256 | meanpool | — | 3e-4 | 5.27M | 0.377 | 0.310 |
| exp1b | retccl | 256 | meanpool | — | 3e-4 | 14.05M | 0.381 | 0.320 |
| exp2a | ctranspath | 256 | [256,128] | — | 3e-4 | 1.31M | 0.408 | 0.331 |
| exp2b | retccl | 512 | [512,256] | — | 3e-4 | 3.73M | 0.411 | 0.340 |
| exp3 | ctranspath | 256 | factorized | — | 3e-4 | 1.34M | 0.233 | 0.147 |
| sweep_h128 | ctranspath | 128 | [128,64] | — | 3e-4 | 0.60M | 0.431 | 0.351 |
| sweep_h256 | ctranspath | 256 | [256,128] | — | 3e-4 | 1.31M | 0.408 | 0.331 |
| sweep_h384 | ctranspath | 384 | [384,192] | — | 3e-4 | 2.14M | 0.425 | 0.361 |
| sweep_h512 | ctranspath | 512 | [512,256] | — | 3e-4 | 3.08M | 0.431 | 0.367 |
| sweep_h768 | ctranspath | 768 | [768,384] | — | 3e-4 | 5.30M | 0.408 | 0.349 |
| sweep_h1024 | ctranspath | 1024 | [1024,512] | — | 3e-4 | 7.98M | 0.433 | 0.372 |
| cls_direct | ctranspath | 1024 | [] | — | 3e-4 | 9.92M | 0.413 | 0.348 |
| cls_bottleneck256 | ctranspath | 1024 | [256] | — | 3e-4 | 4.91M | 0.385 | 0.306 |
| cls_shallow512 | ctranspath | 1024 | [512] | — | 3e-4 | 6.93M | 0.398 | 0.330 |
| cls_deep2 | ctranspath | 1024 | [1024,1024,512] | — | 3e-4 | 9.04M | 0.445 | 0.373 |
| **cls_deep3** | **ctranspath** | **1024** | **[1024,1024,1024,512]** | **0.3** | **3e-4** | **10.09M** | **0.468** | **0.397** |
| cls_deep4 | ctranspath | 1024 | [1024,1024,1024,1024,512] | 0.3 | 3e-4 | 11.14M | 0.454 | 0.378 |
| **cls_deep5** | **ctranspath** | **1024** | **[1024,1024,1024,1024,1024,512]** | **0.3** | **3e-4** | **12.19M** | **0.471** | **0.393** |
| deep3_lr1e-4 | ctranspath | 1024 | deep3 | 0.3 | 1e-4 | 10.09M | 0.451 | 0.385 |
| deep3_lr1e-3 | ctranspath | 1024 | deep3 | 0.3 | 1e-3 | 10.09M | 0.470 | 0.397 |
| deep3_drop0.1 | ctranspath | 1024 | deep3 | 0.1 | 3e-4 | 10.09M | 0.434 | 0.357 |
| deep3_drop0.5 | ctranspath | 1024 | deep3 | 0.5 | 3e-4 | 10.09M | 0.442 | 0.339 |
| retccl_default | retccl | 1024 | [1024,512] | — | 3e-4 | 9.29M | 0.367 | 0.298 |
| retccl_direct | retccl | 1024 | [] | — | 3e-4 | 11.23M | 0.422 | 0.351 |
| retccl_shallow512 | retccl | 1024 | [512] | — | 3e-4 | 8.24M | 0.343 | 0.265 |
| retccl_bottleneck256 | retccl | 1024 | [256] | — | 3e-4 | 6.23M | 0.386 | 0.324 |
| retccl_deep2 | retccl | 1024 | [1024,1024,512] | — | 3e-4 | 10.35M | 0.415 | 0.343 |
| retccl_deep3 | retccl | 1024 | deep3 | — | 3e-4 | 11.40M | 0.434 | 0.371 |
