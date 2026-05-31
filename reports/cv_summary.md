# MoAE 5-fold Cross-Validation 종합 결과

## 1. 실험 개요

CPTAC-BRCA 데이터셋에 대해 5개의 WSI foundation model로부터 추출한 patch-level feature로 MoAE 모델의 5-fold cross-validation을 수행.

### 데이터셋
- **슬라이드**: 653개 SVS (CPTAC-BRCA, 106GB)
- **매칭된 케이스**: 130 slides / 117 unique case IDs (proteome 매칭 완료)
- **타겟**: CPTAC2 Breast Proteome (6,855개 단백질 발현값)
- **타일 크기**: 256×256 @ 1.0 MPP (UNI2-h, CTransPath, RetCCL, EXAONE)
- **타일 크기**: 512×512 @ 1.0 MPP (CONCH)

### 평가 모델 (Head Architecture)
| Config | Description | Params |
|---|---|---|
| moae_h512 | MoAE (h=512, e=4) | 3.67M |
| dualpath_h1536 | DualPath (h=1536) | 27.7M |
| moae_h1024_dp05 | MoAE (h=1024, e=6, dp=0.5) | 20.7M |

## 2. Feature Extraction

| Model | Dim | Source | Status | Time |
|---|---|---|---|---|
| UNI2-h | 1536 | HF (timm) | 653/653 ✅ | ~10h |
| CONCH | 512 | HF | 653/653 ✅ | ~3h |
| CTransPath | 768 | timm standalone | 653/653 ✅ | ~3h |
| RetCCL | 2048 | timm | 653/653 ✅ | ~2h |
| EXAONE Path 2.0 | 768 | Official repo (OOM 패치) | 653/653 ✅ | ~13h |

### EXAONE OOM 해결
- 원인: `_load_wsi()`가 level 0 전체 이미지(61082×117120)를 한번에 로드 → 55GB RSS
- 해결: large_tile(8192×8192) 청크 단위 cucim GPU 디코딩으로 대체
- 결과: VRAM peak 1~2.5GB (슬라이드 크기 무관)

## 3. 5-fold CV 결과

### 전체 모델 × Config 행렬 (mean test Pearson R)

| Feature Model | Dim | moae_h512 | dualpath_h1536 | moae_h1024_dp05 | **Best** |
|---|---|---|---|---|---|
| **CTransPath** | 768 | **0.4546** | 0.4444 | 0.4474 | **0.4546** |
| **UNI2-h** | 1536 | **0.4540** | 0.4246 | 0.4548 | **0.4548** |
| **CONCH** | 512 | 0.4429 | 0.4411 | **0.4468** | **0.4468** |
| **RetCCL** | 2048 | **0.4343** | 0.4071 | 0.3957 | **0.4343** |
| **EXAONE2** | 768 | 0.3914 | 0.3813 | **0.3968** | **0.3968** |

### Top-1000 단백질 Pearson R

| Feature Model | moae_h512 | dualpath_h1536 | moae_h1024_dp05 | **Best** |
|---|---|---|---|---|
| **UNI2-h** | **0.7834** | 0.7677 | 0.7809 | **0.7834** |
| **CONCH** | 0.7769 | **0.7782** | 0.7787 | **0.7787** |
| **CTransPath** | **0.7772** | 0.7760 | 0.7723 | **0.7772** |
| **RetCCL** | **0.7741** | 0.7639 | 0.7523 | **0.7741** |
| **EXAONE2** | 0.7540 | 0.7510 | **0.7598** | **0.7598** |

### moae_h512 — fold별 상세

| Fold | CTransPath | UNI2-h | CONCH | RetCCL | EXAONE2 |
|---|---|---|---|---|---|
| 1 | 0.4481 | 0.4073 | 0.4259 | 0.4165 | 0.3796 |
| 2 | 0.4895 | **0.5004** | **0.4936** | **0.4666** | **0.4245** |
| 3 | **0.4940** | **0.5140** | 0.4833 | **0.4829** | 0.3976 |
| 4 | 0.4180 | 0.3903 | 0.3626 | 0.3713 | 0.3475 |
| 5 | 0.4233 | 0.4579 | 0.4489 | 0.4341 | 0.4076 |

### CTransPath — 이전 manifest vs 새 manifest 비교 (검증)

| Config | Old manifest | New manifest | 차이 |
|---|---|---|---|
| moae_h512 | 0.4558 | 0.4546 | -0.0013 |
| dualpath_h1536 | 0.4412 | 0.4444 | +0.0031 |
| moae_h1024_dp05 | 0.4365 | 0.4474 | +0.0109 |

→ 이전 manifest 결과 유효함 확인.

## 4. 분석

### 순위 (moae_h512 기준)
```
1. CTransPath  0.4546 ± 0.0358  ← 가장 안정적인 분산
2. UNI2-h      0.4540 ± 0.0548  ← CTransPath와 동급, 분산略 큼
3. CONCH       0.4429 ± 0.0524  ← 512-dim으로 좋은 효율
4. RetCCL      0.4343 ± 0.0438
5. EXAONE2     0.3914 ± 0.0294  ← 유의미하게 낮음
```

### 주요 발견
1. **CTransPath / UNI2-h / CONCH** 세 모델이 비슷한 성능대 (0.44~0.45)
   - 세 모델 간 차이가 0.01 이내로 실질적으로 동등
   - CONCH가 512-dim으로 가장 효율적 (저장공간/메모리 1/3)
2. **EXAONE2**는 전체 R에서 유의미하게 낮음 (0.39)
   - 그러나 Top-1000 R에서는 0.7540으로 격차 좁혀짐
   - 특정 단백질 예측에는 경쟁력 있음
   - 원인 추정: EXAONE의 native tile size(224px)와 256px mismatch
3. **Fold 4**가 거의 모든 모델에서 최저치 → 특정 fold에 어려운 case 집중
4. **Top-1000 R**은 모든 모델이 0.75~0.78로 전체 R보다 ~1.7x 높음

### Config별 특성
- **moae_h512**: 가장 안정적이고 일관된 성능 (3.67M params)
- **dualpath_h1536**: 대부분의 모델에서 가장 낮은 성능 (27.7M params, 가장 큼)
- **moae_h1024_dp05**: UNI2-h에서 최고 성능 기록 (0.4548)

## 5. 실험 환경

| 항목 | 사양 |
|---|---|
| CPU | AMD Ryzen 9 3900X (12C/24T) |
| RAM | 62GB |
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| 저장소 | NVMe + SATA SSD, 3.6TB |
| 병렬 처리 | 18 tile extraction sessions → 4→1 CV training jobs |
| 소요 시간 | Feature extraction: ~31h / CV training: ~8h |
