# S2 results — derived unambiguous single-label sentence benchmark (grid_v1, split_v1)

Eligible sentences: TRAIN 6298 / VAL 1261 / TEST 1355 (AMBIGUOUS_PROJECTION and UNLABELED excluded; no NONE class). Labels are projections of adjudicated spans, not independently annotated sentence gold.

## TEST summary (primary macro-F1; micro-F1 = accuracy in single-label classification)

| model | config | TEST macro-F1 [95% CI] | TEST micro-F1 [95% CI] | accuracy |
|---|---|---:|---:|---:|
| S2-B0 majority | ANALYSIS | 0.0483 [0.043, 0.0534] | 0.2775 [0.2396, 0.3162] | 0.2775 |
| S2-B1 TF-IDF + LR | {"ngram_range": [3, 5], "C": 3.0, "class_weight": "balanced", "n_features": 86050, "val_macro_f1": 0.7242, "val_micro_f1": 0.7613} | 0.7175 [0.6399, 0.7747] | 0.7454 [0.6933, 0.7943] | 0.7454 |
| S2-B2 lexical CRF | {"c1": 0.05, "c2": 0.5, "val_macro_f1": 0.727, "val_micro_f1": 0.7653, "seconds": 2} | 0.7218 [0.6577, 0.7807] | 0.7882 [0.7391, 0.8485] | 0.7882 |
| S2-M1 AraBERTv2 seed 13 | lr 2e-05, epoch 3 | 0.6978 [0.6108, 0.7625] | 0.7498 [0.701, 0.8029] | 0.7498 |
| S2-M1 AraBERTv2 seed 37 | lr 2e-05, epoch 4 | 0.7177 [0.64, 0.7817] | 0.7565 [0.705, 0.8145] | 0.7565 |
| S2-M1 AraBERTv2 seed 73 | lr 2e-05, epoch 5 | 0.7162 [0.6349, 0.7793] | 0.7579 [0.7112, 0.8114] | 0.7579 |
| **S2-M1 AraBERTv2 mean ± SD** | | **0.7106 ± 0.0090** | 0.7547 ± 0.0035 | 0.7547 ± 0.0035 |
| S2-M2 CAMeLBERT-MSA seed 13 | lr 2e-05, epoch 4 | 0.7178 [0.6368, 0.7904] | 0.7616 [0.7075, 0.8241] | 0.7616 |
| S2-M2 CAMeLBERT-MSA seed 37 | lr 2e-05, epoch 5 | 0.7095 [0.6312, 0.7701] | 0.755 [0.7043, 0.8122] | 0.755 |
| S2-M2 CAMeLBERT-MSA seed 73 | lr 2e-05, epoch 5 | 0.7167 [0.6429, 0.7761] | 0.7506 [0.6976, 0.8096] | 0.7506 |
| **S2-M2 CAMeLBERT-MSA mean ± SD** | | **0.7147 ± 0.0037** | 0.7557 ± 0.0045 | 0.7557 ± 0.0045 |

## Configurations and validation selection

- B1: fixed {"analyzer": "char_wb", "sublinear_tf": true, "min_df": 2, "max_features": 500000, "lowercase": false, "solver": "lbfgs", "multi_class": "multinomial", "max_iter": 2000, "seed": 13}; grid {"ngram_range": [[2, 4], [3, 5]], "C": [0.3, 1.0, 3.0, 10.0], "class_weight": ["none", "balanced"]}; chosen ngram [3, 5], C=3.0, class_weight=balanced (VAL macro-F1 0.7242); VAL macro-F1 range over grid 0.6625–0.7242
- B2: own-sentence lexical features + run-boundary BOS/EOS + judgment-grid position decile; label transitions learned; no neighbour text; {"algorithm": "lbfgs", "max_iterations": 200, "all_possible_transitions": true}; grid {"c1": [0.05, 0.1, 0.5], "c2": [0.05, 0.1, 0.5]}; chosen c1=0.05, c2=0.5 (VAL macro-F1 0.727); VAL range 0.6747–0.727
- m1_arabert: checkpoint `aubmindlab/bert-base-arabertv2` rev `97522efce17efa33036ac619802d5cec238dcad9`; aubmindlab/bert-base-arabertv2 (fast tokenizer, raw text, no Farasa segmentation); windows 510/stride 128; lr grid [2e-05, 3e-05], epochs ≤ 5; lr selection (seed 13): lr=2e-05: epoch 3, VAL macro-F1 0.7427; lr=3e-05: epoch 3, VAL macro-F1 0.7404 → chosen 2e-05; per-seed best epoch / VAL macro-F1: 13: 3 / 0.7427, 37: 4 / 0.7213, 73: 5 / 0.724
- m2_camelbert: checkpoint `CAMeL-Lab/bert-base-arabic-camelbert-msa` rev `277069fd3645fedb22b746caf38d111aadee0241`; CAMeL-Lab/bert-base-arabic-camelbert-msa (fast tokenizer, raw text, no Farasa segmentation); windows 510/stride 128; lr grid [2e-05, 3e-05], epochs ≤ 5; lr selection (seed 13): lr=2e-05: epoch 4, VAL macro-F1 0.7591; lr=3e-05: epoch 4, VAL macro-F1 0.7488 → chosen 2e-05; per-seed best epoch / VAL macro-F1: 13: 4 / 0.7591, 37: 5 / 0.759, 73: 5 / 0.7456

## Per-role F1 (TEST; neural = mean over seeds)

| role | support | S2-B0 majority | S2-B1 TF-IDF + LR | S2-B2 lexical CRF | M1 | M2 |
|---|---:|---:|---:|---:|---:|---:|
| PREAMBLE | 262 | 0.0 | 0.9242 | 0.9638 | 0.8963 | 0.9175 |
| FACTS | 253 | 0.0 | 0.6478 | 0.7108 | 0.6992 | 0.6858 |
| ISSUE | 118 | 0.0 | 0.6726 | 0.7589 | 0.6858 | 0.6923 |
| ARGUMENT_PLAINTIFF | 48 | 0.0 | 0.4127 | 0.241 | 0.4731 | 0.4424 |
| ARGUMENT_DEFENDANT | 92 | 0.0 | 0.6703 | 0.5783 | 0.6337 | 0.5842 |
| LAW_REFERENCE | 93 | 0.0 | 0.7513 | 0.7979 | 0.7583 | 0.7478 |
| ANALYSIS | 376 | 0.4344 | 0.7612 | 0.8279 | 0.7880 | 0.7948 |
| DECISION | 74 | 0.0 | 0.8442 | 0.8533 | 0.7638 | 0.8154 |
| DECISION_APPEAL | 39 | 0.0 | 0.7733 | 0.7647 | 0.6969 | 0.7519 |

## Paired contrasts (TEST macro-F1, identical document resamples)

| contrast | a | b | diff | 95% CI |
|---|---:|---:|---:|---|
| B1 − B0 | 0.7175 | 0.0483 | 0.6692 | [0.59, 0.7281] |
| B2 − B1 | 0.7218 | 0.7175 | 0.0043 | [-0.0314, 0.0473] |
| M1 seed 13 − B1 | 0.6978 | 0.7175 | -0.0197 | [-0.0578, 0.0229] |
| M1 seed 13 − B2 | 0.6978 | 0.7218 | -0.024 | [-0.0767, 0.0118] |
| M1 seed 37 − B1 | 0.7177 | 0.7175 | 0.0002 | [-0.0342, 0.0442] |
| M1 seed 37 − B2 | 0.7177 | 0.7218 | -0.0041 | [-0.043, 0.0277] |
| M1 seed 73 − B1 | 0.7162 | 0.7175 | -0.0013 | [-0.0352, 0.0386] |
| M1 seed 73 − B2 | 0.7162 | 0.7218 | -0.0056 | [-0.0539, 0.0304] |
| M2 seed 13 − B1 | 0.7178 | 0.7175 | 0.0003 | [-0.0295, 0.0445] |
| M2 seed 13 − B2 | 0.7178 | 0.7218 | -0.0041 | [-0.038, 0.0273] |
| M2 seed 37 − B1 | 0.7095 | 0.7175 | -0.008 | [-0.0323, 0.0229] |
| M2 seed 37 − B2 | 0.7095 | 0.7218 | -0.0123 | [-0.0514, 0.0181] |
| M2 seed 73 − B1 | 0.7167 | 0.7175 | -0.0008 | [-0.0264, 0.0291] |
| M2 seed 73 − B2 | 0.7167 | 0.7218 | -0.0051 | [-0.0459, 0.0238] |
| M1 − M2 seed 13 | 0.6978 | 0.7178 | -0.02 | [-0.0607, 0.005] |
| M1 − M2 seed 37 | 0.7177 | 0.7095 | 0.0082 | [-0.0185, 0.0356] |
| M1 − M2 seed 73 | 0.7162 | 0.7167 | -0.0005 | [-0.029, 0.0236] |
