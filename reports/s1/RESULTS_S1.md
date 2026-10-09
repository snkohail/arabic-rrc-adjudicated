# S1 results — S1 dataset (TRAIN 10,242 / VAL 1,359 / TEST 1,846)

## S1-M1 AraBERTv2

- checkpoint `aubmindlab/bert-base-arabertv2` revision `97522efce17efa33036ac619802d5cec238dcad9`; tokenizer: aubmindlab/bert-base-arabertv2 (fast tokenizer, raw text, no Farasa segmentation)
- long-span handling: window 510 tokens, stride 128, mean of per-window [CLS] hidden states
- grid: lr ∈ [2e-05, 3e-05], epochs ≤ 5 (best VAL epoch), batch 16 examples / ≤ 48 windows, warmup 0.1, wd 0.01, dropout 0.1, fp32; thresholds 0.05…0.95 step 0.05; fallback argmax
- lr selection (seed 13, VAL set-F1): lr=2e-05: epoch 5, thr 0.2, set-F1 0.7681; lr=3e-05: epoch 5, thr 0.2, set-F1 0.7685 → **chosen lr 3e-05**

| seed | best epoch | VAL thr | VAL set-F1 | VAL micro-F1 | TEST set-F1 | TEST macro-F1 | TEST exact | TEST micro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 5 | 0.2 | 0.7685 | 0.7524 | 0.7616 | 0.7183 | 0.6354 | 0.7498 |
| 37 | 5 | 0.3 | 0.7683 | 0.7575 | 0.7737 | 0.7447 | 0.6712 | 0.764 |
| 73 | 5 | 0.3 | 0.7673 | 0.7596 | 0.7725 | 0.7462 | 0.6701 | 0.7655 |
| **mean ± SD** | | | | | **0.7693 ± 0.0054** | 0.7364 ± 0.0128 | 0.6589 ± 0.0166 | |

Per-seed TEST 95% CIs (document-clustered bootstrap): seed 13: set-F1 [0.7144, 0.8044]; seed 37: set-F1 [0.7248, 0.8173]; seed 73: set-F1 [0.7192, 0.8184]

Per-role F1 (TEST, mean over seeds): PREAMBLE 0.925, FACTS 0.680, ISSUE 0.769, ARGUMENT_PLAINTIFF 0.470, ARGUMENT_DEFENDANT 0.681, LAW_REFERENCE 0.794, ANALYSIS 0.827, DECISION 0.773, DECISION_APPEAL 0.708

TEST slices (set-F1, mean ± SD over seeds):

| slice | n | set-F1 | macro-F1 | exact |
|---|---:|---:|---:|---:|
| single_role | 1681 | 0.7704 ± 0.0076 | 0.7388 ± 0.0161 | 0.6847 ± 0.0219 |
| multi_role | 165 | 0.7575 ± 0.0176 | 0.6686 ± 0.0349 | 0.3960 ± 0.0375 |
| nested | 269 | 0.7127 ± 0.0018 | 0.6020 ± 0.0268 | 0.5613 ± 0.0169 |
| non_nested | 1577 | 0.7789 ± 0.0063 | 0.7473 ± 0.0134 | 0.6755 ± 0.0167 |

Controlled decoding (same probabilities; set-F1 multi-label vs forced single-label; paired CI of the difference):

| seed | all: ML | all: forced | diff [95% CI] | multi-role: ML | multi-role: forced | diff [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 13 | 0.7616 | 0.753 | 0.0087 [-0.002, 0.021] | 0.7817 | 0.6364 | 0.1454 [0.108, 0.1891] |
| 37 | 0.7737 | 0.7582 | 0.0154 [0.0079, 0.0244] | 0.7404 | 0.6222 | 0.1182 [0.089, 0.154] |
| 73 | 0.7725 | 0.7589 | 0.0135 [0.0041, 0.0238] | 0.7505 | 0.6182 | 0.1323 [0.1058, 0.1644] |

Label cardinality (TEST, gold vs predicted, mean over seeds):

- overall: gold 1.091, predicted 1.178
- single_role: gold 1.000, predicted 1.149
- multi_role: gold 2.018, predicted 1.475

Paired comparison vs S1-B0 (set-F1, S1-M1 AraBERTv2 − B0, identical document resamples):

| seed | model | B0 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7616 | 0.7459 | 0.0158 | [-0.0046, 0.0358] |
| 37 | 0.7737 | 0.7459 | 0.0278 | [0.0086, 0.0444] |
| 73 | 0.7725 | 0.7459 | 0.0266 | [0.0061, 0.0454] |

## S1-M2 CAMeLBERT-MSA

- checkpoint `CAMeL-Lab/bert-base-arabic-camelbert-msa` revision `277069fd3645fedb22b746caf38d111aadee0241`; tokenizer: CAMeL-Lab/bert-base-arabic-camelbert-msa (fast tokenizer, raw text, no Farasa segmentation)
- long-span handling: window 510 tokens, stride 128, mean of per-window [CLS] hidden states
- grid: lr ∈ [2e-05, 3e-05], epochs ≤ 5 (best VAL epoch), batch 16 examples / ≤ 48 windows, warmup 0.1, wd 0.01, dropout 0.1, fp32; thresholds 0.05…0.95 step 0.05; fallback argmax
- lr selection (seed 13, VAL set-F1): lr=2e-05: epoch 3, thr 0.25, set-F1 0.7725; lr=3e-05: epoch 3, thr 0.3, set-F1 0.777 → **chosen lr 3e-05**

| seed | best epoch | VAL thr | VAL set-F1 | VAL micro-F1 | TEST set-F1 | TEST macro-F1 | TEST exact | TEST micro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 3 | 0.3 | 0.777 | 0.7669 | 0.7827 | 0.7512 | 0.662 | 0.7741 |
| 37 | 2 | 0.3 | 0.7762 | 0.7682 | 0.7884 | 0.7472 | 0.6798 | 0.7815 |
| 73 | 5 | 0.15 | 0.7742 | 0.7561 | 0.7713 | 0.7257 | 0.6224 | 0.7565 |
| **mean ± SD** | | | | | **0.7808 ± 0.0071** | 0.7414 ± 0.0112 | 0.6547 ± 0.0240 | |

Per-seed TEST 95% CIs (document-clustered bootstrap): seed 13: set-F1 [0.7306, 0.8272]; seed 37: set-F1 [0.7422, 0.8288]; seed 73: set-F1 [0.7193, 0.815]

Per-role F1 (TEST, mean over seeds): PREAMBLE 0.947, FACTS 0.701, ISSUE 0.770, ARGUMENT_PLAINTIFF 0.406, ARGUMENT_DEFENDANT 0.655, LAW_REFERENCE 0.814, ANALYSIS 0.836, DECISION 0.837, DECISION_APPEAL 0.706

TEST slices (set-F1, mean ± SD over seeds):

| slice | n | set-F1 | macro-F1 | exact |
|---|---:|---:|---:|---:|
| single_role | 1681 | 0.7816 ± 0.0094 | 0.7457 ± 0.0152 | 0.6766 ± 0.0314 |
| multi_role | 165 | 0.7732 ± 0.0170 | 0.6731 ± 0.0480 | 0.4323 ± 0.0515 |
| nested | 269 | 0.7344 ± 0.0182 | 0.6589 ± 0.0268 | 0.5539 ± 0.0458 |
| non_nested | 1577 | 0.7887 ± 0.0083 | 0.7482 ± 0.0118 | 0.6720 ± 0.0228 |

Controlled decoding (same probabilities; set-F1 multi-label vs forced single-label; paired CI of the difference):

| seed | all: ML | all: forced | diff [95% CI] | multi-role: ML | multi-role: forced | diff [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 13 | 0.7827 | 0.7685 | 0.0142 [0.0043, 0.0258] | 0.7628 | 0.6283 | 0.1345 [0.0993, 0.1755] |
| 37 | 0.7884 | 0.7676 | 0.0208 [0.0126, 0.0311] | 0.7596 | 0.6242 | 0.1354 [0.1037, 0.179] |
| 73 | 0.7713 | 0.7588 | 0.0126 [0.0008, 0.0267] | 0.7972 | 0.6283 | 0.1689 [0.1351, 0.2094] |

Label cardinality (TEST, gold vs predicted, mean over seeds):

- overall: gold 1.091, predicted 1.207
- single_role: gold 1.000, predicted 1.178
- multi_role: gold 2.018, predicted 1.505

Paired comparison vs S1-B0 (set-F1, S1-M2 CAMeLBERT-MSA − B0, identical document resamples):

| seed | model | B0 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7827 | 0.7459 | 0.0368 | [0.0181, 0.0523] |
| 37 | 0.7884 | 0.7459 | 0.0425 | [0.0259, 0.0586] |
| 73 | 0.7713 | 0.7459 | 0.0254 | [0.0048, 0.042] |

## S1-M1 AraBERTv2 vs S1-M2 CAMeLBERT-MSA (set-F1, M1 − M2, same seed paired on identical resamples)

| seed | M1 | M2 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7616 | 0.7827 | -0.0211 | [-0.0365, -0.0017] |
| 37 | 0.7737 | 0.7884 | -0.0147 | [-0.0326, 0.0003] |
| 73 | 0.7725 | 0.7713 | 0.0012 | [-0.0157, 0.0176] |

## S1-B0 (reference)

TEST set-F1 0.7459 [0.703, 0.7846], macro-F1 0.7204, exact 0.6349; threshold 0.3.
