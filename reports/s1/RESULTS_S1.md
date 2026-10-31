# S1 results — S1 dataset (TRAIN 10,242 / VAL 1,359 / TEST 1,846)

## S1-M1 AraBERTv02

- checkpoint `aubmindlab/bert-base-arabertv02` revision `016fb9d6768f522a59c6e0d2d5d5d43a4e1bff60`; tokenizer: aubmindlab/bert-base-arabertv02 (fast tokenizer, raw text, no Farasa segmentation)
- long-span handling: window 510 tokens, stride 128, mean of per-window [CLS] hidden states
- grid: lr ∈ [2e-05, 3e-05], epochs ≤ 5 (best VAL epoch), batch 16 examples / ≤ 48 windows, warmup 0.1, wd 0.01, dropout 0.1, fp32; thresholds 0.05…0.95 step 0.05; fallback argmax
- lr selection (seed 13, VAL set-F1): lr=2e-05: epoch 4, thr 0.15, set-F1 0.7744; lr=3e-05: epoch 4, thr 0.35, set-F1 0.7803 → **chosen lr 3e-05**

| seed | best epoch | VAL thr | VAL set-F1 | VAL micro-F1 | TEST set-F1 | TEST macro-F1 | TEST exact | TEST micro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 4 | 0.35 | 0.7803 | 0.7733 | 0.7856 | 0.7527 | 0.6831 | 0.7788 |
| 37 | 5 | 0.3 | 0.7775 | 0.7685 | 0.7776 | 0.7477 | 0.6717 | 0.7708 |
| 73 | 5 | 0.4 | 0.7825 | 0.7767 | 0.7759 | 0.7455 | 0.6842 | 0.7701 |
| **mean ± SD** | | | | | **0.7797 ± 0.0042** | 0.7486 ± 0.0030 | 0.6797 ± 0.0057 | |

Per-seed TEST 95% CIs (document-clustered bootstrap): seed 13: set-F1 [0.7388, 0.8305]; seed 37: set-F1 [0.7306, 0.8202]; seed 73: set-F1 [0.7207, 0.8222]

Per-role F1 (TEST, mean over seeds): PREAMBLE 0.939, FACTS 0.703, ISSUE 0.780, ARGUMENT_PLAINTIFF 0.431, ARGUMENT_DEFENDANT 0.662, LAW_REFERENCE 0.810, ANALYSIS 0.837, DECISION 0.822, DECISION_APPEAL 0.753

TEST slices (set-F1, mean ± SD over seeds):

| slice | n | set-F1 | macro-F1 | exact |
|---|---:|---:|---:|---:|
| single_role | 1681 | 0.7818 ± 0.0042 | 0.7560 ± 0.0047 | 0.7085 ± 0.0089 |
| multi_role | 165 | 0.7583 ± 0.0138 | 0.6149 ± 0.0373 | 0.3859 ± 0.0337 |
| nested | 269 | 0.7384 ± 0.0068 | 0.6594 ± 0.0092 | 0.6171 ± 0.0132 |
| non_nested | 1577 | 0.7867 ± 0.0059 | 0.7557 ± 0.0046 | 0.6904 ± 0.0059 |

Controlled decoding (same probabilities; set-F1 multi-label vs forced single-label; paired CI of the difference):

| seed | all: ML | all: forced | diff [95% CI] | multi-role: ML | multi-role: forced | diff [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 13 | 0.7856 | 0.7737 | 0.0118 [0.0017, 0.024] | 0.7646 | 0.6384 | 0.1263 [0.0879, 0.1689] |
| 37 | 0.7776 | 0.7598 | 0.0177 [0.0084, 0.0284] | 0.7712 | 0.6222 | 0.149 [0.1044, 0.1979] |
| 73 | 0.7759 | 0.7584 | 0.0175 [0.0091, 0.0263] | 0.7391 | 0.6182 | 0.1209 [0.0913, 0.1582] |

Label cardinality (TEST, gold vs predicted, mean over seeds):

- overall: gold 1.091, predicted 1.149
- single_role: gold 1.000, predicted 1.118
- multi_role: gold 2.018, predicted 1.461

Paired comparison vs S1-B0 (set-F1, S1-M1 AraBERTv02 − B0, identical document resamples):

| seed | model | B0 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7856 | 0.7459 | 0.0397 | [0.0156, 0.06] |
| 37 | 0.7776 | 0.7459 | 0.0317 | [0.0135, 0.0486] |
| 73 | 0.7759 | 0.7459 | 0.03 | [0.0065, 0.0497] |

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

## S1-M1′ AraBERTv2 (Farasa-segmented; supplementary)

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

Paired comparison vs S1-B0 (set-F1, S1-M1′ AraBERTv2 (Farasa-segmented; supplementary) − B0, identical document resamples):

| seed | model | B0 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7616 | 0.7459 | 0.0158 | [-0.0046, 0.0358] |
| 37 | 0.7737 | 0.7459 | 0.0278 | [0.0086, 0.0444] |
| 73 | 0.7725 | 0.7459 | 0.0266 | [0.0061, 0.0454] |

## S1-M1 AraBERTv02 vs S1-M2 CAMeLBERT-MSA (set-F1, M1 − M2, same seed paired on identical resamples)

| seed | M1 | M2 | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7856 | 0.7827 | 0.0029 | [-0.0183, 0.0242] |
| 37 | 0.7776 | 0.7884 | -0.0108 | [-0.0223, 0.0005] |
| 73 | 0.7759 | 0.7713 | 0.0046 | [-0.0069, 0.0153] |

## S1-M1 AraBERTv02 vs S1-M1′ AraBERTv2 (Farasa-segmented; supplementary) (set-F1, M1 − M1′, same seed paired on identical resamples)

| seed | M1 | M1′ | diff | 95% CI |
|---:|---:|---:|---:|---|
| 13 | 0.7856 | 0.7616 | 0.0239 | [0.0033, 0.0443] |
| 37 | 0.7776 | 0.7737 | 0.0039 | [-0.0078, 0.0174] |
| 73 | 0.7759 | 0.7725 | 0.0034 | [-0.0114, 0.0193] |

## S1-B0 (reference)

TEST set-F1 0.7459 [0.703, 0.7846], macro-F1 0.7204, exact 0.6349; threshold 0.3.
