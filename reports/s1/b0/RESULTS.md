# S1-B0 — char n-gram TF-IDF + one-vs-rest logistic regression

## Configuration

- fixed: {"analyzer": "char_wb", "sublinear_tf": true, "min_df": 2, "max_features": 500000, "lowercase": false, "solver": "liblinear", "max_iter": 1000, "fallback": "argmax when no role passes the threshold", "seed": 13}
- grid: {"ngram_range": [[2, 4], [3, 5]], "C": [0.3, 1.0, 3.0, 10.0], "class_weight": ["none", "balanced"], "thresholds": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}
- selection: validation set-F1 (ties: smaller C, smaller n-gram, class_weight none)
- **chosen**: ngram [3, 5], C=3.0, class_weight=none, features=99634, **threshold=0.3**
- examples: {'train': 10242, 'val': 1359, 'test': 1846}

## Validation selection (TRAIN fit, VAL scored at each configuration's best global threshold)

| ngram | C | class_weight | thr | VAL set-F1 | VAL macro-F1 | VAL exact |
|---|---:|---|---:|---:|---:|---:|
| [2, 4] | 0.3 | none | 0.35 | 0.7029 | 0.6429 | 0.6078 |
| [2, 4] | 0.3 | balanced | 0.55 | 0.7168 | 0.6781 | 0.5232 |
| [2, 4] | 1.0 | none | 0.45 | 0.7302 | 0.6828 | 0.6439 |
| [2, 4] | 1.0 | balanced | 0.6 | 0.7197 | 0.685 | 0.5585 |
| [2, 4] | 3.0 | none | 0.35 | 0.7344 | 0.6951 | 0.613 |
| [2, 4] | 3.0 | balanced | 0.5 | 0.7262 | 0.6871 | 0.5386 |
| [2, 4] | 10.0 | none | 0.3 | 0.7316 | 0.6953 | 0.5916 |
| [2, 4] | 10.0 | balanced | 0.6 | 0.7344 | 0.7034 | 0.6026 |
| [3, 5] | 0.3 | none | 0.35 | 0.7005 | 0.6405 | 0.6041 |
| [3, 5] | 0.3 | balanced | 0.65 | 0.7119 | 0.6793 | 0.5791 |
| [3, 5] | 1.0 | none | 0.3 | 0.7309 | 0.6866 | 0.5872 |
| [3, 5] | 1.0 | balanced | 0.7 | 0.7229 | 0.6978 | 0.6056 |
| [3, 5] | 3.0 | none | 0.3 | 0.7373 ** | 0.6965 | 0.6004 |
| [3, 5] | 3.0 | balanced | 0.6 | 0.7291 | 0.6965 | 0.5879 |
| [3, 5] | 10.0 | none | 0.3 | 0.7335 | 0.6999 | 0.5923 |
| [3, 5] | 10.0 | balanced | 0.6 | 0.7345 | 0.7044 | 0.61 |

## TEST (evaluated once)

| metric | point | 95% CI (document-clustered bootstrap, 5000 reps, seed 13) |
|---|---:|---|
| **set-F1 (primary)** | 0.7459 | [0.703, 0.7846] |
| macro-F1 | 0.7204 | [0.6688, 0.7656] |
| exact-set accuracy | 0.6349 | [0.589, 0.6761] |

Per-role F1 (TEST): PREAMBLE 0.9327, FACTS 0.6502, ISSUE 0.7531, ARGUMENT_PLAINTIFF 0.3108, ARGUMENT_DEFENDANT 0.6829, LAW_REFERENCE 0.7911, ANALYSIS 0.7859, DECISION 0.8295, DECISION_APPEAL 0.747

## TEST slices (multi-label decoding)

| slice | n | set-F1 | macro-F1 | exact |
|---|---:|---:|---:|---:|
| single_role | 1681 | 0.7504 | 0.729 | 0.6681 |
| multi_role | 165 | 0.6994 | 0.623 | 0.297 |
| nested | 269 | 0.6799 | 0.6145 | 0.5353 |
| non_nested | 1577 | 0.7571 | 0.732 | 0.6519 |

## Controlled decoding on the same TEST probabilities

same probabilities; multilabel = threshold decoding (argmax fallback); forced_single = argmax only

| examples | n | multi-label set-F1 | forced single-label set-F1 | diff (ML − forced) | paired 95% CI |
|---|---:|---:|---:|---:|---|
| all | 1846 | 0.7459 | 0.7326 | 0.0133 | [0.0041, 0.0221] |
| genuine multi-role | 165 | 0.6994 | 0.596 | 0.1034 | [0.0678, 0.1426] |

## Secondary diagnostics (added after freeze; reporting only)

micro-F1: VAL 0.7311, TEST 0.7383

| subset | n (TEST) | gold cardinality | predicted cardinality |
|---|---:|---:|---:|
| overall | 1846 | 1.091 | 1.1587 |
| single_role | 1681 | 1.0 | 1.1374 |
| multi_role | 165 | 2.0182 | 1.3758 |
