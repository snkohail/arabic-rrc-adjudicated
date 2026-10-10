# Stage 0 — data audit (no training)

## C integrity audit (200 judgments, after cnorm_v1)

- validator: **PASS** (200/200); complete 200; DEFER 0; crossing-region judgments 50 (warning)
- normalisation cnorm_v1: rows 14726 → 14686 (R1 tiny 29, R2 near-identical 11; 26 judgments)
- multi-role regions are explicit adjudicator decisions: **True** (declared gold status)
- rows 14686, unique regions 13574; multi-role 1073 (7.9%) in 184 judgments (2: 1047, 3: 18, >3: 8)
- nesting relations 1832 in 120 judgments; same-role 622
- AraBERT tokens per region: median 35, p95 289, max 21432; >510: 319 (2.35%)

## Exact duplicate

- excluded from modeling: **C0047** — exact duplicate text of C0019 (identical sha256); C0019 retained as the canonical copy
- modeling corpus: 199 unique-text judgments

## Sentence grid

- `grid_v1` final, 12960 sentences over 199 judgments
- grid_sha256 `775369e72d2ae7393eae9e65375385fed3ff509c885725e0137e29d0e2253cf3`
- params: paragraph break `\r?\n[ \t\u00a0\u200f\u200e]*\r?\n`, terminator `[.؟?!]+(?=\s|$)`, single line break = soft wrap, keep if ≥1 letter/digit

## Split

- `split_v1` final, seed 13, sizes {'train': 139, 'val': 30, 'test': 30}
- manifest_sha256 `65fcb1ba20e2dc5253655cf09bd2f2f2049e924291b8c0e8824434f04f74ba01`
- indicators: PREAMBLE, FACTS, ISSUE, ARGUMENT_PLAINTIFF, ARGUMENT_DEFENDANT, LAW_REFERENCE, ANALYSIS, DECISION, DECISION_APPEAL, MULTI_ROLE_DOC, VOLUME_LOW, VOLUME_MED, VOLUME_HIGH
- volume bins: LOW < 26 ≤ MED < 58 ≤ HIGH regions/judgment
- related groups kept together: [['C0041', 'C0071', 'C0088']]

| indicator | train | val | test |
|---|---:|---:|---:|
| PREAMBLE | 139 | 30 | 30 |
| FACTS | 132 | 29 | 29 |
| ISSUE | 105 | 21 | 22 |
| ARGUMENT_PLAINTIFF | 81 | 17 | 18 |
| ARGUMENT_DEFENDANT | 114 | 24 | 25 |
| LAW_REFERENCE | 136 | 29 | 29 |
| ANALYSIS | 139 | 30 | 30 |
| DECISION | 121 | 29 | 24 |
| DECISION_APPEAL | 71 | 16 | 15 |
| MULTI_ROLE_DOC | 129 | 29 | 25 |
| VOLUME_LOW | 45 | 10 | 10 |
| VOLUME_MED | 47 | 10 | 10 |
| VOLUME_HIGH | 47 | 10 | 10 |

## S1 — original-span multi-label

| | train | val | test |
|---|---:|---:|---:|
| judgments | 139 | 30 | 30 |
| examples | 10242 | 1359 | 1846 |
| multi-role n (%) | 734 (7.17%) | 156 (11.48%) | 165 (8.94%) |
| multi-role by #labels | {'2': 715, '3': 14, '4': 2, '5': 2, '6': 1} | {'2': 150, '3': 4, '4': 2} | {'2': 164, '5': 1} |
| nested n (%) | 1273 (12.43%) | 328 (24.14%) | 269 (14.57%) |
| >510 AraBERT tokens | 223 | 42 | 54 |
| regions/judgment min / median / max | 4 / 40 / 485 | 9 / 40 / 125 | 7 / 44 / 184 |
| volume bins LOW/MED/HIGH | 45/47/47 | 10/10/10 | 10/10/10 |

Role distribution (S1 label occurrences):

| role | train | val | test |
|---|---:|---:|---:|
| PREAMBLE | 525 | 104 | 115 |
| FACTS | 2703 | 307 | 391 |
| ISSUE | 1182 | 70 | 175 |
| ARGUMENT_PLAINTIFF | 371 | 106 | 115 |
| ARGUMENT_DEFENDANT | 763 | 142 | 132 |
| LAW_REFERENCE | 1344 | 230 | 267 |
| ANALYSIS | 3397 | 439 | 689 |
| DECISION | 476 | 81 | 86 |
| DECISION_APPEAL | 243 | 44 | 44 |

## S2 — derived unambiguous single-label sentence benchmark

Labels are projections of adjudicated spans onto the grid, not independently annotated sentence gold.

| | train | val | test |
|---|---:|---:|---:|
| judgments | 139 | 30 | 30 |
| grid sentences | 9283 | 1764 | 1913 |
| eligible (LABELED) n (%) | 6298 (67.84%) | 1261 (71.49%) | 1355 (70.83%) |
| AMBIGUOUS_PROJECTION n (%) | 1534 (16.52%) | 377 (21.37%) | 322 (16.83%) |
| ambiguous: judgments affected | 119 | 30 | 26 |
| ambiguous by kind | {'co_extensive_multi_role': 914, 'other': 79, 'nested': 541} | {'co_extensive_multi_role': 251, 'nested': 102, 'other': 24} | {'co_extensive_multi_role': 176, 'nested': 136, 'other': 10} |
| UNLABELED n (%) | 1451 (15.63%) | 126 (7.14%) | 236 (12.34%) |

Role distribution (S2 eligible sentences):

| role | train | val | test |
|---|---:|---:|---:|
| PREAMBLE | 855 | 195 | 262 |
| FACTS | 1295 | 253 | 253 |
| ISSUE | 629 | 64 | 118 |
| ARGUMENT_PLAINTIFF | 243 | 70 | 48 |
| ARGUMENT_DEFENDANT | 472 | 109 | 92 |
| LAW_REFERENCE | 612 | 69 | 93 |
| ANALYSIS | 1659 | 398 | 376 |
| DECISION | 349 | 70 | 74 |
| DECISION_APPEAL | 184 | 33 | 39 |
