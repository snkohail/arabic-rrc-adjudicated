# Minimal C-gold audit

C source: `<annotation_root>/Annotator_C`

Normalisation `cnorm_v1` (applied in code, annotation files untouched): rows 14726 -> 14686; dropped R1 tiny (<5 chars or no letter) = 29, R2 near-identical same-label duplicates (len diff <= 20) = 11; judgments touched = 26. Inherited labels kept as adjudicated.

Rules:
- `R1_tiny`: drop rhetorical row if (end - begin) < 5 characters or the span contains no Unicode letter (regex [^\W\d_])
- `R2_near_identical`: for two rows with the same label where one span contains the other and the length difference is <= 20 characters, drop the inner (shorter) row and keep the outer; R1 runs before R2
- `not_applied`: labels duplicating an enclosing region's label are kept as adjudicated
- row-level log with case id, offsets, label, rule, reason and removed text: PRIVATE artifact `<data_dir>/audit/c_normalization_log.json` (not in Git)

## A. Integrity

- cases: 200; complete (>=1 rhetorical span): 200; with DEFER: 0
- validator: **PASS** (200 pass / 0 fail)
- cases with crossing (partially overlapping) regions: 50 (warning only)
- multi-role confirmation basis: declared gold status: Annotator_C is treated as the human-adjudicated final gold; every multi-role region is taken as an explicit adjudicator decision. Metadata flags left by the annotation tool are not used as evidence.
- metadata flags trusted: False (files declaring `NOT_human_adjudicated_gold`: 200; confirmation by flags alone would be False)
- row provenance (tool artefact counts): {'llm_adjudicated': 9534, 'agreed_A_and_B': 3575, 'unadjudicated_B': 387, 'unadjudicated_A': 1169, 'agreed_A_and_B_restored': 21}
- row candidate_source: {'B': 5275, 'A': 2083, 'A+B': 2072, 'UNION': 104}
- multi-role regions: 1073; all rows reviewed: 905; with unreviewed rows: 131; with UNION marker: 37; missing provenance: 0
- every multi-role region is an explicit adjudicator decision: **True**

## B. Gold size

- raw rhetorical rows: 14686
- unique span regions: 13574

## C. Nine-role counts (rows / regions carrying role)

| role | rows | regions |
|---|---:|---:|
| PREAMBLE | 747 | 747 |
| FACTS | 3452 | 3452 |
| ISSUE | 1445 | 1445 |
| ARGUMENT_PLAINTIFF | 592 | 592 |
| ARGUMENT_DEFENDANT | 1039 | 1039 |
| LAW_REFERENCE | 1864 | 1864 |
| ANALYSIS | 4564 | 4564 |
| DECISION | 652 | 652 |
| DECISION_APPEAL | 331 | 331 |

## D. Multi-role structure

- regions with >1 role: 1073 (7.9% of regions)
- judgments with >=1 multi-role region: 184
- by role count: 2 = 1047, 3 = 18, >3 = 8

## E. Nesting

- nested span relations: 1832
- judgments with nesting: 120
- same-role nested relations: 622

## F. Length (AraBERTv2 tokens, unique regions)

- median 35, p95 289, max 21432
- exceeding 510 tokens: 319 (2.35%)
- raw text, no Farasa pre-segmentation, no special tokens
