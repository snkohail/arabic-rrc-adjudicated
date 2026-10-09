# Test predictions

Per-item predictions on the test judgments for every system reported in the paper.
One JSON object per line, no judgment text.

## S1 — span level (multi-label)

`{"id": "C0006:2-347", "case_id": "C0006", "gold": ["ISSUE", "PREAMBLE"], "probs": [...]}`

- `id` is `<case_id>:<start>-<end>` — character offsets of the span region in the
  normalised judgment; they resolve against the manifests in `splits/`.
- `gold` is the adjudicated role set (never empty).
- `probs` are sigmoid scores in the role order below.
- Decoding (see `rrc/metrics.py`): keep every role at or above the run's
  validation-tuned threshold; if none qualifies, keep the top role. File names encode
  learning rate and seed, e.g. `lr3e-05_seed13_test.jsonl`.
- `softmax/` holds the single-label softmax controls (same format; the prediction is
  the argmax).

## S2 — sentence level (single-label)

`{"id": "C0006:s0003", "case_id": "C0006", "gold": "LAW_REFERENCE", "pred": "LAW_REFERENCE", "probs": [...]}`

- `id` is `<case_id>:s<index>` — sentence index in the grid (`splits/grid_v1.json`).
- `gold` is the projected majority role, `pred` the prediction, `probs` the scores in
  the same role order (softmax for the encoders, LR probabilities for B1, CRF marginals
  for B2; B0 votes the majority class and has no scores).

## Role order

`PREAMBLE, FACTS, ISSUE, ARGUMENT_PLAINTIFF, ARGUMENT_DEFENDANT, LAW_REFERENCE, ANALYSIS, DECISION, DECISION_APPEAL`

## What's here

| folder | system |
|---|---|
| `s1/b0` | character n-gram TF-IDF + one-vs-rest logistic regression |
| `s1/m1_arabert`, `s1/m1_arabertv02`, `s1/m2_camelbert` | fine-tuned AraBERT v2 / v0.2 / CAMeLBERT-MSA |
| `s1/softmax/*` | single-label softmax controls |
| `s2/b0` | majority baseline |
| `s2/b1` | TF-IDF + logistic regression |
| `s2/b2` | lexical CRF |
| `s2/m1_arabert`, `s2/m1_arabertv02`, `s2/m2_camelbert` | fine-tuned encoders on grid sentences |

Aggregate scores derived from these files are under `reports/`. Together with the
manifests in `splits/`, these files let you recompute every table in the paper without
access to the judgments.
