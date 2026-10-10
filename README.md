# arabic-rrc-adjudicated

Code for rhetorical role classification (RRC) of Arabic court judgments, where the gold
annotation allows multiple roles on one span and nested spans.

The repo contains the code, configs, the split and sentence-grid manifests (case ids,
offsets and hashes only), aggregate results, and the per-item test predictions of every
system reported in the paper. 

## Setup

```bash
pip install -r requirements.txt
export RRC_ANNOTATION_ROOT=/path/to/annotation   # annotator_A/ annotator_B/ Annotator_C/
export RRC_DATA_DIR=/path/to/private/output      # derived data, caches (never committed)
python -m pytest -q tests                        # unit tests run without the data
```

Or copy `configs/local.example.json` to `configs/local.json` (git-ignored) instead of
setting environment variables.

## Tasks

| task | unit | labels | models |
|---|---|---|---|
| S1 | adjudicated span | set of roles (multi-hot) | B0 char n-gram TF-IDF + LR; M1 AraBERT (v2, v02); M2 CAMeLBERT-MSA; softmax single-label controls |
| S2 | grid sentence | one role (unambiguous projections only) | B0 majority; B1 TF-IDF + LR; B2 lexical CRF; M1; M2 |

Nine roles: `PREAMBLE, FACTS, ISSUE, ARGUMENT_PLAINTIFF, ARGUMENT_DEFENDANT, LAW_REFERENCE,
ANALYSIS, DECISION, DECISION_APPEAL`.

## Reproducing the results

```bash
make data      # integrity audit, duplicate check, build S1/S2 datasets, summary report
make s1        # S1 baseline and transformer runs, aggregation
make s2        # S2 baselines and transformer runs, aggregation
make softmax   # S1 single-label softmax controls
python -m rrc expert-audit build           # build the blind audit sample
python -m rrc expert-audit score --responses responses.json
python scripts/grid_robustness_v1_v3.py    # sentence-grid sensitivity (grid_v1 vs grid_v3)
```

`python -m rrc --help` lists all commands. The manifests in `splits/` are the exact
versions behind every reported number; `freeze-grid`, `split` and `grid-v3` are one-time
commands that refuse to overwrite them.

Without the confidential corpus you can still recompute every reported table from
`predictions/` + `splits/`, and run the mock example below.

## Example

`examples/mock/` holds four short synthetic judgments (invented text, not from the corpus)
so the loader, validator, projection and audit renderer can run without real data:
`python examples/mock/run_demo.py`. See `examples/mock/README.md`.

## Test predictions

`predictions/` holds the per-item test predictions of every reported system (ids, gold
labels, probabilities — no text). Format details in `predictions/README.md`.

## Layout

```
examples/     synthetic mock data and demo
rrc/          the package (loading, normalisation, split, grid, projection, models, metrics, audit)
configs/      role definitions shown to experts; example local config
splits/       grid and split manifests (+ sha256)
reports/      aggregate results (counts and scores only, no text)
predictions/  per-item test predictions (no text)
scripts/      run scripts
tests/        pytest (unit tests need no data; integration tests skip without RRC_ANNOTATION_ROOT)
```

## License

MIT, see `LICENSE`.
