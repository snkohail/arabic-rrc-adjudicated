# Mock example (synthetic data)

**Everything in this folder is invented.** The four "judgments" are short synthetic Arabic texts written
for this demo, with fictitious case numbers and fictitious article numbers. They are not taken from, derived
from, or representative of the confidential corpus used in the paper. Their only purpose is to show the input
file format and to let the data-side tools run end to end without access to real data.

## Contents

```
make_mock_data.py   regenerates annotation/ and mock_items.json (deterministic)
annotation/         annotator_A/ annotator_B/ Annotator_C/  one M000k.json per mock judgment
mock_items.json     16 showcase items: 4 judgments x 4 structural forms
run_demo.py         loads, validates, counts structure, projects onto grid_v1, renders a mock expert-audit page
output/             written by run_demo.py; the committed mock_expert_audit_illustrative.html is
                    a static copy with a synthetic-data notice, for viewing in a browser
```

Each mock judgment contains one instance of each form:

| form | what | where in every judgment |
|---|---|---|
| 1 | single-role region | one of PREAMBLE / FACTS / ISSUE / DECISION |
| 2 | co-extensive multi-role region | one sentence carrying FACTS and ARGUMENT_PLAINTIFF |
| 3 | nested structure | an ANALYSIS region containing a LAW_REFERENCE sentence |
| 4 | ambiguous projection sentence | a sentence covered equally by an ARGUMENT_DEFENDANT and an ANALYSIS region |

Annotator A and B files are synthetic variants of C (A omits the inner span and the crossing, B gives the
multi-role sentence one role) so that the loader's three-folder layout, including B's blind text-less format,
is exercised.

## Run

```bash
python examples/mock/make_mock_data.py      # optional: files are committed
python examples/mock/run_demo.py
```

The demo prints validation results, structural counts, the per-sentence projection status
(LABELED / AMBIGUOUS_PROJECTION with its kind / UNLABELED), checks the 16 items, and writes the blind
expert-audit interface for them to `output/mock_expert_audit.html`.

The manifests in `splits/` describe the real corpus, so `python -m rrc check`, `split`, `datasets`
and the model commands do not apply to this folder.
