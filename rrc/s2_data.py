"""S2 data access: the derived unambiguous single-label sentence benchmark.

Only sentences with ``status == LABELED`` are used (AMBIGUOUS_PROJECTION and UNLABELED
sentences are excluded; there is no NONE class).  Sequences for the CRF are the eligible
sentences of one judgment in document order (excluded sentences leave gaps).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List

from .config import DERIVED_DIR


def load_s2(name: str) -> List[dict]:
    rows = []
    with (DERIVED_DIR / "s2" / f"{name}.jsonl").open(encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["status"] == "LABELED":
                rows.append(r)
    rows.sort(key=lambda r: (r["case_id"], r["sent_idx"]))
    return rows


def sequences(rows: List[dict]) -> "OrderedDict[str, List[dict]]":
    """Eligible sentences grouped per judgment (used for bookkeeping only; NOT for the CRF)."""
    seqs: "OrderedDict[str, List[dict]]" = OrderedDict()
    for r in rows:
        seqs.setdefault(r["case_id"], []).append(r)
    return seqs


def crf_sequences(rows: List[dict]) -> List[List[dict]]:
    """CRF sequences: maximal runs of *grid-adjacent* eligible sentences of one judgment.

    An excluded sentence (AMBIGUOUS_PROJECTION or UNLABELED) between two eligible sentences
    breaks the sequence, so the CRF never learns a transition between non-adjacent sentences.
    ``rows`` must be sorted by (case_id, sent_idx).
    """
    out: List[List[dict]] = []
    cur: List[dict] = []
    for r in rows:
        if cur and (r["case_id"] != cur[-1]["case_id"] or r["sent_idx"] != cur[-1]["sent_idx"] + 1):
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def grid_sizes(name: str) -> Dict[str, int]:
    """Number of grid sentences per judgment (all statuses), for judgment-relative position features."""
    n: Dict[str, int] = {}
    with (DERIVED_DIR / "s2" / f"{name}.jsonl").open(encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            n[r["case_id"]] = max(n.get(r["case_id"], 0), r["sent_idx"] + 1)
    return n
