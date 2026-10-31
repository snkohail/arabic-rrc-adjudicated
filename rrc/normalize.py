"""Minimal, deterministic normalisation of C rows (``cnorm_v1``).

Applied in code at load time; the annotation folder is never modified.  Only two
mechanical artefacts are repaired (pre-declared, 2026-09-02).  Adjudicated
labels are otherwise left untouched; in particular a label that is also carried
by an enclosing region is KEPT as adjudicated.

R1  tiny rows: a rhetorical row is dropped if its span is shorter than
    ``min_span_chars`` characters or contains no letter.
R2  near-identical same-label duplicates: if two rows carry the same label, one
    span contains the other, and their lengths differ by at most
    ``near_identical_max_len_diff`` characters, the inner (shorter) row is dropped
    and the outer one kept.

R1 runs before R2.  Every dropped row is logged (case id, offsets, label, rule,
and for R2 the kept span), without text.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Tuple

from .config import C_NORMALIZATION
from .io import Document, Span

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def normalize_document(doc: Document, params: dict = C_NORMALIZATION) -> Tuple[Document, List[dict]]:
    log: List[dict] = []
    rows = list(doc.spans)
    keep: List[Span] = []
    # R1
    for s in rows:
        if s.kind != "rhetorical":
            keep.append(s)
            continue
        txt = doc.text[s.begin:s.end]
        if (s.end - s.begin) < params["min_span_chars"] or not _LETTER.search(txt):
            log.append({"case_id": doc.case_id, "begin": s.begin, "end": s.end, "label": s.label, "rule": "R1_tiny"})
        else:
            keep.append(s)
    # R2
    rh = [s for s in keep if s.kind == "rhetorical"]
    by_label: Dict[str, List[Span]] = {}
    for s in rh:
        by_label.setdefault(s.label, []).append(s)
    drop = set()
    for label, group in by_label.items():
        group = sorted(group, key=lambda s: (s.begin, -s.end))
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if b.begin >= a.end:
                    break
                inner, outer = (b, a) if (a.begin <= b.begin and b.end <= a.end) else ((a, b) if (b.begin <= a.begin and a.end <= b.end) else (None, None))
                if inner is None or inner.key == outer.key:
                    continue
                if (outer.end - outer.begin) - (inner.end - inner.begin) <= params["near_identical_max_len_diff"]:
                    if inner.key not in drop:
                        drop.add(inner.key)
                        log.append({"case_id": doc.case_id, "begin": inner.begin, "end": inner.end, "label": label,
                                    "rule": "R2_near_identical", "kept_begin": outer.begin, "kept_end": outer.end})
    out = [s for s in keep if s.kind != "rhetorical" or s.key not in drop]
    meta = dict(doc.meta)
    meta["normalization"] = {"version": params["version"], "dropped_rows": len(log)}
    return replace(doc, spans=out, meta=meta), log


def normalize_corpus(docs: Dict[str, Document], params: dict = C_NORMALIZATION) -> Tuple[Dict[str, Document], dict]:
    out: Dict[str, Document] = {}
    log: List[dict] = []
    for cid in sorted(docs):
        nd, l = normalize_document(docs[cid], params)
        out[cid] = nd
        log.extend(l)
    summary = {
        "version": params["version"],
        "params": dict(params),
        "rows_before": sum(len(d.rhetorical()) for d in docs.values()),
        "rows_after": sum(len(d.rhetorical()) for d in out.values()),
        "dropped_R1_tiny": sum(1 for x in log if x["rule"] == "R1_tiny"),
        "dropped_R2_near_identical": sum(1 for x in log if x["rule"] == "R2_near_identical"),
        "judgments_touched": len({x["case_id"] for x in log}),
        "log": log,
    }
    return out, summary
