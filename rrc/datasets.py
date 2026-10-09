"""Build the S1 (span, multi-label) and S2 (derived unambiguous single-label sentence) datasets.

Outputs go to the private data dir (git-ignored, contains text).  Count summaries
go to ``reports/datasets/`` (committed).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .config import DERIVED_DIR, ROLES
from .gold import doc_structure
from .io import Document
from .projection import project_document
from .sentences import sentence_grid
from .split import VOLUME_BINS, split_of, volume_bin
from .tokens import token_lengths


def _write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_s1(docs: Dict[str, Document], manifest: dict, with_tokens: bool = True) -> Dict[str, List[dict]]:
    sp = split_of(manifest)
    out: Dict[str, List[dict]] = {s: [] for s in manifest["sizes"]}
    for cid in sorted(docs):
        d = docs[cid]
        st = doc_structure(d)
        nested = st.nested_indices
        outer = {o for o, _ in st.nesting}
        inner = {i for _, i in st.nesting}
        for k, r in enumerate(st.regions):
            out[sp[cid]].append({
                "id": f"{cid}:{r.begin}-{r.end}",
                "case_id": cid,
                "begin": r.begin,
                "end": r.end,
                "text": d.text[r.begin:r.end],
                "labels": list(r.labels),
                "n_labels": len(r.labels),
                "is_multi": r.is_multi,
                "is_nested": k in nested,
                "nesting_role": ("both" if k in outer and k in inner else "outer" if k in outer
                                 else "inner" if k in inner else "none"),
            })
    if with_tokens:
        for rows in out.values():
            L = token_lengths([r["text"] for r in rows])
            for r, n in zip(rows, L):
                r["n_tokens_arabert"] = n
    return out


def build_s2(docs: Dict[str, Document], manifest: dict) -> Dict[str, List[dict]]:
    """All grid sentences with their projection status; S2 modeling uses status == LABELED only."""
    sp = split_of(manifest)
    out: Dict[str, List[dict]] = {s: [] for s in manifest["sizes"]}
    for cid in sorted(docs):
        d = docs[cid]
        for i, p in enumerate(project_document(d, sentence_grid(d.text))):
            out[sp[cid]].append({
                "id": f"{cid}:s{i:04d}",
                "case_id": cid,
                "sent_idx": i,
                "begin": p.begin,
                "end": p.end,
                "text": d.text[p.begin:p.end],
                "status": p.status,
                "label": p.label,
                "roles_intersecting": p.roles_intersecting,
                "max_roles": p.max_roles,
                "ambiguity_kind": p.ambiguity_kind,
            })
    return out


def s1_summary(s1: Dict[str, List[dict]], manifest: dict) -> dict:
    summ = {}
    bounds = (manifest["volume_bin_boundaries"]["low_lt"], manifest["volume_bin_boundaries"]["high_ge"])
    for s, rows in s1.items():
        c = Counter(l for r in rows for l in r["labels"])
        per_doc = Counter(r["case_id"] for r in rows)
        n = len(rows)
        multi = [r for r in rows if r["is_multi"]]
        summ[s] = {
            "judgments": len(per_doc),
            "examples": n,
            "multi_role": {"n": len(multi), "pct": round(100.0 * len(multi) / n, 2),
                           "by_n_labels": {str(k): v for k, v in sorted(Counter(r["n_labels"] for r in multi).items())}},
            "nested": {"n": sum(r["is_nested"] for r in rows), "pct": round(100.0 * sum(r["is_nested"] for r in rows) / n, 2)},
            "label_counts": {l: c[l] for l in ROLES},
            "region_volume": {"regions_total": n, "per_judgment_min": min(per_doc.values()),
                              "per_judgment_median": sorted(per_doc.values())[len(per_doc) // 2],
                              "per_judgment_max": max(per_doc.values()),
                              "bins": {b: sum(1 for v in per_doc.values() if volume_bin(v, bounds) == b) for b in VOLUME_BINS}},
        }
        if rows and "n_tokens_arabert" in rows[0]:
            summ[s]["over_510_tokens"] = sum(r["n_tokens_arabert"] > 510 for r in rows)
    return summ


def s2_summary(s2: Dict[str, List[dict]]) -> dict:
    summ = {}
    for s, rows in s2.items():
        n = len(rows)
        amb = [r for r in rows if r["status"] == "AMBIGUOUS_PROJECTION"]
        unl = [r for r in rows if r["status"] == "UNLABELED"]
        lab = [r for r in rows if r["status"] == "LABELED"]
        summ[s] = {
            "judgments": len({r["case_id"] for r in rows}),
            "grid_sentences": n,
            "eligible_sentences": len(lab),
            "eligible_pct": round(100.0 * len(lab) / n, 2),
            "ambiguous_projection": {"n": len(amb), "pct": round(100.0 * len(amb) / n, 2),
                                     "judgments_affected": len({r["case_id"] for r in amb}),
                                     "kind": dict(Counter(r["ambiguity_kind"] for r in amb))},
            "unlabeled": {"n": len(unl), "pct": round(100.0 * len(unl) / n, 2)},
            "label_counts": {l: sum(r["label"] == l for r in lab) for l in ROLES},
        }
    return summ


def write_datasets(s1, s2, root: Path = DERIVED_DIR) -> Dict[str, str]:
    paths = {}
    for s, rows in s1.items():
        p = root / "s1" / f"{s}.jsonl"; _write_jsonl(rows, p); paths[f"s1/{s}"] = str(p)
    for s, rows in s2.items():
        p = root / "s2" / f"{s}.jsonl"; _write_jsonl(rows, p); paths[f"s2/{s}"] = str(p)
    return paths
