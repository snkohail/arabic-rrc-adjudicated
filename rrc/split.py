"""Deterministic, group-aware, lightly stratified document split — FROZEN once written.

Unit of assignment: a duplicate/related group (usually a single judgment).
Stratification indicators per judgment (binary, split attributes only, never
model inputs): presence of each of the nine roles, presence of at least one
multi-role region, and a coarse regions-per-judgment tercile bin
(VOLUME_LOW / VOLUME_MED / VOLUME_HIGH) whose boundaries are the 1/3 and 2/3
order statistics of the modeling corpus.  The algorithm is iterative
stratification (Sechidis et al., 2011) applied to groups: repeatedly take the
indicator with the fewest unassigned positives and place each remaining group
carrying it into the split whose desired share of that indicator is largest
(ties: larger remaining capacity, then seeded RNG).  Generated once with seed
13; nothing about model performance enters the procedure.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import C_NORMALIZATION, EXCLUDED_FROM_MODELING, ROLES, SPLIT
from .gold import doc_structure
from .io import Document

VOLUME_BINS = SPLIT["volume_bins"]
INDICATORS = ROLES + ["MULTI_ROLE_DOC"] + VOLUME_BINS


def region_counts(docs: Dict[str, Document]) -> Dict[str, int]:
    return {cid: len(doc_structure(d).regions) for cid, d in docs.items()}


def volume_boundaries(counts: Dict[str, int]) -> Tuple[int, int]:
    """Tercile boundaries: low < b1 <= med < b2 <= high (order statistics, deterministic)."""
    xs = sorted(counts.values())
    n = len(xs)
    return xs[n // 3], xs[(2 * n) // 3]


def volume_bin(count: int, bounds: Tuple[int, int]) -> str:
    b1, b2 = bounds
    return VOLUME_BINS[0] if count < b1 else VOLUME_BINS[1] if count < b2 else VOLUME_BINS[2]


def doc_indicators(docs: Dict[str, Document]) -> Tuple[Dict[str, Dict[str, int]], Tuple[int, int], Dict[str, int]]:
    counts = region_counts(docs)
    bounds = volume_boundaries(counts)
    out = {}
    for cid, d in docs.items():
        st = doc_structure(d)
        present = {l for r in st.regions for l in r.labels}
        ind = {r: int(r in present) for r in ROLES}
        ind["MULTI_ROLE_DOC"] = int(st.has_multi)
        vb = volume_bin(counts[cid], bounds)
        for b in VOLUME_BINS:
            ind[b] = int(b == vb)
        out[cid] = ind
    return out, bounds, counts


def stratified_group_split(indicators: Dict[str, Dict[str, int]], groups: Sequence[Sequence[str]],
                           sizes: Dict[str, int], seed: int) -> Dict[str, str]:
    ids = set(indicators)
    covered = [i for g in groups for i in g]
    if sorted(covered) != sorted(ids):
        raise ValueError("groups must partition the document ids exactly")
    if sum(sizes.values()) != len(ids):
        raise ValueError(f"split sizes {sizes} do not sum to {len(ids)} documents")
    rng = random.Random(seed)
    splits = list(sizes)
    N = len(ids)
    labels = list(next(iter(indicators.values())).keys())
    total = Counter()
    for i in ids:
        total.update({k: v for k, v in indicators[i].items() if v})
    desired = {s: {k: total[k] * sizes[s] / N for k in labels} for s in splits}
    capacity = dict(sizes)
    order = [tuple(g) for g in groups]
    rng.shuffle(order)
    glabels = {}
    for g in order:
        c = Counter()
        for i in g:
            c.update({k: v for k, v in indicators[i].items() if v})
        glabels[g] = c
    unassigned = set(order)
    assign: Dict[str, str] = {}

    def place(g, label):
        feasible = [s for s in splits if capacity[s] >= len(g)]
        if not feasible:
            raise RuntimeError("no split has capacity left")
        if label is None:
            key = lambda s: (capacity[s], rng.random())
        else:
            key = lambda s: (desired[s][label], capacity[s], rng.random())
        best = max(feasible, key=key)
        for i in g:
            assign[i] = best
        for k, v in glabels[g].items():
            desired[best][k] -= v
        capacity[best] -= len(g)
        unassigned.discard(g)

    while unassigned:
        remaining = Counter()
        for g in unassigned:
            remaining.update(glabels[g])
        cands = [k for k in labels if remaining[k] > 0]
        if not cands:
            for g in sorted(unassigned, key=lambda g: (-len(g), g)):
                place(g, None)
            break
        label = min(cands, key=lambda k: (remaining[k], labels.index(k)))
        for g in [g for g in order if g in unassigned and glabels[g][label] > 0]:
            place(g, label)
    return assign


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def manifest_hash(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "created_utc")}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def build_manifest(docs: Dict[str, Document], dedup: dict, status: str = "FROZEN",
                   params: dict = SPLIT, grid_sha256: str = "", note: str = "") -> dict:
    """``docs`` is the modeling corpus (excluded duplicates already removed)."""
    ind, bounds, counts = doc_indicators(docs)
    groups = [[i for i in g if i in docs] for g in dedup["groups"]]
    groups = [g for g in groups if g]
    assign = stratified_group_split(ind, groups, params["sizes"], params["seed"])
    splits = {s: sorted(i for i, a in assign.items() if a == s) for s in params["sizes"]}
    ind_counts = {s: {k: sum(ind[i][k] for i in ids) for k in INDICATORS} for s, ids in splits.items()}
    volume = {s: {"judgments": len(ids), "regions_total": sum(counts[i] for i in ids),
                  "regions_min": min(counts[i] for i in ids), "regions_median": sorted(counts[i] for i in ids)[len(ids) // 2],
                  "regions_max": max(counts[i] for i in ids)} for s, ids in splits.items()}
    man = {
        "name": params["name"],
        "status": status,
        "seed": params["seed"],
        "sizes": dict(params["sizes"]),
        "method": ("group-aware iterative stratification over document-level indicators "
                   "(9 role-presence flags + MULTI_ROLE_DOC + regions-per-judgment tercile bin); "
                   "tie-breaks by capacity then seeded RNG; generated once; no model-based selection"),
        "unit": "judgment (canonical_case_id); duplicate/related groups never straddle splits",
        "indicators": INDICATORS,
        "volume_bin_boundaries": {"low_lt": bounds[0], "high_ge": bounds[1], "rule": "count < b1 -> LOW; b1 <= count < b2 -> MED; count >= b2 -> HIGH; b1,b2 = 1/3 and 2/3 order statistics"},
        "c_normalization": C_NORMALIZATION["version"],
        "grid_sha256": grid_sha256,
        "excluded_from_modeling": dict(EXCLUDED_FROM_MODELING),
        "dedup_params": dedup["params"],
        "related_groups": [g for g in groups if len(g) > 1],
        "note": note,
        "text_sha256": {cid: d.text_sha256 for cid, d in sorted(docs.items())},
        "regions_per_judgment": {cid: counts[cid] for cid in sorted(docs)},
        "splits": splits,
        "indicator_counts": ind_counts,
        "region_volume": volume,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    man["manifest_sha256"] = manifest_hash(man)
    return man


def write_manifest(man: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(man["manifest_sha256"] + "  " + path.name + "\n")


def load_manifest(path: Path, verify: bool = True) -> dict:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    if verify and manifest_hash(man) != man["manifest_sha256"]:
        raise ValueError(f"manifest hash mismatch: {path}")
    return man


def split_of(man: dict) -> Dict[str, str]:
    return {cid: s for s, ids in man["splits"].items() for cid in ids}


def check_manifest_against_docs(man: dict, docs: Dict[str, Document]) -> List[str]:
    """Return a list of problems (empty == consistent). ``docs`` = modeling corpus."""
    problems = []
    ids = set(docs)
    listed = set(split_of(man))
    if ids != listed:
        problems.append(f"id sets differ: {sorted(ids ^ listed)[:10]}")
    for cid in man["excluded_from_modeling"]:
        if cid in listed:
            problems.append(f"excluded id {cid} appears in a split")
    for cid, sha in man["text_sha256"].items():
        if cid in docs and docs[cid].text_sha256 != sha:
            problems.append(f"text changed for {cid}")
    for s, n in man["sizes"].items():
        if len(man["splits"][s]) != n:
            problems.append(f"{s} has {len(man['splits'][s])} != {n}")
    so = split_of(man)
    for g in man["related_groups"]:
        if len({so[i] for i in g}) != 1:
            problems.append(f"group {g} straddles splits")
    if docs and man.get("regions_per_judgment"):
        rc = region_counts(docs)
        bad = [c for c in rc if man["regions_per_judgment"].get(c) != rc[c]]
        if bad:
            problems.append(f"region counts changed for {bad[:5]}")
    return problems
