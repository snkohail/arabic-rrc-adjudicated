"""Exact / near-duplicate judgment grouping from text alone.

Two judgments are linked when their word-8-gram shingle sets have Jaccard
similarity >= 0.20 or containment (|A∩B| / min(|A|,|B|)) >= 0.50.  Linked
judgments are merged into groups by union-find.  Groups are kept inside one
split.  Thresholds are pre-declared in ``rrc.config.DEDUP``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .config import DEDUP
from .io import sha256_text

_WS = re.compile(r"\s+")


def shingles(text: str, n: int) -> Set[str]:
    words = _WS.sub(" ", text).strip().split(" ")
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(texts: Dict[str, str], params: dict = DEDUP) -> dict:
    n = params["shingle_words"]
    jt, ct = params["jaccard_threshold"], params["containment_threshold"]
    ids = sorted(texts)
    sh = {i: shingles(texts[i], n) for i in ids}
    sha = {i: sha256_text(texts[i]) for i in ids}
    uf = _UF(ids)
    pairs = []
    below: List[Tuple[float, str, str]] = []
    for x in range(len(ids)):
        a = ids[x]
        for y in range(x + 1, len(ids)):
            b = ids[y]
            inter = len(sh[a] & sh[b])
            if inter == 0:
                continue
            jac = inter / len(sh[a] | sh[b])
            cont = inter / min(len(sh[a]), len(sh[b]))
            if jac >= jt or cont >= ct:
                kind = "exact" if sha[a] == sha[b] else "near"
                pairs.append({"a": a, "b": b, "jaccard": round(jac, 4), "containment": round(cont, 4), "kind": kind})
                uf.union(a, b)
            else:
                below.append((jac, a, b))
    groups: Dict[str, List[str]] = {}
    for i in ids:
        groups.setdefault(uf.find(i), []).append(i)
    glist = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    below.sort(reverse=True)
    return {
        "params": dict(params),
        "n_documents": len(ids),
        "n_groups": len(glist),
        "linked_pairs": sorted(pairs, key=lambda p: -p["jaccard"]),
        "multi_member_groups": [g for g in glist if len(g) > 1],
        "groups": glist,
        "highest_unlinked_jaccard": [{"jaccard": round(j, 4), "a": a, "b": b} for j, a, b in below[:5]],
    }
