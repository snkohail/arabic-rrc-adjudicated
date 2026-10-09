"""Project adjudicated C regions onto the frozen sentence grid by greatest character overlap.

For each sentence ``[b, e)`` and each role, coverage is the number of sentence
characters lying inside the union of all regions carrying that role (nested
same-role regions are not double-counted).  Outcomes (pre-declared, 2026-09-02):

* ``LABELED``              – exactly one role has the unique maximum coverage; the
                              sentence is eligible for S2 with that role.
* ``AMBIGUOUS_PROJECTION`` – two or more roles share the maximum coverage; the sentence
                              is excluded from S2 and reported under RQ1.  No tie-breaking.
* ``UNLABELED``            – no adjudicated span overlaps the sentence; excluded from S2.
                              Absence of annotation is not a NONE judgment.

For ambiguous sentences a structural reason is recorded from information already
present in the regions: ``co_extensive_multi_role`` (one identical-boundary region
carries all tied roles), ``nested`` (tied roles come from regions in a containment
relation), or ``other``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ROLES
from .gold import Region, doc_structure
from .io import Document
from .sentences import Grid


def _union_len(intervals: List[Tuple[int, int]]) -> int:
    total, cur_b, cur_e = 0, None, None
    for b, e in sorted(intervals):
        if cur_e is None or b > cur_e:
            if cur_e is not None:
                total += cur_e - cur_b
            cur_b, cur_e = b, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_b
    return total


def role_coverage(regions: Sequence[Region], b: int, e: int) -> Dict[str, int]:
    per_role: Dict[str, List[Tuple[int, int]]] = {}
    for r in regions:
        lo, hi = max(b, r.begin), min(e, r.end)
        if hi > lo:
            for l in r.labels:
                per_role.setdefault(l, []).append((lo, hi))
    return {l: _union_len(iv) for l, iv in per_role.items()}


@dataclass
class ProjectedSentence:
    begin: int
    end: int
    status: str                      # LABELED | AMBIGUOUS_PROJECTION | UNLABELED
    label: Optional[str]             # set only when LABELED
    roles_intersecting: List[str]
    max_roles: List[str]             # roles sharing the maximum coverage
    coverage: Dict[str, int]
    ambiguity_kind: Optional[str] = None  # co_extensive_multi_role | nested | other


def _ambiguity_kind(regions: Sequence[Region], b: int, e: int, tied: List[str]) -> str:
    hit = [r for r in regions if min(e, r.end) > max(b, r.begin)]
    best = set(tied)
    if any(best <= set(r.labels) for r in hit):
        return "co_extensive_multi_role"
    for i, a in enumerate(hit):
        for c in hit[i + 1:]:
            if (set(a.labels) | set(c.labels)) >= best and (
                (a.begin <= c.begin and c.end <= a.end) or (c.begin <= a.begin and a.end <= c.end)):
                return "nested"
    return "other"


def project_sentence(regions: Sequence[Region], b: int, e: int) -> ProjectedSentence:
    cov = role_coverage(regions, b, e)
    if not cov:
        return ProjectedSentence(b, e, "UNLABELED", None, [], [], {})
    roles = sorted(cov, key=ROLES.index)
    best = max(cov.values())
    max_roles = [r for r in roles if cov[r] == best]
    if len(max_roles) == 1:
        return ProjectedSentence(b, e, "LABELED", max_roles[0], roles, max_roles, cov)
    return ProjectedSentence(b, e, "AMBIGUOUS_PROJECTION", None, roles, max_roles, cov,
                             _ambiguity_kind(regions, b, e, max_roles))


def project_document(doc: Document, grid: Grid) -> List[ProjectedSentence]:
    regions = doc_structure(doc).regions
    return [project_sentence(regions, b, e) for b, e in grid]


def projection_stats(projected: Dict[str, List[ProjectedSentence]]) -> dict:
    allp = [p for v in projected.values() for p in v]
    n = len(allp)
    amb = [p for p in allp if p.status == "AMBIGUOUS_PROJECTION"]
    unl = [p for p in allp if p.status == "UNLABELED"]
    lab = [p for p in allp if p.status == "LABELED"]
    kinds = {"co_extensive_multi_role": 0, "nested": 0, "other": 0}
    for p in amb:
        kinds[p.ambiguity_kind] += 1
    pct = lambda k: round(100.0 * k / n, 2) if n else 0.0
    return {
        "judgments": len(projected),
        "grid_sentences": n,
        "eligible_sentences": len(lab),
        "ambiguous_projection": len(amb),
        "ambiguous_projection_pct": pct(len(amb)),
        "ambiguous_judgments_affected": sum(1 for v in projected.values() if any(p.status == "AMBIGUOUS_PROJECTION" for p in v)),
        "ambiguity_kind": kinds,
        "unlabeled_sentences": len(unl),
        "unlabeled_pct": pct(len(unl)),
        "sentences_intersecting_more_than_one_role": sum(1 for p in allp if len(p.roles_intersecting) > 1),
        "label_counts": {r: sum(1 for p in lab if p.label == r) for r in ROLES},
    }
