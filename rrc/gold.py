"""From raw rhetorical rows to the S1 unit: the unique character-span region.

A *region* is one distinct ``[begin, end)`` boundary pair.  Rows that share a
boundary pair are collapsed into a single region whose target is the set of
their labels (multi-hot).  Structural relations between regions:

* nesting  – strict containment of one region by another (different boundaries)
* crossing – partial overlap (neither contains the other)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .io import Document, Span


@dataclass(frozen=True)
class Region:
    begin: int
    end: int
    labels: Tuple[str, ...]  # sorted, unique
    rows: Tuple[Span, ...]

    @property
    def length(self) -> int:
        return self.end - self.begin

    @property
    def is_multi(self) -> bool:
        return len(self.labels) > 1


def collapse_regions(spans: Iterable[Span]) -> List[Region]:
    """Group rhetorical rows by identical boundaries; labels become a sorted tuple."""
    buckets: Dict[Tuple[int, int], List[Span]] = {}
    for s in spans:
        if s.kind != "rhetorical":
            continue
        buckets.setdefault((s.begin, s.end), []).append(s)
    out = []
    for (b, e), rows in sorted(buckets.items()):
        labels = tuple(sorted({r.label for r in rows}))
        out.append(Region(b, e, labels, tuple(rows)))
    return out


def contains(outer: Region, inner: Region) -> bool:
    return (outer.begin <= inner.begin and inner.end <= outer.end
            and (outer.begin, outer.end) != (inner.begin, inner.end))


def nesting_pairs(regions: Sequence[Region]) -> List[Tuple[int, int]]:
    """(outer_index, inner_index) for every strict-containment pair."""
    pairs = []
    for i, a in enumerate(regions):
        for j, b in enumerate(regions):
            if i != j and contains(a, b):
                pairs.append((i, j))
    return pairs


def crossing_pairs(regions: Sequence[Region]) -> List[Tuple[int, int]]:
    """(i, j), i<j, for partially overlapping regions (neither contains the other)."""
    pairs = []
    for i in range(len(regions)):
        a = regions[i]
        for j in range(i + 1, len(regions)):
            b = regions[j]
            if a.begin < b.begin < a.end < b.end or b.begin < a.begin < b.end < a.end:
                pairs.append((i, j))
    return pairs


def shares_role(a: Region, b: Region) -> bool:
    return bool(set(a.labels) & set(b.labels))


@dataclass
class DocStructure:
    case_id: str
    regions: List[Region]
    nesting: List[Tuple[int, int]]
    crossing: List[Tuple[int, int]]

    @property
    def nested_indices(self) -> Set[int]:
        s: Set[int] = set()
        for o, i in self.nesting:
            s.add(o)
            s.add(i)
        return s

    @property
    def same_role_nesting(self) -> List[Tuple[int, int]]:
        return [(o, i) for o, i in self.nesting if shares_role(self.regions[o], self.regions[i])]

    @property
    def has_multi(self) -> bool:
        return any(r.is_multi for r in self.regions)

    @property
    def has_nesting(self) -> bool:
        return bool(self.nesting)


def doc_structure(doc: Document) -> DocStructure:
    regions = collapse_regions(doc.spans)
    return DocStructure(doc.case_id, regions, nesting_pairs(regions), crossing_pairs(regions))
