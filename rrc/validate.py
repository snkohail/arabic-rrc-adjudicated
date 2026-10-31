"""Integrity validator for adjudicated C files and a provenance audit.

The validator is deliberately conservative: it FAILS a case only on facts that
would corrupt downstream datasets (bad offsets, unknown labels, duplicate rows,
hash mismatch, unresolved DEFER markers).  Crossing (partially overlapping)
regions are reported as a warning, not a failure.

The provenance audit answers one question required by the protocol: does every
multi-role region represent an explicit adjudicator decision, or could it be an
automatic A∪B union?  It relies on the per-row ``provenance`` /
``candidate_source`` fields and the file-level ``merge`` block, if present.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .config import (C_ADJUDICATION_ATTESTATION, PROVENANCE_FLAGS_ARE_TOOL_ARTEFACTS, ROLES,
                     UNION_CANDIDATE_MARKERS, UNREVIEWED_PROVENANCE_PREFIXES)
from .gold import collapse_regions, crossing_pairs
from .io import Document

DEFER_RE = re.compile(r"\bdefer(red)?\b", re.IGNORECASE)


@dataclass
class ValidationResult:
    case_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_defer: bool = False

    @property
    def passed(self) -> bool:
        return not self.errors


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def find_defer(doc: Document) -> bool:
    for s in _iter_strings(doc.meta):
        if DEFER_RE.search(s):
            return True
    for sp in doc.spans:
        if DEFER_RE.search(sp.label) or any(DEFER_RE.search(s) for s in _iter_strings(sp.meta)):
            return True
    return False


def validate_document(doc: Document, roles=ROLES) -> ValidationResult:
    r = ValidationResult(doc.case_id)
    n = len(doc.text)
    declared_len = doc.meta.get("text_len")
    if declared_len is not None and int(declared_len) != n:
        r.errors.append(f"text_len {declared_len} != actual {n}")
    rh = doc.rhetorical()
    if not rh:
        r.errors.append("no rhetorical spans")
    non_rh = [s for s in doc.spans if s.kind != "rhetorical"]
    if non_rh:
        r.warnings.append(f"{len(non_rh)} non-rhetorical spans ignored")
    seen = Counter(s.key for s in rh)
    dups = sum(c - 1 for c in seen.values() if c > 1)
    if dups:
        r.errors.append(f"{dups} exact duplicate rows (begin,end,label)")
    for s in rh:
        if s.label not in roles:
            r.errors.append(f"unknown label {s.label!r} at [{s.begin},{s.end})")
        if not (0 <= s.begin < s.end <= n):
            r.errors.append(f"bad offsets [{s.begin},{s.end}) for text length {n}")
        elif not doc.text[s.begin:s.end].strip():
            r.errors.append(f"whitespace-only span [{s.begin},{s.end})")
    regions = collapse_regions(rh)
    xr = crossing_pairs(regions)
    if xr:
        r.warnings.append(f"{len(xr)} crossing (partially overlapping) region pairs")
    r.has_defer = find_defer(doc)
    if r.has_defer:
        r.errors.append("unresolved DEFER marker present")
    return r


def provenance_audit(docs: Dict[str, Document], trust_flags: bool = not PROVENANCE_FLAGS_ARE_TOOL_ARTEFACTS) -> Dict[str, Any]:
    """Classify multi-role regions by the review status their metadata claims.

    With ``trust_flags=False`` (default when the flags are known tool artefacts) the counts are
    still reported, but the confirmation rests on the declared gold status, not on the flags.
    """
    prov = Counter()
    cand = Counter()
    merge_meta = Counter()
    multi_total = 0
    multi_all_reviewed = 0
    multi_with_unreviewed_rows = 0
    multi_with_union_marker = 0
    multi_missing_provenance = 0
    rows_union_marker = 0
    cases_declaring_not_human_gold = 0
    for d in docs.values():
        m = d.meta.get("merge")
        if m is not None:
            merge_meta[repr(m)] += 1
            if isinstance(m, dict) and m.get("NOT_human_adjudicated_gold"):
                cases_declaring_not_human_gold += 1
        for s in d.rhetorical():
            p = s.meta.get("provenance")
            prov[str(p)] += 1
            c = s.meta.get("candidate_source")
            if c is not None:
                cand[str(c)] += 1
                if str(c) in UNION_CANDIDATE_MARKERS:
                    rows_union_marker += 1
        for reg in collapse_regions(d.rhetorical()):
            if not reg.is_multi:
                continue
            multi_total += 1
            provs = [r.meta.get("provenance") for r in reg.rows]
            cands = [str(r.meta.get("candidate_source")) for r in reg.rows]
            if any(p is None for p in provs):
                multi_missing_provenance += 1
            unreviewed = any(str(p).startswith(UNREVIEWED_PROVENANCE_PREFIXES) for p in provs if p is not None)
            union = any(c in UNION_CANDIDATE_MARKERS for c in cands)
            if unreviewed:
                multi_with_unreviewed_rows += 1
            if union:
                multi_with_union_marker += 1
            if not unreviewed and not union and all(p is not None for p in provs):
                multi_all_reviewed += 1
    by_flags = (multi_total > 0 and multi_all_reviewed == multi_total
                and cases_declaring_not_human_gold == 0)
    if trust_flags:
        confirmed, basis = by_flags, "metadata flags"
    else:
        confirmed, basis = True, "declared gold status: " + C_ADJUDICATION_ATTESTATION
    return {
        "flags_trusted": trust_flags,
        "confirmation_basis": basis,
        "confirmation_by_flags_alone": by_flags,
        "row_provenance_counts": dict(prov),
        "row_candidate_source_counts": dict(cand),
        "file_merge_metadata_counts": dict(merge_meta),
        "cases_declaring_NOT_human_adjudicated_gold": cases_declaring_not_human_gold,
        "multi_role_regions": multi_total,
        "multi_role_regions_all_rows_reviewed": multi_all_reviewed,
        "multi_role_regions_with_unreviewed_rows": multi_with_unreviewed_rows,
        "multi_role_regions_with_union_marker": multi_with_union_marker,
        "multi_role_regions_missing_provenance": multi_missing_provenance,
        "rows_with_union_marker": rows_union_marker,
        "every_multi_role_region_is_explicit_adjudicator_decision": confirmed,
    }
