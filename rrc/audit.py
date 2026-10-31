"""Minimal C-gold audit (protocol section 2, items A–F). Nothing else is computed."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

from .config import C_NORMALIZATION, DATA_DIR, ENCODER_INPUT_LIMIT, ROLES, TOKENIZER_ARABERT
from .gold import doc_structure
from .io import Document
from .normalize import normalize_corpus
from .tokens import token_lengths
from .validate import provenance_audit, validate_document


def run_audit(raw_docs: Dict[str, Document], c_source: str, with_tokens: bool = True) -> dict:
    """``raw_docs`` must be the un-normalised C; normalisation is applied and logged here."""
    docs, norm = normalize_corpus(raw_docs)
    norm_summary = {k: v for k, v in norm.items() if k != "log"}
    norm_summary["rules"] = {
        "R1_tiny": f"drop rhetorical row if (end - begin) < {C_NORMALIZATION['min_span_chars']} characters "
                   f"or the span contains no Unicode letter (regex [^\\W\\d_])",
        "R2_near_identical": f"for two rows with the same label where one span contains the other and the length "
                             f"difference is <= {C_NORMALIZATION['near_identical_max_len_diff']} characters, drop the inner "
                             f"(shorter) row and keep the outer; R1 runs before R2",
        "not_applied": "labels duplicating an enclosing region's label are kept as adjudicated",
    }
    # private, traceable row-level log (adds the removed span text; never committed)
    norm_log = []
    for x in norm["log"]:
        d = raw_docs[x["case_id"]]
        row = dict(x)
        row["text"] = d.text[x["begin"]:x["end"]]
        row["reason"] = ("shorter than %d chars or no letter" % C_NORMALIZATION["min_span_chars"] if x["rule"] == "R1_tiny"
                         else "inner copy of same-label span [%d,%d) differing by <= %d chars" % (x["kept_begin"], x["kept_end"], C_NORMALIZATION["near_identical_max_len_diff"]))
        norm_log.append(row)
    # ---- A. integrity
    results = {cid: validate_document(d) for cid, d in docs.items()}
    n_pass = sum(r.passed for r in results.values())
    n_defer = sum(r.has_defer for r in results.values())
    prov = provenance_audit(docs)
    integrity = {
        "c_source": c_source,
        "n_cases": len(docs),
        "n_complete": sum(1 for d in docs.values() if d.rhetorical()),
        "n_with_DEFER": n_defer,
        "validator_pass": n_pass,
        "validator_fail": len(docs) - n_pass,
        "validator_status": "PASS" if n_pass == len(docs) else "FAIL",
        "failing_cases": {cid: r.errors for cid, r in results.items() if not r.passed},
        "warnings_total": sum(len(r.warnings) for r in results.values()),
        "cases_with_crossing_regions": sum(1 for r in results.values() if any("crossing" in w for w in r.warnings)),
        "provenance": prov,
    }
    # ---- B–E. gold size, roles, multi-role, nesting
    structs = {cid: doc_structure(d) for cid, d in docs.items()}
    n_rows = sum(len(d.rhetorical()) for d in docs.values())
    regions = [r for s in structs.values() for r in s.regions]
    n_regions = len(regions)
    role_rows = Counter()
    for d in docs.values():
        for s in d.rhetorical():
            role_rows[s.label] += 1
    role_regions = Counter(l for r in regions for l in r.labels)
    multi = [r for r in regions if r.is_multi]
    by_k = Counter(len(r.labels) for r in multi)
    combos = Counter("+".join(r.labels) for r in multi)
    nest_pairs = sum(len(s.nesting) for s in structs.values())
    nest_docs = sum(1 for s in structs.values() if s.has_nesting)
    same_role = sum(len(s.same_role_nesting) for s in structs.values())
    nested_regions = sum(len(s.nested_indices) for s in structs.values())
    # ---- F. length
    length = {"tokenizer": TOKENIZER_ARABERT, "note": "raw text, no Farasa pre-segmentation, no special tokens"}
    if with_tokens:
        texts = [docs[s.case_id].text[r.begin:r.end] for s in structs.values() for r in s.regions]
        L = np.array(token_lengths(texts), dtype=int)
        length.update({
            "median": float(np.median(L)),
            "p95": float(np.percentile(L, 95)),
            "max": int(L.max()),
            "n_over_limit": int((L > ENCODER_INPUT_LIMIT).sum()),
            "pct_over_limit": round(100.0 * float((L > ENCODER_INPUT_LIMIT).mean()), 2),
            "limit": ENCODER_INPUT_LIMIT,
        })
    return {
        "normalization": norm_summary,
        "normalization_log": norm_log,
        "A_integrity": integrity,
        "B_gold_size": {"raw_rhetorical_rows": n_rows, "unique_span_regions": n_regions},
        "C_role_counts": {"rows": {r: role_rows[r] for r in ROLES}, "regions_carrying_role": {r: role_regions[r] for r in ROLES}},
        "D_multi_role": {
            "regions_multi": len(multi),
            "pct_regions_multi": round(100.0 * len(multi) / max(1, n_regions), 2),
            "judgments_with_multi": sum(1 for s in structs.values() if s.has_multi),
            "by_role_count": {"2": by_k[2], "3": by_k[3], ">3": sum(v for k, v in by_k.items() if k > 3)},
            "top_combinations": combos.most_common(10),
        },
        "E_nesting": {
            "nested_relations": nest_pairs,
            "judgments_with_nesting": nest_docs,
            "same_role_nested_relations": same_role,
            "regions_participating_in_nesting": nested_regions,
        },
        "F_length_arabert_tokens": length,
    }


def audit_markdown(a: dict) -> str:
    A, B, Cc, D, E, F = (a[k] for k in ("A_integrity", "B_gold_size", "C_role_counts", "D_multi_role", "E_nesting", "F_length_arabert_tokens"))
    P = A["provenance"]
    N = a["normalization"]
    lines = ["# Minimal C-gold audit", "", f"C source: `{A['c_source']}`", "",
             f"Normalisation `{N['version']}` (applied in code, annotation files untouched): rows {N['rows_before']} -> {N['rows_after']}; "
             f"dropped R1 tiny (<{N['params']['min_span_chars']} chars or no letter) = {N['dropped_R1_tiny']}, "
             f"R2 near-identical same-label duplicates (len diff <= {N['params']['near_identical_max_len_diff']}) = {N['dropped_R2_near_identical']}; "
             f"judgments touched = {N['judgments_touched']}. Inherited labels kept as adjudicated.",
             "", "Rules:", *[f"- `{k}`: {v}" for k, v in N["rules"].items()],
             f"- row-level log with case id, offsets, label, rule, reason and removed text: PRIVATE artifact `{N.get('private_row_log', '<data_dir>/audit/c_normalization_log.json')}` (not in Git)", ""]
    lines += ["## A. Integrity", "",
              f"- cases: {A['n_cases']}; complete (>=1 rhetorical span): {A['n_complete']}; with DEFER: {A['n_with_DEFER']}",
              f"- validator: **{A['validator_status']}** ({A['validator_pass']} pass / {A['validator_fail']} fail)",
              f"- cases with crossing (partially overlapping) regions: {A['cases_with_crossing_regions']} (warning only)",
              f"- multi-role confirmation basis: {P['confirmation_basis']}",
              f"- metadata flags trusted: {P['flags_trusted']} (files declaring `NOT_human_adjudicated_gold`: "
              f"{P['cases_declaring_NOT_human_adjudicated_gold']}; confirmation by flags alone would be {P['confirmation_by_flags_alone']})",
              f"- row provenance (tool artefact counts): {P['row_provenance_counts']}",
              f"- row candidate_source: {P['row_candidate_source_counts']}",
              f"- multi-role regions: {P['multi_role_regions']}; all rows reviewed: {P['multi_role_regions_all_rows_reviewed']}; "
              f"with unreviewed rows: {P['multi_role_regions_with_unreviewed_rows']}; with UNION marker: {P['multi_role_regions_with_union_marker']}; "
              f"missing provenance: {P['multi_role_regions_missing_provenance']}",
              f"- every multi-role region is an explicit adjudicator decision: **{P['every_multi_role_region_is_explicit_adjudicator_decision']}**", ""]
    if A["failing_cases"]:
        lines += ["Failing cases:", ""] + [f"- {c}: {'; '.join(e)}" for c, e in A["failing_cases"].items()] + [""]
    lines += ["## B. Gold size", "", f"- raw rhetorical rows: {B['raw_rhetorical_rows']}", f"- unique span regions: {B['unique_span_regions']}", ""]
    lines += ["## C. Nine-role counts (rows / regions carrying role)", "", "| role | rows | regions |", "|---|---:|---:|"]
    lines += [f"| {r} | {Cc['rows'][r]} | {Cc['regions_carrying_role'][r]} |" for r in ROLES] + [""]
    lines += ["## D. Multi-role structure", "",
              f"- regions with >1 role: {D['regions_multi']} ({D['pct_regions_multi']}% of regions)",
              f"- judgments with >=1 multi-role region: {D['judgments_with_multi']}",
              f"- by role count: 2 = {D['by_role_count']['2']}, 3 = {D['by_role_count']['3']}, >3 = {D['by_role_count']['>3']}", ""]
    lines += ["## E. Nesting", "", f"- nested span relations: {E['nested_relations']}",
              f"- judgments with nesting: {E['judgments_with_nesting']}", f"- same-role nested relations: {E['same_role_nested_relations']}", ""]
    lines += ["## F. Length (AraBERTv2 tokens, unique regions)", ""]
    if "median" in F:
        lines += [f"- median {F['median']:.0f}, p95 {F['p95']:.0f}, max {F['max']}",
                  f"- exceeding {F['limit']} tokens: {F['n_over_limit']} ({F['pct_over_limit']}%)"]
    lines += [f"- {F['note']}", ""]
    return "\n".join(lines)


def write_audit(a: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = a.pop("normalization_log")
    private = DATA_DIR / "audit"
    private.mkdir(parents=True, exist_ok=True)
    (private / "c_normalization_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    a["normalization"]["private_row_log"] = "<data_dir>/audit/c_normalization_log.json"
    (out_dir / "c_gold_audit.json").write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "c_gold_audit.md").write_text(audit_markdown(a), encoding="utf-8")
