"""Assemble the stage-0 markdown report from committed JSON artifacts (counts only)."""
from __future__ import annotations

import json

from . import config as C


def _j(p):
    return json.loads(p.read_text())


def stage0_report() -> str:
    a = _j(C.REPORTS_DIR / "audit" / "c_gold_audit.json")
    g = _j(C.SPLITS_DIR / f"{C.GRID_VERSION}.json")
    m = _j(C.SPLITS_DIR / f"{C.SPLIT['name']}.json")
    d = _j(C.REPORTS_DIR / "datasets" / "dataset_summary.json")
    A, B, D, E, F, N = a["A_integrity"], a["B_gold_size"], a["D_multi_role"], a["E_nesting"], a["F_length_arabert_tokens"], a["normalization"]
    S = list(m["sizes"])
    L = ["# Stage 0 — frozen data artifacts (no training)", ""]
    L += ["## C integrity audit (200 judgments, after cnorm_v1)", "",
          f"- validator: **{A['validator_status']}** ({A['validator_pass']}/{A['n_cases']}); complete {A['n_complete']}; DEFER {A['n_with_DEFER']}; crossing-region judgments {A['cases_with_crossing_regions']} (warning)",
          f"- normalisation {N['version']}: rows {N['rows_before']} → {N['rows_after']} (R1 tiny {N['dropped_R1_tiny']}, R2 near-identical {N['dropped_R2_near_identical']}; {N['judgments_touched']} judgments)",
          f"- multi-role regions are explicit adjudicator decisions: **{A['provenance']['every_multi_role_region_is_explicit_adjudicator_decision']}** ({A['provenance']['confirmation_basis'].split(':')[0]})",
          f"- rows {B['raw_rhetorical_rows']}, unique regions {B['unique_span_regions']}; multi-role {D['regions_multi']} ({D['pct_regions_multi']}%) in {D['judgments_with_multi']} judgments (2: {D['by_role_count']['2']}, 3: {D['by_role_count']['3']}, >3: {D['by_role_count']['>3']})",
          f"- nesting relations {E['nested_relations']} in {E['judgments_with_nesting']} judgments; same-role {E['same_role_nested_relations']}",
          f"- AraBERT tokens per region: median {F['median']:.0f}, p95 {F['p95']:.0f}, max {F['max']}; >510: {F['n_over_limit']} ({F['pct_over_limit']}%)", ""]
    L += ["## Exact duplicate", "", *[f"- excluded from modeling: **{k}** — {v}" for k, v in m["excluded_from_modeling"].items()],
          f"- modeling corpus: {len(m['text_sha256'])} unique-text judgments", ""]
    L += ["## Frozen sentence grid", "", f"- `{g['name']}` status {g['status']}, {g['n_sentences']} sentences over {g['n_judgments']} judgments",
          f"- grid_sha256 `{g['grid_sha256']}`", f"- params: paragraph break `{g['params']['paragraph_break_regex']}`, terminator `{g['params']['sentence_terminator_regex']}`, single line break = soft wrap, keep if ≥1 letter/digit", ""]
    L += ["## Frozen split", "", f"- `{m['name']}` status **{m['status']}**, seed {m['seed']}, sizes {m['sizes']}",
          f"- manifest_sha256 `{m['manifest_sha256']}`",
          f"- indicators: {', '.join(m['indicators'])}",
          f"- volume bins: LOW < {m['volume_bin_boundaries']['low_lt']} ≤ MED < {m['volume_bin_boundaries']['high_ge']} ≤ HIGH regions/judgment",
          f"- related groups kept together: {m['related_groups']}", ""]
    L += ["| indicator | " + " | ".join(S) + " |", "|---|" + "---:|" * len(S)]
    for k in m["indicators"]:
        L.append(f"| {k} | " + " | ".join(str(m["indicator_counts"][s][k]) for s in S) + " |")
    L += [""]
    s1, s2 = d["s1"], d["s2"]
    L += ["## S1 — original-span multi-label", "", "| | " + " | ".join(S) + " |", "|---|" + "---:|" * len(S),
          "| judgments | " + " | ".join(str(s1[s]["judgments"]) for s in S) + " |",
          "| examples | " + " | ".join(str(s1[s]["examples"]) for s in S) + " |",
          "| multi-role n (%) | " + " | ".join(f"{s1[s]['multi_role']['n']} ({s1[s]['multi_role']['pct']}%)" for s in S) + " |",
          "| multi-role by #labels | " + " | ".join(str(s1[s]["multi_role"]["by_n_labels"]) for s in S) + " |",
          "| nested n (%) | " + " | ".join(f"{s1[s]['nested']['n']} ({s1[s]['nested']['pct']}%)" for s in S) + " |",
          "| >510 AraBERT tokens | " + " | ".join(str(s1[s].get("over_510_tokens", "-")) for s in S) + " |",
          "| regions/judgment min / median / max | " + " | ".join(f"{s1[s]['region_volume']['per_judgment_min']} / {s1[s]['region_volume']['per_judgment_median']} / {s1[s]['region_volume']['per_judgment_max']}" for s in S) + " |",
          "| volume bins LOW/MED/HIGH | " + " | ".join("/".join(str(s1[s]["region_volume"]["bins"][b]) for b in C.SPLIT["volume_bins"]) for s in S) + " |", ""]
    L += ["Role distribution (S1 label occurrences):", "", "| role | " + " | ".join(S) + " |", "|---|" + "---:|" * len(S)]
    for r in C.ROLES:
        L.append(f"| {r} | " + " | ".join(str(s1[s]["label_counts"][r]) for s in S) + " |")
    L += [""]
    L += ["## S2 — derived unambiguous single-label sentence benchmark", "",
          "Labels are projections of adjudicated spans onto the frozen grid, not independently annotated sentence gold.", "",
          "| | " + " | ".join(S) + " |", "|---|" + "---:|" * len(S),
          "| judgments | " + " | ".join(str(s2[s]["judgments"]) for s in S) + " |",
          "| grid sentences | " + " | ".join(str(s2[s]["grid_sentences"]) for s in S) + " |",
          "| eligible (LABELED) n (%) | " + " | ".join(f"{s2[s]['eligible_sentences']} ({s2[s]['eligible_pct']}%)" for s in S) + " |",
          "| AMBIGUOUS_PROJECTION n (%) | " + " | ".join(f"{s2[s]['ambiguous_projection']['n']} ({s2[s]['ambiguous_projection']['pct']}%)" for s in S) + " |",
          "| ambiguous: judgments affected | " + " | ".join(str(s2[s]["ambiguous_projection"]["judgments_affected"]) for s in S) + " |",
          "| ambiguous by kind | " + " | ".join(str(s2[s]["ambiguous_projection"]["kind"]) for s in S) + " |",
          "| UNLABELED n (%) | " + " | ".join(f"{s2[s]['unlabeled']['n']} ({s2[s]['unlabeled']['pct']}%)" for s in S) + " |", ""]
    L += ["Role distribution (S2 eligible sentences):", "", "| role | " + " | ".join(S) + " |", "|---|" + "---:|" * len(S)]
    for r in C.ROLES:
        L.append(f"| {r} | " + " | ".join(str(s2[s]["label_counts"][r]) for s in S) + " |")
    L += [""]
    return "\n".join(L)
