"""Run the pipeline's data-side tools on the SYNTHETIC mock folder and render a mock expert-audit page.

Steps: load the three annotator folders -> integrity validation -> structural counts (regions, multi-role,
nesting, crossing) -> grid_v1 sentence projection with LABELED / AMBIGUOUS_PROJECTION / UNLABELED -> check the
16 showcase items -> render the blind expert-audit interface for them (examples/mock/output/).

Usage: python examples/mock/run_demo.py [annotation_dir]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from rrc.expert_audit import STRATA, _window, build_context_ext, render_html, role_definitions   # noqa: E402
from rrc.gold import doc_structure                                                              # noqa: E402
from rrc.io import load_c_only, load_corpus                                                      # noqa: E402
from rrc.projection import project_document                                                       # noqa: E402
from rrc.sentences import sentence_grid                                                            # noqa: E402
from rrc.validate import validate_document                                                         # noqa: E402


def main(root: Path) -> None:
    corpus = load_corpus(root)                      # A, B (blind, text via sha256) and C all load
    docs = load_c_only(root, normalize=True)        # C with cnorm_v1, as in the experiments
    print(f"loaded {len(corpus.C)} mock judgments (A/B/C); C after cnorm_v1: {len(docs)}\n")

    print("integrity validation (C)")
    for cid, d in docs.items():
        r = validate_document(d)
        print(f"  {cid}: {'PASS' if r.passed else 'FAIL ' + '; '.join(r.errors)}" + (f"  warnings: {'; '.join(r.warnings)}" if r.warnings else ""))

    print("\nstructure (C)")
    for cid, d in docs.items():
        S = doc_structure(d)
        print(f"  {cid}: regions {len(S.regions)}, multi-role {sum(r.is_multi for r in S.regions)}, nesting pairs {len(S.nesting)}, crossing pairs {len(S.crossing)}")

    print("\ngrid_v1 projection (per sentence)")
    proj = {}
    for cid, d in docs.items():
        P = project_document(d, sentence_grid(d.text)); proj[cid] = P
        print(f"  {cid}: " + ", ".join(f"s{i}={p.status if p.status != 'LABELED' else p.label}" + (f"({p.ambiguity_kind})" if p.ambiguity_kind else "") for i, p in enumerate(P)))

    items_spec = json.loads((HERE / "mock_items.json").read_text(encoding="utf-8"))["items"]
    print(f"\nshowcase items ({len(items_spec)})")
    ok = 0
    for it in items_spec:
        d = docs[it["case_id"]]; S = doc_structure(d); byb = {(r.begin, r.end): r for r in S.regions}
        if it["form"] == 1:
            r = byb[(it["begin"], it["end"])]; good = len(r.labels) == 1 and S.regions.index(r) not in S.nested_indices
        elif it["form"] == 2:
            good = len(byb[(it["begin"], it["end"])].labels) > 1
        elif it["form"] == 3:
            o = S.regions.index(byb[tuple(it["outer"])]); i = S.regions.index(byb[tuple(it["inner"])]); good = (o, i) in S.nesting
        else:
            p = proj[it["case_id"]][it["sent_idx"]]; good = p.status == "AMBIGUOUS_PROJECTION" and (p.begin, p.end) == (it["begin"], it["end"])
        ok += good
        print(f"  form {it['form']} {it['name']:<30} {it['case_id']} {it.get('begin', it.get('outer'))}-{it.get('end', it.get('inner'))}  {'ok' if good else 'MISMATCH'}")
    print(f"  {ok}/{len(items_spec)} items show the intended form")

    # blind expert-audit page for the 16 items (same renderer as the real instrument; synthetic text only)
    items, key = [], []
    for k, it in enumerate(items_spec):
        text = docs[it["case_id"]].text; s = it["form"]
        base = {"item_id": f"E{s}-{k // 4 + 1:02d}", "stratum": s, "stratum_name": STRATA[s], "case_id": it["case_id"], "display_order": k}
        if s == 3:
            (ob, oe), (ib, ie) = it["outer"], it["inner"]; pre, _, post = _window(text, ob, oe, 200)
            items.append({**base, "outer": [ob, oe], "inner": [ib, ie], "pre": pre, "post": post, "outer_text": text[ob:oe], "inner_offset_in_outer": [ib - ob, ie - ob]})
            key.append({**base, "c_outer_labels": it["outer_labels"], "c_inner_labels": it["inner_labels"]})
        else:
            pre, sp, post = _window(text, it["begin"], it["end"])
            items.append({**base, "begin": it["begin"], "end": it["end"], "pre": pre, "span": sp, "post": post, **({"sent_idx": it["sent_idx"]} if s == 4 else {})})
            key.append({**base, "c_labels": it.get("labels") or it.get("tied_roles")})
    out = HERE / "output"; out.mkdir(exist_ok=True)
    (out / "mock_expert_audit.html").write_text(render_html(items, build_context_ext(items, docs), role_definitions()), encoding="utf-8")
    (out / "mock_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "mock_responses_template.json").write_text(json.dumps({it["item_id"]: {} for it in items}, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'mock_expert_audit.html'} (+ mock_key.json, mock_responses_template.json)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "annotation")
