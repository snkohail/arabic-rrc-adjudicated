"""Command-line entry point: ``python -m rrc <command>``.

Data stage: audit -> dedup -> freeze-grid -> split -> check -> datasets -> report.
Experiments: s1-b0, s1-neural, s1-softmax, s1-aggregate, s2-baselines, s2-neural, s2-aggregate.
Validation: expert-audit; grid-v3 (sentence-grid sensitivity).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C
from .audit import run_audit, write_audit
from .datasets import build_s1, build_s2, s1_summary, s2_summary, write_datasets
from .dedup import duplicate_groups
from .io import load_c_only
from .sentences import check_grid_manifest, grid_manifest
from .split import build_manifest, check_manifest_against_docs, load_manifest, write_manifest

GRID_PATH = C.SPLITS_DIR / f"{C.GRID_VERSION}.json"
SPLIT_PATH = C.SPLITS_DIR / f"{C.SPLIT['name']}.json"


def _all_docs(args, normalize=True):
    return load_c_only(C.annotation_root(args.annotation_root), normalize=normalize)


def _modeling_docs(args):
    docs = _all_docs(args)
    return {k: v for k, v in docs.items() if k not in C.EXCLUDED_FROM_MODELING}


def _load_grid():
    return json.loads(GRID_PATH.read_text())


def cmd_audit(args):
    raw = _all_docs(args, normalize=False)
    a = run_audit(raw, "<annotation_root>/" + C.ANNOTATOR_DIRS["C"], with_tokens=not args.no_tokens)
    a["excluded_from_modeling"] = dict(C.EXCLUDED_FROM_MODELING)
    write_audit(a, C.REPORTS_DIR / "audit")
    print((C.REPORTS_DIR / "audit" / "c_gold_audit.md").read_text())


def cmd_dedup(args):
    docs = _all_docs(args)
    rep = duplicate_groups({k: d.text for k, d in docs.items()})
    out = C.REPORTS_DIR / "split" / "duplicate_groups.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_documents", "n_groups", "multi_member_groups", "highest_unlinked_jaccard")}, indent=1))


def cmd_freeze_grid(args):
    if GRID_PATH.exists() and not args.force:
        raise SystemExit(f"{GRID_PATH} exists and is FROZEN; refusing to regenerate.")
    docs = _modeling_docs(args)
    man = grid_manifest(docs)
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRID_PATH.write_text(json.dumps(man, indent=1) + "\n")
    GRID_PATH.with_suffix(".sha256").write_text(man["grid_sha256"] + "  " + GRID_PATH.name + "\n")
    print(f"frozen {man['name']}: {man['n_sentences']} sentences over {man['n_judgments']} judgments; grid_sha256={man['grid_sha256']}")


def cmd_split(args):
    if SPLIT_PATH.exists() and not args.force:
        man = load_manifest(SPLIT_PATH)
        raise SystemExit(f"{SPLIT_PATH} exists with status {man['status']}; refusing to regenerate (use a new split name).")
    docs = _modeling_docs(args)
    all_docs = _all_docs(args)
    rep = duplicate_groups({k: d.text for k, d in all_docs.items()})
    grid_sha = _load_grid()["grid_sha256"] if GRID_PATH.exists() else ""
    man = build_manifest(docs, rep, status="FROZEN", grid_sha256=grid_sha, note=args.note or "")
    problems = check_manifest_against_docs(man, docs)
    if problems:
        raise SystemExit("split integrity problems: " + "; ".join(problems))
    write_manifest(man, SPLIT_PATH)
    print(f"frozen {man['name']} sha256={man['manifest_sha256']} sizes={ {s: len(v) for s, v in man['splits'].items()} }")
    print("volume bin boundaries:", man["volume_bin_boundaries"])
    print("related groups:", man["related_groups"], "excluded:", man["excluded_from_modeling"])
    for s, c in man["indicator_counts"].items():
        print(f"  {s:5s}", c)


def cmd_check(args):
    docs = _modeling_docs(args)
    g = _load_grid()
    gp = check_grid_manifest(g, docs)
    print(f"grid {g['name']} {g['status']} sha256={g['grid_sha256']}:", "OK" if not gp else "PROBLEMS: " + "; ".join(gp))
    man = load_manifest(SPLIT_PATH)
    sp = check_manifest_against_docs(man, docs)
    if man.get("grid_sha256") and man["grid_sha256"] != g["grid_sha256"]:
        sp.append("manifest references a different grid hash")
    print(f"split {man['name']} {man['status']} sha256={man['manifest_sha256']}:", "OK" if not sp else "PROBLEMS: " + "; ".join(sp))
    return 1 if (gp or sp) else 0


def cmd_datasets(args):
    docs = _modeling_docs(args)
    man = load_manifest(SPLIT_PATH)
    problems = check_manifest_against_docs(man, docs) + check_grid_manifest(_load_grid(), docs)
    if problems:
        raise SystemExit("frozen artifacts inconsistent with data: " + "; ".join(problems))
    s1 = build_s1(docs, man, with_tokens=not args.no_tokens)
    s2 = build_s2(docs, man)
    paths = write_datasets(s1, s2)
    summary = {
        "split": {"name": man["name"], "status": man["status"], "sha256": man["manifest_sha256"]},
        "grid": {"name": C.GRID_VERSION, "sha256": _load_grid()["grid_sha256"]},
        "c_normalization": C.C_NORMALIZATION["version"],
        "excluded_from_modeling": man["excluded_from_modeling"],
        "s1": s1_summary(s1, man),
        "s2": s2_summary(s2),
        "files": {k: str(Path(v).relative_to(C.DATA_DIR)) for k, v in paths.items()},
    }
    out = C.REPORTS_DIR / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("s1", "s2")}, indent=1))


def cmd_report(args):
    from .report import stage0_report
    md = stage0_report()
    (C.REPORTS_DIR / "STAGE0.md").write_text(md)
    print(md)


def cmd_s1_b0(args):
    from .s1_b0 import run
    run()


def cmd_s1_neural(args):
    from .s1_neural import run_model
    run_model(args.model, seeds=args.seeds, lrs=args.lrs, epochs=args.epochs, smoke=args.smoke)


def cmd_s1_aggregate(args):
    from .s1_aggregate import aggregate
    print(aggregate())


def cmd_s1_softmax(args):
    from .s1_softmax import run_model
    run_model(args.model, smoke=args.smoke)


def cmd_grid_v3(args):
    from .grid_v3 import build
    build(C.annotation_root(args.annotation_root), force=args.force)


def cmd_s2_baselines(args):
    from .s2_baselines import run_all
    run_all(only=args.only)


def cmd_s2_neural(args):
    from .s2_neural import run_model
    run_model(args.model)


def cmd_s2_aggregate(args):
    from .s2_aggregate import aggregate
    print(aggregate())


def cmd_expert_audit(args):
    from .expert_audit import build, render_from_frozen, score
    if args.action == "build":
        man = build(force=args.force)
        print(json.dumps({k: man[k] for k in ("version", "seed", "candidate_pool", "judgments_per_stratum", "items_sha256", "manifest_sha256")}, indent=1))
    elif args.action == "render":
        print(json.dumps(render_from_frozen(), indent=1))
    elif args.action == "ratification-form":
        from .expert_audit import write_ratification_form
        print(write_ratification_form())
    elif args.action == "ratification":
        from .expert_audit import record_ratification
        print(json.dumps(record_ratification(args.responses), indent=1))
    else:
        out = score(args.responses)
        print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="rrc")
    p.add_argument("--annotation-root", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit"); s.add_argument("--no-tokens", action="store_true"); s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("dedup"); s.set_defaults(fn=cmd_dedup)
    s = sub.add_parser("freeze-grid"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_freeze_grid)
    s = sub.add_parser("split"); s.add_argument("--force", action="store_true"); s.add_argument("--note", default=None); s.set_defaults(fn=cmd_split)
    s = sub.add_parser("check"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("datasets"); s.add_argument("--no-tokens", action="store_true"); s.set_defaults(fn=cmd_datasets)
    s = sub.add_parser("report"); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("s1-b0", help="S1-B0 lexical baseline: select on VAL, evaluate TEST once"); s.set_defaults(fn=cmd_s1_b0)
    s = sub.add_parser("s1-neural", help="S1-M1/M2 transformer span classifiers")
    s.add_argument("--model", choices=["m1_arabert", "m1_arabertv02", "m2_camelbert"], required=True)
    s.add_argument("--seeds", type=int, nargs="*", default=None); s.add_argument("--lrs", type=float, nargs="*", default=None)
    s.add_argument("--epochs", type=int, default=None); s.add_argument("--smoke", type=int, default=0)
    s.set_defaults(fn=cmd_s1_neural)
    s = sub.add_parser("s1-softmax", help="S1 single-label softmax controls (soft targets, argmax decoding)")
    s.add_argument("--model", choices=["m1s_arabert", "m1s_arabertv02", "m2s_camelbert"], required=True); s.add_argument("--smoke", type=int, default=0)
    s.set_defaults(fn=cmd_s1_softmax)
    s = sub.add_parser("s1-aggregate", help="S1 results tables + paired comparisons"); s.set_defaults(fn=cmd_s1_aggregate)
    s = sub.add_parser("grid-v3", help="build the text-only Stanza sentence grid (one-time; refuses to overwrite)"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_grid_v3)
    s = sub.add_parser("s2-baselines", help="S2-B0 majority, S2-B1 TF-IDF+LR, S2-B2 lexical CRF"); s.add_argument("--only", choices=["b0", "b1", "b2"], default=None); s.set_defaults(fn=cmd_s2_baselines)
    s = sub.add_parser("s2-neural", help="S2-M1/M2 transformer sentence classifiers"); s.add_argument("--model", choices=["m1_arabert", "m1_arabertv02", "m2_camelbert"], required=True); s.set_defaults(fn=cmd_s2_neural)
    s = sub.add_parser("s2-aggregate", help="S2 results tables + paired contrasts"); s.set_defaults(fn=cmd_s2_aggregate)
    s = sub.add_parser("expert-audit", help="blind expert validation: build (sample + interface) or score <responses.json>")
    s.add_argument("action", choices=["build", "render", "score", "ratification-form", "ratification"]); s.add_argument("--responses", default=None); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_expert_audit)
    args = p.parse_args(argv)
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
