"""Aggregate S2 results and paired contrasts (macro-F1, document-clustered bootstrap).

Encoders: M1 = AraBERTv02 (the paper's M1), M2 = CAMeLBERT-MSA, M1' = AraBERTv2 (supplementary).
Contrasts (pre-declared): B1−B0, B2−B1, encoder−B1 and encoder−B2 per seed, and M1−M2 / M1−M1' paired
seed-by-seed.  No superiority claim unless the paired CI excludes 0.
"""
from __future__ import annotations

import json
from statistics import mean, pstdev

from .config import DATA_DIR, REPORTS_DIR, ROLES
from .s2_data import load_s2
from .s2_eval import load_predictions, paired_macro_f1

OUT = REPORTS_DIR / "s2"
PRIV = DATA_DIR / "predictions"
ENCODERS = {"m1_arabertv02": ("M1", "S2-M1 AraBERTv02"), "m2_camelbert": ("M2", "S2-M2 CAMeLBERT-MSA"), "m1_arabert": ("M1′", "S2-M1′ AraBERTv2 (supplementary)")}
PAIRS = (("m1_arabertv02", "m2_camelbert"), ("m1_arabertv02", "m1_arabert"))


def _msd(v):
    return f"{mean(v):.4f} ± {pstdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def aggregate() -> str:
    test = load_s2("test")
    ids = [r["id"] for r in test]
    preds = {}
    res = {k: json.loads((OUT / k / "results.json").read_text()) for k in ("b0", "b1", "b2") if (OUT / k / "results.json").exists()}
    for k in res:
        preds[k] = load_predictions(PRIV / f"s2_{k}" / "test.jsonl")
    neural = {}
    for k in ENCODERS:
        p = OUT / k / "runs.json"
        if p.exists():
            runs = json.loads(p.read_text()); lr = runs["chosen_lr"]
            seeds = {r["seed"]: r for r in runs["runs"].values() if r["lr"] == lr and "test" in r}
            for sd in seeds:
                preds[f"{k}:{sd}"] = load_predictions(PRIV / f"s2_{k}" / f"lr{lr:g}_seed{sd}_test.jsonl")
            neural[k] = (runs, seeds)
    L = ["# S2 results — derived unambiguous single-label sentence benchmark (grid_v1, split_v1)", "",
         f"Eligible sentences: TRAIN {res['b1']['counts']['train']} / VAL {res['b1']['counts']['val']} / TEST {res['b1']['counts']['test']} "
         "(AMBIGUOUS_PROJECTION and UNLABELED excluded; no NONE class). Labels are projections of adjudicated spans, not independently annotated sentence gold.", "",
         "## TEST summary (primary macro-F1; micro-F1 = accuracy in single-label classification)", "",
         "| model | config | TEST macro-F1 [95% CI] | TEST micro-F1 [95% CI] | accuracy |", "|---|---|---:|---:|---:|"]
    names = {"b0": "S2-B0 majority", "b1": "S2-B1 TF-IDF + LR", "b2": "S2-B2 lexical CRF"}
    for k, r in res.items():
        T = r["test"]; cfg = r.get("majority_label") or (json.dumps(r["chosen"]) if "chosen" in r else "")
        L.append(f"| {names[k]} | {cfg} | {T['macro_f1']} [{T['ci']['macro_f1']['ci_low']}, {T['ci']['macro_f1']['ci_high']}] | {T['micro_f1']} [{T['ci']['micro_f1']['ci_low']}, {T['ci']['micro_f1']['ci_high']}] | {T['accuracy']} |")
    for k, (runs, seeds) in neural.items():
        label = ENCODERS[k][1]
        for sd in sorted(seeds):
            T = seeds[sd]["test"]
            L.append(f"| {label} seed {sd} | lr {runs['chosen_lr']:g}, epoch {seeds[sd]['best']['epoch']} | {T['macro_f1']} [{T['ci']['macro_f1']['ci_low']}, {T['ci']['macro_f1']['ci_high']}] | {T['micro_f1']} [{T['ci']['micro_f1']['ci_low']}, {T['ci']['micro_f1']['ci_high']}] | {T['accuracy']} |")
        L.append(f"| **{label} mean ± SD** | | **{_msd([seeds[s]['test']['macro_f1'] for s in seeds])}** | {_msd([seeds[s]['test']['micro_f1'] for s in seeds])} | {_msd([seeds[s]['test']['accuracy'] for s in seeds])} |")
    L += [""]
    # configs
    L += ["## Configurations and validation selection", ""]
    if "b1" in res:
        c = res["b1"]["chosen"]; L += [f"- B1: fixed {json.dumps(res['b1']['fixed'])}; grid {json.dumps(res['b1']['grid'])}; chosen ngram {c['ngram_range']}, C={c['C']}, class_weight={c['class_weight']} (VAL macro-F1 {c['val_macro_f1']}); VAL macro-F1 range over grid {min(t['val_macro_f1'] for t in res['b1']['table'])}–{max(t['val_macro_f1'] for t in res['b1']['table'])}"]
    if "b2" in res:
        c = res["b2"]["chosen"]; L += [f"- B2: {res['b2']['features']}; {json.dumps(res['b2']['fixed'])}; grid {json.dumps(res['b2']['grid'])}; chosen c1={c['c1']}, c2={c['c2']} (VAL macro-F1 {c['val_macro_f1']}); VAL range {min(t['val_macro_f1'] for t in res['b2']['table'])}–{max(t['val_macro_f1'] for t in res['b2']['table'])}"]
    for k, (runs, seeds) in neural.items():
        L += [f"- {k}: checkpoint `{runs['checkpoint']}` rev `{runs['revision']}`; {runs['tokenizer']}; windows {runs['long_span_handling']['window_tokens']}/stride {runs['long_span_handling']['stride_tokens']}; "
              f"lr grid {runs['hyperparameters']['lr_grid']}, epochs ≤ {runs['hyperparameters']['epochs']}; lr selection (seed 13): " + "; ".join(f"lr={s['lr']:g}: epoch {s['best_epoch']}, VAL macro-F1 {s['val_macro_f1']}" for s in runs["lr_selection"]) + f" → chosen {runs['chosen_lr']:g}; "
              + "per-seed best epoch / VAL macro-F1: " + ", ".join(f"{sd}: {seeds[sd]['best']['epoch']} / {seeds[sd]['best']['val_macro_f1']}" for sd in sorted(seeds))]
    L += [""]
    # per-role
    L += ["## Per-role F1 (TEST; neural = mean over seeds)", "", "| role | support | " + " | ".join(names[k] for k in res) + " | " + " | ".join(ENCODERS[k][0] for k in neural) + " |",
          "|---|---:|" + "---:|" * (len(res) + len(neural))]
    sup = res["b1"]["test"]["gold_support"]
    for r in ROLES:
        row = [f"| {r} | {sup[r]} |"] + [f" {res[k]['test']['per_role_f1'][r]} |" for k in res] + [f" {mean(s['test']['per_role_f1'][r] for s in seeds.values()):.4f} |" for _, seeds in neural.values()]
        L.append("".join(row))
    L += [""]
    # contrasts
    comp = {}
    L += ["## Paired contrasts (TEST macro-F1, identical document resamples)", "", "| contrast | a | b | diff | 95% CI |", "|---|---:|---:|---:|---|"]

    def add(name, a, b):
        pc = paired_macro_f1(test, preds[a], preds[b]); comp[name] = pc
        L.append(f"| {name} | {pc['a']} | {pc['b']} | {pc['diff_point']} | [{pc['diff_ci_low']}, {pc['diff_ci_high']}] |")

    if "b1" in preds and "b0" in preds: add("B1 − B0", "b1", "b0")
    if "b2" in preds and "b1" in preds: add("B2 − B1", "b2", "b1")
    for k, (runs, seeds) in neural.items():
        tag = ENCODERS[k][0]
        for sd in sorted(seeds):
            for base in ("b1", "b2"):
                if base in preds: add(f"{tag} seed {sd} − {base.upper()}", f"{k}:{sd}", base)
    for ka, kb in PAIRS:
        if ka in neural and kb in neural:
            for sd in sorted(set(neural[ka][1]) & set(neural[kb][1])):
                add(f"{ENCODERS[ka][0]} − {ENCODERS[kb][0]} seed {sd}", f"{ka}:{sd}", f"{kb}:{sd}")
    L += [""]
    (OUT / "comparisons.json").write_text(json.dumps(comp, indent=1))
    md = "\n".join(L); (OUT / "RESULTS_S2.md").write_text(md)
    return md
