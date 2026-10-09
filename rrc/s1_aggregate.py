"""Aggregate S1 results: B0 (frozen) + M1/M2 seeds, mean/SD, and paired comparisons on set-F1.

Pairing rule (pre-declared): each neural seed's TEST probabilities are compared with the frozen
B0 TEST probabilities on identical document resamples; M1 vs M2 are paired seed-by-seed
(13 vs 13, 37 vs 37, 73 vs 73).  Decoding for every system = its own VAL-selected global
threshold with argmax fallback.  No claim of superiority unless the paired CI excludes 0.
"""
from __future__ import annotations

import json
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np

from .config import DATA_DIR, REPORTS_DIR, ROLES
from .s1_eval import load_predictions, paired_set_f1
from .s1_neural import load_split

OUT = REPORTS_DIR / "s1"
PRIV = DATA_DIR / "predictions"


def _b0():
    sel = json.loads((OUT / "b0" / "validation_selection.json").read_text())
    res = json.loads((OUT / "b0" / "test_results.json").read_text())
    ids, P = load_predictions(PRIV / "s1_b0" / "test.jsonl")
    return {"thr": sel["chosen"]["threshold"], "P": P, "ids": ids, "test": res["test"]}


def _neural(key: str):
    runs = json.loads((OUT / key / "runs.json").read_text())
    lr = runs["chosen_lr"]
    seeds = {}
    for k, r in runs["runs"].items():
        if r["lr"] == lr and "test" in r:
            ids, P = load_predictions(PRIV / key / f"lr{lr:g}_seed{r['seed']}_test.jsonl")
            seeds[r["seed"]] = {"thr": r["best"]["val_threshold"], "P": P, "ids": ids, "run": r}
    return runs, seeds


def _msd(vals: List[float]):
    return f"{mean(vals):.4f} ± {pstdev(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"


def aggregate() -> str:
    test = load_split("test")
    ids_ref = [r["id"] for r in test]
    b0 = _b0(); assert b0["ids"] == ids_ref
    L = ["# S1 results — frozen S1 dataset (TRAIN 10,242 / VAL 1,359 / TEST 1,846)", ""]
    comp = {"b0_threshold": b0["thr"], "vs_b0": {}, "m1_vs_m2": {}}
    models = {}
    for key, label in (("m1_arabert", "S1-M1 AraBERTv2"), ("m2_camelbert", "S1-M2 CAMeLBERT-MSA")):
        if not (OUT / key / "runs.json").exists():
            continue
        runs, seeds = _neural(key)
        models[key] = seeds
        L += [f"## {label}", "",
              f"- checkpoint `{runs['checkpoint']}` revision `{runs['revision']}`; tokenizer: {runs['tokenizer']}",
              f"- long-span handling: window {runs['long_span_handling']['window_tokens']} tokens, stride {runs['long_span_handling']['stride_tokens']}, {runs['long_span_handling']['pooling']}",
              f"- grid: lr ∈ {runs['hyperparameters']['lr_grid']}, epochs ≤ {runs['hyperparameters']['epochs']} (best VAL epoch), batch {runs['hyperparameters']['batch_examples']} examples / ≤ {runs['hyperparameters']['max_windows_per_batch']} windows, "
              f"warmup {runs['hyperparameters']['warmup_ratio']}, wd {runs['hyperparameters']['weight_decay']}, dropout {runs['hyperparameters']['dropout']}, {runs['hyperparameters']['precision']}; thresholds {runs['hyperparameters']['thresholds'][0]}…{runs['hyperparameters']['thresholds'][-1]} step 0.05; fallback argmax",
              f"- lr selection (seed {runs['hyperparameters']['selection_seed']}, VAL set-F1): " + "; ".join(f"lr={s['lr']:g}: epoch {s['best_epoch']}, thr {s['val_threshold']}, set-F1 {s['val_set_f1']}" for s in runs["lr_selection"]) + f" → **chosen lr {runs['chosen_lr']:g}**", "",
              "| seed | best epoch | VAL thr | VAL set-F1 | VAL micro-F1 | TEST set-F1 | TEST macro-F1 | TEST exact | TEST micro-F1 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for sd in sorted(seeds):
            r = seeds[sd]["run"]; T = r["test"]["multilabel"]
            L.append(f"| {sd} | {r['best']['epoch']} | {r['best']['val_threshold']} | {r['best']['val_set_f1']} | {r['val']['diagnostics']['micro_f1']} | {T['set_f1']} | {T['macro_f1']} | {T['exact_set_acc']} | {r['test']['diagnostics']['micro_f1']} |")
        sf = [seeds[s]["run"]["test"]["multilabel"]["set_f1"] for s in sorted(seeds)]
        mf = [seeds[s]["run"]["test"]["multilabel"]["macro_f1"] for s in sorted(seeds)]
        ex = [seeds[s]["run"]["test"]["multilabel"]["exact_set_acc"] for s in sorted(seeds)]
        L += [f"| **mean ± SD** | | | | | **{_msd(sf)}** | {_msd(mf)} | {_msd(ex)} | |", ""]
        L += ["Per-seed TEST 95% CIs (document-clustered bootstrap): " + "; ".join(
            f"seed {s}: set-F1 [{seeds[s]['run']['test']['ci']['set_f1']['ci_low']}, {seeds[s]['run']['test']['ci']['set_f1']['ci_high']}]" for s in sorted(seeds)), ""]
        L += ["Per-role F1 (TEST, mean over seeds): " + ", ".join(f"{r} {mean(seeds[s]['run']['test']['per_role_f1'][r] for s in seeds):.3f}" for r in ROLES), ""]
        L += ["TEST slices (set-F1, mean ± SD over seeds):", "", "| slice | n | set-F1 | macro-F1 | exact |", "|---|---:|---:|---:|---:|"]
        for sl in ("single_role", "multi_role", "nested", "non_nested"):
            n = seeds[sorted(seeds)[0]]["run"]["test"]["slices"][sl]["n"]
            L.append(f"| {sl} | {n} | {_msd([seeds[s]['run']['test']['slices'][sl]['set_f1'] for s in seeds])} | {_msd([seeds[s]['run']['test']['slices'][sl]['macro_f1'] for s in seeds])} | {_msd([seeds[s]['run']['test']['slices'][sl]['exact_set_acc'] for s in seeds])} |")
        L += ["", "Controlled decoding (same probabilities; set-F1 multi-label vs forced single-label; paired CI of the difference):", "",
              "| seed | all: ML | all: forced | diff [95% CI] | multi-role: ML | multi-role: forced | diff [95% CI] |", "|---:|---:|---:|---|---:|---:|---|"]
        for sd in sorted(seeds):
            cd = seeds[sd]["run"]["test"]["controlled_decoding"]; a, m = cd["all_examples"], cd["multi_role_examples"]
            pa, pm = a["paired_ci_diff_multilabel_minus_forced"], m["paired_ci_diff_multilabel_minus_forced"]
            L.append(f"| {sd} | {a['multilabel_set_f1']} | {a['forced_single_set_f1']} | {pa['diff_point']} [{pa['diff_ci_low']}, {pa['diff_ci_high']}] | {m['multilabel_set_f1']} | {m['forced_single_set_f1']} | {pm['diff_point']} [{pm['diff_ci_low']}, {pm['diff_ci_high']}] |")
        L += ["", "Label cardinality (TEST, gold vs predicted, mean over seeds):", ""]
        for sub in ("overall", "single_role", "multi_role"):
            g = seeds[sorted(seeds)[0]]["run"]["test"]["diagnostics"]["cardinality"][sub]["gold_cardinality"]
            p = mean(seeds[s]["run"]["test"]["diagnostics"]["cardinality"][sub]["pred_cardinality"] for s in seeds)
            L.append(f"- {sub}: gold {g:.3f}, predicted {p:.3f}")
        L += [""]
        # paired vs B0
        comp["vs_b0"][key] = {}
        L += [f"Paired comparison vs frozen S1-B0 (set-F1, {label} − B0, identical document resamples):", "", "| seed | model | B0 | diff | 95% CI |", "|---:|---:|---:|---:|---|"]
        for sd in sorted(seeds):
            pc = paired_set_f1(test, seeds[sd]["P"], seeds[sd]["thr"], b0["P"], b0["thr"])
            comp["vs_b0"][key][str(sd)] = pc
            L.append(f"| {sd} | {pc['a']} | {pc['b']} | {pc['diff_point']} | [{pc['diff_ci_low']}, {pc['diff_ci_high']}] |")
        L += [""]
    if "m1_arabert" in models and "m2_camelbert" in models:
        L += ["## S1-M1 AraBERTv2 vs S1-M2 CAMeLBERT-MSA (set-F1, M1 − M2, same seed paired on identical resamples)", "", "| seed | M1 | M2 | diff | 95% CI |", "|---:|---:|---:|---:|---|"]
        for sd in sorted(set(models["m1_arabert"]) & set(models["m2_camelbert"])):
            a, b = models["m1_arabert"][sd], models["m2_camelbert"][sd]
            pc = paired_set_f1(test, a["P"], a["thr"], b["P"], b["thr"])
            comp["m1_vs_m2"][str(sd)] = pc
            L.append(f"| {sd} | {pc['a']} | {pc['b']} | {pc['diff_point']} | [{pc['diff_ci_low']}, {pc['diff_ci_high']}] |")
        L += [""]
    L += ["## Frozen S1-B0 (reference)", "", f"TEST set-F1 {b0['test']['multilabel']['set_f1']} [{b0['test']['ci']['set_f1']['ci_low']}, {b0['test']['ci']['set_f1']['ci_high']}], macro-F1 {b0['test']['multilabel']['macro_f1']}, exact {b0['test']['multilabel']['exact_set_acc']}; threshold {b0['thr']}.", ""]
    (OUT / "comparisons.json").write_text(json.dumps(comp, indent=1))
    md = "\n".join(L)
    (OUT / "RESULTS_S1.md").write_text(md)
    return md
