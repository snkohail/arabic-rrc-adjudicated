"""S1-B0: character n-gram TF-IDF + nine one-vs-rest logistic regressions.

Pre-specified, small grid (TRAIN fit, VALIDATION selection by set-F1):
  ngram_range in {(2,4), (3,5)}  x  C in {0.3, 1, 3, 10}  x  class_weight in {None, balanced}
  one global threshold in {0.05, 0.10, ..., 0.95}
Fixed: analyzer=char_wb, sublinear_tf, min_df=2, max_features=500000, lowercase=False,
liblinear solver, empty-prediction fallback = argmax.  Selection ties -> smaller C,
smaller n-gram, class_weight None.  The chosen configuration is refit on TRAIN only and
TEST is evaluated once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from . import bootstrap as B
from .config import DATA_DIR, DERIVED_DIR, REPORTS_DIR, ROLES
from .metrics import PerExample, decode_argmax, decode_multilabel, slice_masks, to_binary

GRID = {"ngram_range": [(2, 4), (3, 5)], "C": [0.3, 1.0, 3.0, 10.0], "class_weight": [None, "balanced"]}
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)]
FIXED = {"analyzer": "char_wb", "sublinear_tf": True, "min_df": 2, "max_features": 500000, "lowercase": False,
         "solver": "liblinear", "max_iter": 1000, "fallback": "argmax when no role passes the threshold", "seed": 13}
OUT = REPORTS_DIR / "s1" / "b0"
PRIV = DATA_DIR / "predictions" / "s1_b0"


def load_split(name: str) -> List[dict]:
    with (DERIVED_DIR / "s1" / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def fit_lr(X, Y, C, cw):
    clf = OneVsRestClassifier(LogisticRegression(C=C, class_weight=cw, solver=FIXED["solver"], max_iter=FIXED["max_iter"],
                                                 random_state=FIXED["seed"]), n_jobs=len(ROLES))
    return clf.fit(X, Y)


def select(train, val) -> dict:
    Ytr, Yva = to_binary([r["labels"] for r in train]), to_binary([r["labels"] for r in val])
    table = []
    for ng in GRID["ngram_range"]:
        t0 = time.time()
        vec = TfidfVectorizer(analyzer=FIXED["analyzer"], ngram_range=ng, sublinear_tf=FIXED["sublinear_tf"], min_df=FIXED["min_df"],
                              max_features=FIXED["max_features"], lowercase=FIXED["lowercase"], dtype=np.float32)
        Xtr = vec.fit_transform([r["text"] for r in train]); Xva = vec.transform([r["text"] for r in val])
        nfeat = Xtr.shape[1]
        for C in GRID["C"]:
            for cw in GRID["class_weight"]:
                clf = fit_lr(Xtr, Ytr, C, cw)
                P = clf.predict_proba(Xva)
                best = None
                for thr in THRESHOLDS:
                    s = PerExample(decode_multilabel(P, thr), Yva).summary()
                    if best is None or s["set_f1"] > best["set_f1"]:
                        best = dict(s, threshold=thr)
                table.append({"ngram_range": list(ng), "C": C, "class_weight": cw or "none", "n_features": int(nfeat),
                              "threshold": best["threshold"], "val_set_f1": best["set_f1"], "val_macro_f1": best["macro_f1"],
                              "val_exact_set_acc": best["exact_set_acc"]})
                print(f"  ngram={ng} C={C} cw={cw or 'none'} -> thr={best['threshold']} set-F1={best['set_f1']:.4f} "
                      f"macro-F1={best['macro_f1']:.4f} exact={best['exact_set_acc']:.4f}", flush=True)
        print(f"  [{ng}] {time.time() - t0:.0f}s", flush=True)
    order = lambda r: (-r["val_set_f1"], r["C"], r["ngram_range"][1], r["class_weight"] != "none")
    chosen = sorted(table, key=order)[0]
    return {"grid": {"ngram_range": [list(x) for x in GRID["ngram_range"]], "C": GRID["C"], "class_weight": ["none", "balanced"],
                     "thresholds": THRESHOLDS}, "fixed": FIXED, "selection_criterion": "validation set-F1 (ties: smaller C, smaller n-gram, class_weight none)",
            "table": table, "chosen": chosen}


def evaluate_test(train, test, chosen: dict, val=None) -> dict:
    ng = tuple(chosen["ngram_range"]); C = chosen["C"]; cw = None if chosen["class_weight"] == "none" else chosen["class_weight"]
    thr = chosen["threshold"]
    vec = TfidfVectorizer(analyzer=FIXED["analyzer"], ngram_range=ng, sublinear_tf=FIXED["sublinear_tf"], min_df=FIXED["min_df"],
                          max_features=FIXED["max_features"], lowercase=FIXED["lowercase"], dtype=np.float32)
    Xtr = vec.fit_transform([r["text"] for r in train])
    clf = fit_lr(Xtr, to_binary([r["labels"] for r in train]), C, cw)
    out = {}
    for name, rows in (("val", val), ("test", test)):
        if rows is None:
            continue
        P = clf.predict_proba(vec.transform([r["text"] for r in rows]))
        PRIV.mkdir(parents=True, exist_ok=True)
        with (PRIV / f"{name}.jsonl").open("w") as f:
            for r, p in zip(rows, P):
                f.write(json.dumps({"id": r["id"], "case_id": r["case_id"], "gold": r["labels"], "probs": [round(float(x), 6) for x in p]}) + "\n")
        Yg = to_binary([r["labels"] for r in rows])
        docs = [r["case_id"] for r in rows]
        ml = PerExample(decode_multilabel(P, thr), Yg)
        am = PerExample(decode_argmax(P), Yg)
        res = {"n": len(rows), "n_documents": len(set(docs)), "threshold": thr,
               "multilabel": ml.summary(), "per_role_f1": ml.per_role_f1(),
               "slices": {k: ml.summary(np.where(m)[0]) for k, m in slice_masks(rows).items()}}
        if name == "test":
            res["ci"] = {
                "set_f1": B.ci(docs, lambda i: float(ml.set_f1[i].mean())),
                "macro_f1": B.ci(docs, lambda i: ml.macro_f1(i)),
                "exact_set_acc": B.ci(docs, lambda i: float(ml.exact[i].mean())),
            }
            multi = np.array([r["is_multi"] for r in rows], dtype=bool)
            mi = np.where(multi)[0]
            res["controlled_decoding"] = {
                "description": "same probabilities; multilabel = threshold decoding (argmax fallback); forced_single = argmax only",
                "all_examples": {"multilabel_set_f1": ml.summary()["set_f1"], "forced_single_set_f1": am.summary()["set_f1"],
                                 "paired_ci_diff_multilabel_minus_forced": B.paired_ci(docs, lambda i: float(ml.set_f1[i].mean()), lambda i: float(am.set_f1[i].mean()))},
                "multi_role_examples": {"n": int(multi.sum()), "multilabel_set_f1": ml.summary(mi)["set_f1"], "forced_single_set_f1": am.summary(mi)["set_f1"],
                                        "paired_ci_diff_multilabel_minus_forced": B.paired_ci(
                                            [docs[i] for i in mi], lambda i: float(ml.set_f1[mi][i].mean()), lambda i: float(am.set_f1[mi][i].mean()))},
                "forced_single_all": am.summary(), "forced_single_slices": {k: am.summary(np.where(m)[0]) for k, m in slice_masks(rows).items()},
            }
        out[name] = res
    return out


def markdown(sel: dict, res: dict, counts: dict) -> str:
    ch = sel["chosen"]; T = res["test"]; M = T["multilabel"]; CD = T["controlled_decoding"]
    L = ["# S1-B0 — char n-gram TF-IDF + one-vs-rest logistic regression", "",
         "## Configuration", "",
         f"- fixed: {json.dumps(sel['fixed'])}", f"- grid: {json.dumps(sel['grid'])}", f"- selection: {sel['selection_criterion']}",
         f"- **chosen**: ngram {ch['ngram_range']}, C={ch['C']}, class_weight={ch['class_weight']}, features={ch['n_features']}, **threshold={ch['threshold']}**",
         f"- examples: {counts}", "",
         "## Validation selection (TRAIN fit, VAL scored at each configuration's best global threshold)", "",
         "| ngram | C | class_weight | thr | VAL set-F1 | VAL macro-F1 | VAL exact |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in sel["table"]:
        mark = " **" if r is ch else ""
        L.append(f"| {r['ngram_range']} | {r['C']} | {r['class_weight']} | {r['threshold']} | {r['val_set_f1']}{mark} | {r['val_macro_f1']} | {r['val_exact_set_acc']} |")
    ci = T["ci"]
    L += ["", "## TEST (evaluated once)", "", "| metric | point | 95% CI (document-clustered bootstrap, 5000 reps, seed 13) |", "|---|---:|---|",
          f"| **set-F1 (primary)** | {M['set_f1']} | [{ci['set_f1']['ci_low']}, {ci['set_f1']['ci_high']}] |",
          f"| macro-F1 | {M['macro_f1']} | [{ci['macro_f1']['ci_low']}, {ci['macro_f1']['ci_high']}] |",
          f"| exact-set accuracy | {M['exact_set_acc']} | [{ci['exact_set_acc']['ci_low']}, {ci['exact_set_acc']['ci_high']}] |", "",
          "Per-role F1 (TEST): " + ", ".join(f"{k} {v}" for k, v in T["per_role_f1"].items()), "",
          "## TEST slices (multi-label decoding)", "", "| slice | n | set-F1 | macro-F1 | exact |", "|---|---:|---:|---:|---:|"]
    for k, s in T["slices"].items():
        L.append(f"| {k} | {s['n']} | {s['set_f1']} | {s['macro_f1']} | {s['exact_set_acc']} |")
    a, m = CD["all_examples"], CD["multi_role_examples"]
    L += ["", "## Controlled decoding on the same TEST probabilities", "", CD["description"], "",
          "| examples | n | multi-label set-F1 | forced single-label set-F1 | diff (ML − forced) | paired 95% CI |", "|---|---:|---:|---:|---:|---|",
          f"| all | {T['n']} | {a['multilabel_set_f1']} | {a['forced_single_set_f1']} | {a['paired_ci_diff_multilabel_minus_forced']['diff_point']} | [{a['paired_ci_diff_multilabel_minus_forced']['diff_ci_low']}, {a['paired_ci_diff_multilabel_minus_forced']['diff_ci_high']}] |",
          f"| genuine multi-role | {m['n']} | {m['multilabel_set_f1']} | {m['forced_single_set_f1']} | {m['paired_ci_diff_multilabel_minus_forced']['diff_point']} | [{m['paired_ci_diff_multilabel_minus_forced']['diff_ci_low']}, {m['paired_ci_diff_multilabel_minus_forced']['diff_ci_high']}] |", ""]
    return "\n".join(L)


def run() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    counts = {"train": len(train), "val": len(val), "test": len(test)}
    print("S1 counts:", counts, flush=True)
    sel = select(train, val)
    print("chosen:", sel["chosen"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_selection.json").write_text(json.dumps(sel, indent=1))
    res = evaluate_test(train, test, sel["chosen"], val=val)
    (OUT / "test_results.json").write_text(json.dumps({"counts": counts, **res}, indent=1))
    md = markdown(sel, res, counts)
    (OUT / "RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    run()
