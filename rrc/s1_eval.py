"""Shared S1 TEST evaluation (identical to the frozen B0 procedure) + secondary diagnostics.

Given per-example probabilities and the VAL-selected global threshold, produce the frozen
report block: set-F1 / macro-F1 / exact-set accuracy with document-clustered bootstrap CIs,
predeclared slices, per-role F1, controlled decoding (threshold+fallback vs forced argmax)
with paired CIs, plus micro-F1 and label cardinality (reporting only).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import bootstrap as B
from .metrics import PerExample, decode_argmax, decode_multilabel, diagnostics, slice_masks, to_binary


def save_predictions(rows: List[dict], P: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r, p in zip(rows, P):
            f.write(json.dumps({"id": r["id"], "case_id": r["case_id"], "gold": r["labels"], "probs": [round(float(x), 6) for x in p]}) + "\n")


def load_predictions(path: Path):
    ids, probs = [], []
    with path.open() as f:
        for l in f:
            d = json.loads(l); ids.append(d["id"]); probs.append(d["probs"])
    return ids, np.array(probs, dtype=np.float64)


def evaluate_split(rows: List[dict], P: np.ndarray, thr: float, with_ci: bool) -> dict:
    Yg = to_binary([r["labels"] for r in rows])
    docs = [r["case_id"] for r in rows]
    ml = PerExample(decode_multilabel(P, thr), Yg)
    res = {"n": len(rows), "n_documents": len(set(docs)), "threshold": thr, "multilabel": ml.summary(),
           "per_role_f1": ml.per_role_f1(), "slices": {k: ml.summary(np.where(m)[0]) for k, m in slice_masks(rows).items()},
           "diagnostics": diagnostics(P, Yg, rows, thr)}
    if with_ci:
        am = PerExample(decode_argmax(P), Yg)
        res["ci"] = {"set_f1": B.ci(docs, lambda i: float(ml.set_f1[i].mean())),
                     "macro_f1": B.ci(docs, lambda i: ml.macro_f1(i)),
                     "exact_set_acc": B.ci(docs, lambda i: float(ml.exact[i].mean()))}
        multi = np.array([r["is_multi"] for r in rows], dtype=bool); mi = np.where(multi)[0]
        res["controlled_decoding"] = {
            "description": "same probabilities; multilabel = threshold decoding (argmax fallback); forced_single = argmax only",
            "all_examples": {"multilabel_set_f1": ml.summary()["set_f1"], "forced_single_set_f1": am.summary()["set_f1"],
                             "paired_ci_diff_multilabel_minus_forced": B.paired_ci(docs, lambda i: float(ml.set_f1[i].mean()), lambda i: float(am.set_f1[i].mean()))},
            "multi_role_examples": {"n": int(multi.sum()), "multilabel_set_f1": ml.summary(mi)["set_f1"], "forced_single_set_f1": am.summary(mi)["set_f1"],
                                    "paired_ci_diff_multilabel_minus_forced": B.paired_ci([docs[i] for i in mi], lambda i: float(ml.set_f1[mi][i].mean()),
                                                                                          lambda i: float(am.set_f1[mi][i].mean()))},
            "forced_single_all": am.summary(), "forced_single_slices": {k: am.summary(np.where(m)[0]) for k, m in slice_masks(rows).items()}}
    return res


def paired_set_f1(rows: List[dict], P_a: np.ndarray, thr_a: float, P_b: np.ndarray, thr_b: float) -> dict:
    """Paired document-clustered bootstrap of set-F1(a) - set-F1(b), both under threshold+fallback decoding."""
    Yg = to_binary([r["labels"] for r in rows]); docs = [r["case_id"] for r in rows]
    a = PerExample(decode_multilabel(P_a, thr_a), Yg); b = PerExample(decode_multilabel(P_b, thr_b), Yg)
    return B.paired_ci(docs, lambda i: float(a.set_f1[i].mean()), lambda i: float(b.set_f1[i].mean()))
