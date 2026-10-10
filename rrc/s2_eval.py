"""S2 evaluation: single-label sentence classification.

Primary metric macro-F1 over the nine roles; secondary micro-F1 and per-role F1; accuracy is
recorded (in single-label classification micro-F1 equals accuracy).  Document-clustered bootstrap
CIs (5,000 replicates, seed 13) for macro-F1 and micro-F1; paired document bootstrap on macro-F1
for model contrasts.  Implemented on one-hot matrices through the shared PerExample statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import bootstrap as B
from .config import ROLES
from .metrics import PerExample, micro_f1


def onehot(labels: Sequence[str]) -> np.ndarray:
    Y = np.zeros((len(labels), len(ROLES)), dtype=np.int8)
    for i, l in enumerate(labels):
        Y[i, ROLES.index(l)] = 1
    return Y


def evaluate(rows: List[dict], pred: Sequence[str], with_ci: bool) -> dict:
    Yg = onehot([r["label"] for r in rows]); Yp = onehot(pred)
    pe = PerExample(Yp, Yg)
    docs = [r["case_id"] for r in rows]
    res = {"n": len(rows), "n_documents": len(set(docs)), "macro_f1": round(pe.macro_f1(), 4), "micro_f1": round(micro_f1(pe), 4),
           "accuracy": round(float(pe.exact.mean()), 4), "per_role_f1": pe.per_role_f1(),
           "gold_support": {r: int(Yg[:, i].sum()) for i, r in enumerate(ROLES)},
           "pred_counts": {r: int(Yp[:, i].sum()) for i, r in enumerate(ROLES)}}
    if with_ci:
        res["ci"] = {"macro_f1": B.ci(docs, lambda i: pe.macro_f1(i)), "micro_f1": B.ci(docs, lambda i: micro_f1(pe, i))}
    return res


def paired_macro_f1(rows: List[dict], pred_a: Sequence[str], pred_b: Sequence[str]) -> dict:
    Yg = onehot([r["label"] for r in rows]); docs = [r["case_id"] for r in rows]
    a = PerExample(onehot(pred_a), Yg); b = PerExample(onehot(pred_b), Yg)
    return B.paired_ci(docs, lambda i: a.macro_f1(i), lambda i: b.macro_f1(i))


def save_predictions(rows: List[dict], pred: Sequence[str], path: Path, probs: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for k, (r, p) in enumerate(zip(rows, pred)):
            d = {"id": r["id"], "case_id": r["case_id"], "gold": r["label"], "pred": p}
            if probs is not None:
                d["probs"] = [round(float(x), 6) for x in probs[k]]
            f.write(json.dumps(d) + "\n")


def load_predictions(path: Path) -> List[str]:
    with path.open() as f:
        return [json.loads(l)["pred"] for l in f]
