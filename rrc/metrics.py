"""Multi-label metrics for S1 and the two decoding rules.

* set-F1 (PRIMARY): per example 2|P∩G| / (|P|+|G|), averaged over examples.
* macro-F1: F1 per role over all examples, averaged over the nine roles (zero_division=0).
* exact-set accuracy: fraction of examples with P == G.

Decoding rules applied to the SAME probability matrix:
* ``decode_multilabel``: every role with probability >= threshold; if none passes, the
  argmax role (gold sets are never empty, so an empty prediction is never correct).
* ``decode_argmax``: exactly one role, the argmax (forced single-label).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .config import ROLES

R = len(ROLES)


def to_binary(label_lists: Sequence[Sequence[str]]) -> np.ndarray:
    Y = np.zeros((len(label_lists), R), dtype=np.int8)
    for i, labels in enumerate(label_lists):
        for l in labels:
            Y[i, ROLES.index(l)] = 1
    return Y


def decode_multilabel(P: np.ndarray, threshold: float, fallback_argmax: bool = True) -> np.ndarray:
    Yp = (P >= threshold).astype(np.int8)
    if fallback_argmax:
        empty = Yp.sum(1) == 0
        Yp[np.where(empty)[0], P[empty].argmax(1)] = 1
    return Yp


def decode_argmax(P: np.ndarray) -> np.ndarray:
    Yp = np.zeros_like(P, dtype=np.int8)
    Yp[np.arange(len(P)), P.argmax(1)] = 1
    return Yp


class PerExample:
    """Per-example statistics so that any subset (bootstrap resample, slice) is a cheap reduction."""

    def __init__(self, Yp: np.ndarray, Yg: np.ndarray):
        Yp = Yp.astype(np.int64); Yg = Yg.astype(np.int64)
        inter = (Yp & Yg).sum(1)
        denom = Yp.sum(1) + Yg.sum(1)
        self.set_f1 = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1), 1.0)
        self.exact = (Yp == Yg).all(1).astype(float)
        self.tp = (Yp & Yg)
        self.fp = (Yp & (1 - Yg))
        self.fn = ((1 - Yp) & Yg)
        self.n = len(Yg)

    def macro_f1(self, idx=None) -> float:
        tp = self.tp if idx is None else self.tp[idx]
        fp = self.fp if idx is None else self.fp[idx]
        fn = self.fn if idx is None else self.fn[idx]
        tp, fp, fn = tp.sum(0), fp.sum(0), fn.sum(0)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        return float(f1.mean())

    def per_role_f1(self) -> Dict[str, float]:
        tp, fp, fn = self.tp.sum(0), self.fp.sum(0), self.fn.sum(0)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        return {r: round(float(f1[i]), 4) for i, r in enumerate(ROLES)}

    def summary(self, idx=None) -> Dict[str, float]:
        sf = self.set_f1 if idx is None else self.set_f1[idx]
        ex = self.exact if idx is None else self.exact[idx]
        return {"n": int(len(sf)), "set_f1": round(float(sf.mean()), 4), "macro_f1": round(self.macro_f1(idx), 4),
                "exact_set_acc": round(float(ex.mean()), 4)}


def slice_masks(rows: List[dict]) -> Dict[str, np.ndarray]:
    multi = np.array([r["is_multi"] for r in rows], dtype=bool)
    nested = np.array([r["is_nested"] for r in rows], dtype=bool)
    return {"single_role": ~multi, "multi_role": multi, "nested": nested, "non_nested": ~nested}


# ----------------------------------------------------------------------------- secondary diagnostics
def micro_f1(pe: "PerExample", idx=None) -> float:
    """Micro-F1 over all (example, role) decisions (secondary/reporting only)."""
    tp = (pe.tp if idx is None else pe.tp[idx]).sum()
    fp = (pe.fp if idx is None else pe.fp[idx]).sum()
    fn = (pe.fn if idx is None else pe.fn[idx]).sum()
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else 0.0


def cardinality(Yp: np.ndarray, Yg: np.ndarray, rows: List[dict]) -> Dict[str, Dict[str, float]]:
    """Mean number of gold and predicted roles per example: overall, single-role subset, multi-role subset."""
    multi = np.array([r["is_multi"] for r in rows], dtype=bool)
    out = {}
    for name, m in (("overall", np.ones(len(rows), dtype=bool)), ("single_role", ~multi), ("multi_role", multi)):
        out[name] = {"n": int(m.sum()), "gold_cardinality": round(float(Yg[m].sum(1).mean()), 4),
                     "pred_cardinality": round(float(Yp[m].sum(1).mean()), 4)}
    return out


def diagnostics(P: np.ndarray, Yg: np.ndarray, rows: List[dict], threshold: float) -> Dict[str, object]:
    Yp = decode_multilabel(P, threshold)
    pe = PerExample(Yp, Yg)
    return {"micro_f1": round(micro_f1(pe), 4), "cardinality": cardinality(Yp, Yg, rows)}
