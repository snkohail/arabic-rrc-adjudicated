"""Document-clustered bootstrap (document = sampling unit).

Judgments are resampled with replacement; the examples of a judgment drawn k times
enter the statistic k times.  ``paired`` evaluates two systems on identical resamples
and returns the CI of the difference.  5,000 replicates, seed 13 (pre-declared).
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

N_REPLICATES = 5000
SEED = 13


def _doc_index(doc_ids: Sequence[str]):
    docs = sorted(set(doc_ids))
    where = {d: np.where(np.asarray(doc_ids) == d)[0] for d in docs}
    return docs, where


def resample_indices(doc_ids: Sequence[str], n_rep: int = N_REPLICATES, seed: int = SEED):
    docs, where = _doc_index(doc_ids)
    rng = np.random.default_rng(seed)
    D = len(docs)
    for _ in range(n_rep):
        draw = rng.integers(0, D, size=D)
        yield np.concatenate([where[docs[k]] for k in draw])


def ci(doc_ids: Sequence[str], stat: Callable[[np.ndarray], float], n_rep: int = N_REPLICATES, seed: int = SEED,
       alpha: float = 0.05) -> Dict[str, float]:
    vals = np.array([stat(idx) for idx in resample_indices(doc_ids, n_rep, seed)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": round(float(stat(np.arange(len(doc_ids)))), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "n_replicates": n_rep, "seed": seed, "n_documents": len(set(doc_ids))}


def paired_ci(doc_ids: Sequence[str], stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
              n_rep: int = N_REPLICATES, seed: int = SEED, alpha: float = 0.05) -> Dict[str, float]:
    """CI of (a - b) on identical document resamples."""
    diffs = np.array([stat_a(idx) - stat_b(idx) for idx in resample_indices(doc_ids, n_rep, seed)])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    full = np.arange(len(doc_ids))
    return {"a": round(float(stat_a(full)), 4), "b": round(float(stat_b(full)), 4),
            "diff_point": round(float(stat_a(full) - stat_b(full)), 4), "diff_ci_low": round(float(lo), 4),
            "diff_ci_high": round(float(hi), 4), "n_replicates": n_rep, "seed": seed, "n_documents": len(set(doc_ids))}
