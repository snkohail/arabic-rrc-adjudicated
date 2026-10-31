import numpy as np

from rrc import bootstrap as B
from rrc.metrics import PerExample, decode_argmax, decode_multilabel, to_binary


def test_set_f1_exact_and_macro_on_toy():
    gold = to_binary([["FACTS"], ["ANALYSIS", "LAW_REFERENCE"], ["ISSUE"]])
    pred = to_binary([["FACTS"], ["ANALYSIS"], ["FACTS"]])
    pe = PerExample(pred, gold)
    assert np.allclose(pe.set_f1, [1.0, 2 / 3, 0.0])
    s = pe.summary()
    assert s["exact_set_acc"] == round(1 / 3, 4) and s["set_f1"] == round((1 + 2 / 3) / 3, 4)
    # roles: FACTS tp1 fp1 -> 2/3 ; ANALYSIS 1.0 ; LAW_REFERENCE fn -> 0 ; ISSUE fn -> 0 ; others no support -> 0
    assert s["macro_f1"] == round((2 / 3 + 1.0) / 9, 4)


def test_decoders():
    P = np.array([[0.9, 0.6, 0.1, 0, 0, 0, 0, 0, 0], [0.2, 0.3, 0.1, 0, 0, 0, 0, 0, 0]])
    ml = decode_multilabel(P, 0.5)
    assert ml[0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert ml[1].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0]      # empty -> argmax fallback
    assert decode_multilabel(P, 0.5, fallback_argmax=False)[1].sum() == 0
    am = decode_argmax(P)
    assert am.sum(1).tolist() == [1, 1] and am[0, 0] == 1 and am[1, 1] == 1


def test_bootstrap_is_document_clustered_and_deterministic():
    docs = ["a"] * 3 + ["b"] * 2 + ["c"] * 5
    vals = np.array([1.0] * 3 + [0.0] * 2 + [0.5] * 5)
    stat = lambda idx: float(vals[idx].mean())
    r1 = B.ci(docs, stat, n_rep=300, seed=13)
    r2 = B.ci(docs, stat, n_rep=300, seed=13)
    assert r1 == r2 and r1["ci_low"] <= r1["point"] <= r1["ci_high"] and r1["n_documents"] == 3
    idx = next(B.resample_indices(docs, 1, 13))
    assert len(idx) in (6, 7, 8, 9, 10, 11, 12, 13, 15)   # whole documents only
    for d in set(np.array(docs)[idx]):
        assert (np.array(docs)[idx] == d).sum() % docs.count(d) == 0
    p = B.paired_ci(docs, stat, lambda i: stat(i) - 0.1, n_rep=200, seed=13)
    assert abs(p["diff_point"] - 0.1) < 1e-9 and abs(p["diff_ci_low"] - 0.1) < 1e-9 and abs(p["diff_ci_high"] - 0.1) < 1e-9


def test_micro_f1_and_cardinality():
    from rrc.metrics import cardinality, diagnostics, micro_f1
    gold = to_binary([["FACTS"], ["ANALYSIS", "LAW_REFERENCE"], ["ISSUE"]])
    pred = to_binary([["FACTS"], ["ANALYSIS"], ["FACTS"]])
    pe = PerExample(pred, gold)
    assert abs(micro_f1(pe) - 2 * 2 / (2 * 2 + 1 + 2)) < 1e-9     # tp=2, fp=1, fn=2
    rows = [{"is_multi": False}, {"is_multi": True}, {"is_multi": False}]
    c = cardinality(pred, gold, rows)
    assert c["overall"] == {"n": 3, "gold_cardinality": round(4 / 3, 4), "pred_cardinality": 1.0}
    assert c["multi_role"] == {"n": 1, "gold_cardinality": 2.0, "pred_cardinality": 1.0}
    P = np.array([[0.9, 0, 0, 0, 0, 0, 0, 0, 0]] * 3, dtype=float)
    d = diagnostics(P, gold, rows, 0.5)
    assert set(d) == {"micro_f1", "cardinality"}
