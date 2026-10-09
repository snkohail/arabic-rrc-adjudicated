import numpy as np

from rrc.s2_baselines import sent_features, seq_features
from rrc.s2_data import crf_sequences, sequences
from rrc.s2_eval import evaluate, onehot, paired_macro_f1


def test_evaluate_single_label_metrics():
    rows = [{"id": str(i), "case_id": "d%d" % (i % 3), "label": l} for i, l in enumerate(["FACTS", "FACTS", "ISSUE", "ANALYSIS", "ISSUE", "FACTS"])]
    pred = ["FACTS", "ISSUE", "ISSUE", "ANALYSIS", "ISSUE", "FACTS"]
    ev = evaluate(rows, pred, with_ci=True)
    assert ev["accuracy"] == ev["micro_f1"] == round(5 / 6, 4)
    # FACTS: tp2 fn1 -> 0.8 ; ISSUE: tp2 fp1 -> 0.8 ; ANALYSIS 1.0 ; six absent roles -> 0
    assert ev["macro_f1"] == round((0.8 + 0.8 + 1.0) / 9, 4)
    assert ev["ci"]["macro_f1"]["n_documents"] == 3 and ev["gold_support"]["FACTS"] == 3
    pc = paired_macro_f1(rows, pred, [r["label"] for r in rows])
    assert pc["diff_point"] < 0 and pc["diff_ci_high"] <= 0


def test_sequences_keep_document_order():
    rows = [{"case_id": "b", "sent_idx": 2}, {"case_id": "a", "sent_idx": 5}, {"case_id": "a", "sent_idx": 1}]
    rows.sort(key=lambda r: (r["case_id"], r["sent_idx"]))
    s = sequences(rows)
    assert list(s) == ["a", "b"] and [r["sent_idx"] for r in s["a"]] == [1, 5]


def test_crf_sequences_break_at_excluded_sentences():
    rows = [{"case_id": "a", "sent_idx": 0}, {"case_id": "a", "sent_idx": 1}, {"case_id": "a", "sent_idx": 3},   # idx 2 excluded
            {"case_id": "a", "sent_idx": 4}, {"case_id": "b", "sent_idx": 0}, {"case_id": "b", "sent_idx": 2}]
    runs = crf_sequences(rows)
    assert [[r["sent_idx"] for r in run] for run in runs] == [[0, 1], [3, 4], [0], [2]]
    assert [run[0]["case_id"] for run in runs] == ["a", "a", "b", "b"]


def test_crf_features_use_own_sentence_only_and_crfsuite_fits():
    import sklearn_crfsuite
    f = sent_features("وحيث إن المحكمة قررت 2023.", 0, 40, True, False)
    assert f["BOS"] and not f["EOS"] and f["waw_start"] and f["has_digit"] and f["pos_decile"] == 0
    assert sent_features("x", 39, 40, False, True)["pos_decile"] == 9
    assert "w=المحكمة" in f and "first=وحيث" in f and not any(k.startswith("prev") or k.startswith("next") for k in f)
    seq = [{"case_id": "a", "sent_idx": 0, "text": "محاكم دبي"}, {"case_id": "a", "sent_idx": 1, "text": "حيث إن النيابة اتهمت"}, {"case_id": "a", "sent_idx": 2, "text": "حكمت المحكمة"}]
    gn = {"a": 3}
    X = [seq_features(seq, gn)] * 4; y = [["PREAMBLE", "FACTS", "DECISION"]] * 4
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=20).fit(X, y)
    assert list(crf.predict([seq_features(seq, gn)])[0]) == ["PREAMBLE", "FACTS", "DECISION"]
