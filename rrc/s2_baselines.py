"""S2 lexical baselines.

S2-B0 majority: the most frequent TRAIN role for every sentence.
S2-B1 char n-gram TF-IDF + multinomial logistic regression; grid as S1-B0
       (ngram {(2,4),(3,5)} x C {0.3,1,3,10} x class_weight {none,balanced}), VAL macro-F1.
S2-B2 lexical CRF over sequences of grid-adjacent eligible sentences (sklearn-crfsuite, L-BFGS).
       An excluded sentence (AMBIGUOUS_PROJECTION / UNLABELED) BREAKS the sequence, so no transition
       is learned between non-adjacent sentences.  Features come from the sentence itself only (word
       unigrams, first/last words, first-word prefix, length buckets, digit/punctuation flags,
       initial waw), plus run-boundary flags (BOS/EOS of the adjacent run) and the sentence's
       position decile within the judgment's full grid; label transitions are learned by the CRF.
       No neighbouring-sentence text is used.  Grid c1, c2 in {0.05, 0.1, 0.5}, VAL macro-F1.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import DATA_DIR, REPORTS_DIR, ROLES
from .s2_data import crf_sequences, grid_sizes, load_s2
from .s2_eval import evaluate, save_predictions

OUT = REPORTS_DIR / "s2"
PRIV = DATA_DIR / "predictions"
B1_GRID = {"ngram_range": [(2, 4), (3, 5)], "C": [0.3, 1.0, 3.0, 10.0], "class_weight": [None, "balanced"]}
B1_FIXED = {"analyzer": "char_wb", "sublinear_tf": True, "min_df": 2, "max_features": 500000, "lowercase": False,
            "solver": "lbfgs", "multi_class": "multinomial", "max_iter": 2000, "seed": 13}
B2_GRID = {"c1": [0.05, 0.1, 0.5], "c2": [0.05, 0.1, 0.5]}
B2_FIXED = {"algorithm": "lbfgs", "max_iterations": 200, "all_possible_transitions": True}

_TOK = re.compile(r"\w+", re.UNICODE)


def _bucket(n: int, edges=(5, 10, 20, 40, 80)) -> str:
    for e in edges:
        if n < e:
            return f"<{e}"
    return f">={edges[-1]}"


def sent_features(text: str, grid_pos: int, grid_n: int, bos: bool, eos: bool) -> Dict[str, object]:
    """Own-sentence lexical features; grid_pos/grid_n = position within the judgment's full grid."""
    toks = _TOK.findall(text)
    f: Dict[str, object] = {"bias": 1.0, "n_tok": _bucket(len(toks)), "n_char": _bucket(len(text), (40, 80, 160, 320, 640)),
                            "has_digit": any(c.isdigit() for c in text), "has_colon": ":" in text, "has_paren": "(" in text or ")" in text,
                            "has_quote": '"' in text or "«" in text, "pos_decile": min(9, (10 * grid_pos) // max(1, grid_n)),
                            "BOS": bos, "EOS": eos}
    if toks:
        f["first=" + toks[0]] = 1.0; f["last=" + toks[-1]] = 1.0; f["pre3=" + toks[0][:3]] = 1.0
        f["waw_start"] = toks[0].startswith("و")
        if len(toks) > 1:
            f["first2=" + toks[0] + "_" + toks[1]] = 1.0
    for t in set(toks):
        f["w=" + t] = 1.0
    return f


def seq_features(seq: List[dict], grid_n: Dict[str, int]) -> List[dict]:
    n = len(seq)
    return [sent_features(r["text"], r["sent_idx"], grid_n[r["case_id"]], i == 0, i == n - 1) for i, r in enumerate(seq)]


def run_b0(train, val, test) -> dict:
    maj = Counter(r["label"] for r in train).most_common(1)[0][0]
    res = {"model": "S2-B0 majority", "majority_label": maj, "train_label_counts": dict(Counter(r["label"] for r in train))}
    for name, rows in (("val", val), ("test", test)):
        pred = [maj] * len(rows)
        save_predictions(rows, pred, PRIV / "s2_b0" / f"{name}.jsonl")
        res[name] = evaluate(rows, pred, with_ci=(name == "test"))
    return res


def run_b1(train, val, test, log) -> dict:
    ytr = [r["label"] for r in train]
    table = []
    for ng in B1_GRID["ngram_range"]:
        vec = TfidfVectorizer(analyzer=B1_FIXED["analyzer"], ngram_range=ng, sublinear_tf=True, min_df=B1_FIXED["min_df"],
                              max_features=B1_FIXED["max_features"], lowercase=False, dtype=np.float32)
        Xtr = vec.fit_transform([r["text"] for r in train]); Xva = vec.transform([r["text"] for r in val])
        for C in B1_GRID["C"]:
            for cw in B1_GRID["class_weight"]:
                clf = LogisticRegression(C=C, class_weight=cw, solver="lbfgs", max_iter=B1_FIXED["max_iter"], random_state=13).fit(Xtr, ytr)
                ev = evaluate(val, list(clf.predict(Xva)), with_ci=False)
                table.append({"ngram_range": list(ng), "C": C, "class_weight": cw or "none", "n_features": int(Xtr.shape[1]),
                              "val_macro_f1": ev["macro_f1"], "val_micro_f1": ev["micro_f1"]})
                log(f"  B1 ngram={ng} C={C} cw={cw or 'none'} VAL macro-F1={ev['macro_f1']:.4f} micro-F1={ev['micro_f1']:.4f}")
    chosen = sorted(table, key=lambda r: (-r["val_macro_f1"], r["C"], r["ngram_range"][1], r["class_weight"] != "none"))[0]
    ng = tuple(chosen["ngram_range"]); cw = None if chosen["class_weight"] == "none" else "balanced"
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ng, sublinear_tf=True, min_df=2, max_features=500000, lowercase=False, dtype=np.float32)
    Xtr = vec.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(C=chosen["C"], class_weight=cw, solver="lbfgs", max_iter=2000, random_state=13).fit(Xtr, ytr)
    res = {"model": "S2-B1 char n-gram TF-IDF + multinomial LR", "fixed": B1_FIXED,
           "grid": {"ngram_range": [list(x) for x in B1_GRID["ngram_range"]], "C": B1_GRID["C"], "class_weight": ["none", "balanced"]},
           "selection_criterion": "VAL macro-F1 (ties: smaller C, smaller n-gram, class_weight none)", "table": table, "chosen": chosen}
    for name, rows in (("val", val), ("test", test)):
        X = vec.transform([r["text"] for r in rows]); P = clf.predict_proba(X)
        pred = [clf.classes_[i] for i in P.argmax(1)]
        Pfull = np.zeros((len(rows), len(ROLES))); 
        for j, c in enumerate(clf.classes_):
            Pfull[:, ROLES.index(c)] = P[:, j]
        save_predictions(rows, pred, PRIV / "s2_b1" / f"{name}.jsonl", Pfull)
        res[name] = evaluate(rows, pred, with_ci=(name == "test"))
    return res


def run_b2(train, val, test, log) -> dict:
    import sklearn_crfsuite
    gn = {"train": grid_sizes("train"), "val": grid_sizes("val"), "test": grid_sizes("test")}
    tr_seqs = crf_sequences(train)
    Xtr = [seq_features(s, gn["train"]) for s in tr_seqs]; ytr = [[r["label"] for r in s] for s in tr_seqs]
    va_seqs = crf_sequences(val); Xva = [seq_features(s, gn["val"]) for s in va_seqs]
    val_rows = [r for s in va_seqs for r in s]
    seq_stats = {name: {"eligible_sentences": len(rows), "judgments": len({r["case_id"] for r in rows}), "crf_runs": len(crf_sequences(rows)),
                        "mean_run_length": round(len(rows) / max(1, len(crf_sequences(rows))), 2), "singleton_runs": sum(1 for s in crf_sequences(rows) if len(s) == 1)}
                 for name, rows in (("train", train), ("val", val), ("test", test))}
    log(f"  B2 sequence construction (excluded sentences break runs): {seq_stats}")
    table = []
    for c1 in B2_GRID["c1"]:
        for c2 in B2_GRID["c2"]:
            t0 = time.time()
            crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=B2_FIXED["max_iterations"], all_possible_transitions=True)
            crf.fit(Xtr, ytr)
            pred = [l for seq in crf.predict(Xva) for l in seq]
            ev = evaluate(val_rows, pred, with_ci=False)
            table.append({"c1": c1, "c2": c2, "val_macro_f1": ev["macro_f1"], "val_micro_f1": ev["micro_f1"], "seconds": round(time.time() - t0)})
            log(f"  B2 c1={c1} c2={c2} VAL macro-F1={ev['macro_f1']:.4f} micro-F1={ev['micro_f1']:.4f} ({time.time() - t0:.0f}s)")
    chosen = sorted(table, key=lambda r: (-r["val_macro_f1"], r["c1"], r["c2"]))[0]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=chosen["c1"], c2=chosen["c2"], max_iterations=B2_FIXED["max_iterations"], all_possible_transitions=True)
    crf.fit(Xtr, ytr)
    res = {"model": "S2-B2 lexical CRF over runs of grid-adjacent eligible sentences", "fixed": B2_FIXED, "grid": B2_GRID,
           "sequence_construction": "excluded (AMBIGUOUS_PROJECTION/UNLABELED) sentences break the sequence; no transition across non-adjacent sentences",
           "sequence_stats": seq_stats,
           "features": "own-sentence lexical features + run-boundary BOS/EOS + judgment-grid position decile; label transitions learned; no neighbour text",
           "selection_criterion": "VAL macro-F1 (ties: smaller c1, smaller c2)", "table": table, "chosen": chosen}
    for name, rows in (("val", val), ("test", test)):
        seqs = crf_sequences(rows); X = [seq_features(s, gn[name]) for s in seqs]
        flat_rows = [r for s in seqs for r in s]
        pred = [l for seq in crf.predict(X) for l in seq]
        marg = crf.predict_marginals(X)
        P = np.array([[m.get(r, 0.0) for r in ROLES] for seq in marg for m in seq])
        save_predictions(flat_rows, pred, PRIV / "s2_b2" / f"{name}.jsonl", P)
        res[name] = evaluate(flat_rows, pred, with_ci=(name == "test"))
    return res


def run_all(only: str = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logf = DATA_DIR / "logs" / "s2_baselines.log"; logf.parent.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with logf.open("a") as f:
            f.write(m + "\n")

    train, val, test = load_s2("train"), load_s2("val"), load_s2("test")
    log(f"S2 eligible sentences: train {len(train)} val {len(val)} test {len(test)}")
    for key, fn in (("b0", lambda: run_b0(train, val, test)), ("b1", lambda: run_b1(train, val, test, log)), ("b2", lambda: run_b2(train, val, test, log))):
        if only and key != only:
            continue
        t0 = time.time(); res = fn(); res["counts"] = {"train": len(train), "val": len(val), "test": len(test)}
        (OUT / key).mkdir(parents=True, exist_ok=True)
        (OUT / key / "results.json").write_text(json.dumps(res, indent=1))
        log(f"{key}: TEST macro-F1 {res['test']['macro_f1']} micro-F1 {res['test']['micro_f1']} ({time.time() - t0:.0f}s)")
