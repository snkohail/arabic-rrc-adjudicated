"""S1 single-label CONTROL: S1-M1s / S1-M2s — softmax head trained with soft targets.

Everything is inherited from the frozen sigmoid pipeline (``rrc.s1_neural``): encoders, 510-token
windows with stride 128 and mean-pooled [CLS], AdamW + linear warmup, gradient clipping, batch
composition, the LR grid selected on VAL with seed 13 then seeds 37/73, five epochs with best-epoch
selection on VAL set-F1.  Only three things differ, by design:

  * output layer: 9-way softmax (same ``nn.Linear`` head, interpreted as softmax logits);
  * loss: cross-entropy against SOFT targets — for a region with gold set G the target is
    1/|G| on each role in G and 0 elsewhere (one-hot for single-role regions);
  * decoding: argmax only (no threshold, no multi-role output).

Evaluation is against the multi-label gold on TEST with the same document-clustered bootstrap
(5,000 replicates, seed 13) and paired against the frozen sigmoid model of the same seed under
(a) threshold decoding and (b) forced argmax.  New output directory ``reports/s1_softmax``;
nothing under ``reports/s1`` is read for writing or modified.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from . import bootstrap as B
from .config import DATA_DIR, REPORTS_DIR, ROLES
from .metrics import PerExample, decode_argmax, decode_multilabel, to_binary
from .s1_eval import load_predictions, save_predictions
from .s1_neural import CHUNK, HP as HP_SIGMOID, MODELS as SIGMOID_MODELS, SpanClassifier, _revision, collate, device, load_split, make_batches, seed_all, tokenize_examples

CONTROL = {"m1s_arabert": "m1_arabert", "m1s_arabertv02": "m1_arabertv02", "m2s_camelbert": "m2_camelbert"}      # control key -> frozen sigmoid key
MODELS = {k: SIGMOID_MODELS[v] for k, v in CONTROL.items()}
HP = {**{k: v for k, v in HP_SIGMOID.items() if k not in ("thresholds", "fallback")},
      "output_layer": "9-way softmax", "loss": "cross-entropy against soft targets (1/|G| on each gold role)",
      "decoding": "argmax only", "selection_criterion": "VAL set-F1 under argmax (same metric as the sigmoid models)"}
OUT = REPORTS_DIR / "s1_softmax"
PRIV = DATA_DIR / "predictions" / "s1_softmax"
FROZEN_OUT = REPORTS_DIR / "s1"
FROZEN_PRIV = DATA_DIR / "predictions"


def soft_targets(label_sets: List[List[str]]) -> torch.Tensor:
    """Soft-target construction.  Y is the multi-hot gold matrix (n, 9); each row is divided by
    its number of gold roles, so a region with gold set G gets 1/|G| on every role in G and 0
    elsewhere.  Single-role regions therefore get a one-hot target; a two-role region gets 0.5/0.5."""
    Y = torch.tensor(to_binary(label_sets), dtype=torch.float32)          # multi-hot, (n, |ROLES|)
    n_gold = Y.sum(dim=1, keepdim=True).clamp_min(1.0)                    # |G| per region (never 0 in S1)
    return Y / n_gold                                                     # rows sum to 1


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CE(target, softmax(logits)) = -sum_k target_k * log_softmax(logits)_k, averaged over the batch."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@torch.no_grad()
def predict(model, ex_windows, pad_id, dev) -> np.ndarray:
    model.eval()
    order = sorted(range(len(ex_windows)), key=lambda i: (len(ex_windows[i]), len(ex_windows[i][0])))
    P = np.zeros((len(ex_windows), len(ROLES)), dtype=np.float64)
    for b in make_batches(ex_windows, order):
        ids, att, owner, n = collate(ex_windows, b, pad_id, dev)
        P[b] = torch.softmax(model(ids, att, owner, n), dim=-1).float().cpu().numpy()
    return P


def _summ(P, rows, idx=None):
    Yg = to_binary([r["labels"] for r in rows]); pe = PerExample(decode_argmax(P), Yg)
    s = pe.summary(idx) if idx is not None else pe.summary()
    return {"set_f1": s["set_f1"], "exact_set_acc": s["exact_set_acc"], "macro_f1": s["macro_f1"], "n": s["n"]}


def train_one(model_key, lr, seed, train, val, test, tok, tr_w, va_w, te_w, log) -> dict:
    from transformers import get_linear_schedule_with_warmup
    seed_all(seed); dev = device()
    model = SpanClassifier(MODELS[model_key], HP["dropout"]).to(dev)
    Ttr = soft_targets([r["labels"] for r in train])
    rng = random.Random(seed)
    n_batches = len(make_batches(tr_w, list(range(len(train))))); total = n_batches * HP["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HP["weight_decay"])
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup_ratio"] * total), total)
    pad = tok.pad_token_id; history, best = [], None
    for epoch in range(1, HP["epochs"] + 1):
        model.train(); t0 = time.time(); tot = 0.0
        order = list(range(len(train))); rng.shuffle(order); batches = make_batches(tr_w, order)
        for step, b in enumerate(batches):
            ids, att, owner, n = collate(tr_w, b, pad, dev)
            loss = soft_cross_entropy(model(ids, att, owner, n), Ttr[b].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"]); opt.step(); sch.step(); tot += float(loss)
            if step % 100 == 0:
                log(f"    [{model_key} lr={lr} seed={seed}] epoch {epoch} step {step}/{len(batches)} loss {float(loss):.4f}")
        Pva = predict(model, va_w, pad, dev); s = _summ(Pva, val)
        rec = {"epoch": epoch, "train_loss": round(tot / len(batches), 4), **{f"val_{k}": v for k, v in s.items() if k != "n"}, "seconds": round(time.time() - t0)}
        history.append(rec); log(f"  [{model_key} lr={lr} seed={seed}] {rec}")
        if best is None or s["set_f1"] > best["val_set_f1"]:
            best = {"epoch": epoch, "val_set_f1": s["set_f1"], "val_macro_f1": s["macro_f1"], "val_exact_set_acc": s["exact_set_acc"],
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), "P_val": Pva}
    model.load_state_dict(best.pop("state")); P_val = best.pop("P_val")
    run = {"model": model_key, "checkpoint": MODELS[model_key], "lr": lr, "seed": seed, "history": history, "best": best}
    tag = f"{model_key}/lr{lr:g}_seed{seed}"
    save_predictions(val, P_val, PRIV / f"{tag}_val.jsonl"); run["val"] = _summ(P_val, val)
    if test is not None:
        P_te = predict(model, te_w, pad, dev); save_predictions(test, P_te, PRIV / f"{tag}_test.jsonl")
        run["test"] = evaluate_test(model_key, seed, test, P_te)
    del model
    if dev.type == "mps":
        torch.mps.empty_cache()
    return run


def _frozen_sigmoid(model_key: str, seed: int, test_rows):
    """TEST probabilities + VAL-selected threshold of the frozen sigmoid run with the same seed (read-only)."""
    fk = CONTROL[model_key]; summ = json.loads((FROZEN_OUT / fk / "runs.json").read_text())
    lr = summ["chosen_lr"]; rk = f"lr{lr:g}_seed{seed}"; thr = summ["runs"][rk]["best"]["val_threshold"]
    ids, P = load_predictions(FROZEN_PRIV / fk / f"{rk}_test.jsonl")
    assert ids == [r["id"] for r in test_rows], "frozen prediction order differs from TEST rows"
    return P, thr, rk


def evaluate_test(model_key: str, seed: int, rows, P: np.ndarray) -> dict:
    Yg = to_binary([r["labels"] for r in rows]); docs = [r["case_id"] for r in rows]
    multi = np.array([r["is_multi"] for r in rows], dtype=bool); mi = np.where(multi)[0]; mdocs = [docs[i] for i in mi]
    sm = PerExample(decode_argmax(P), Yg)
    Ps, thr, rk = _frozen_sigmoid(model_key, seed, rows)
    sg_thr = PerExample(decode_multilabel(Ps, thr), Yg); sg_arg = PerExample(decode_argmax(Ps), Yg)
    f = lambda pe, m=None: (lambda i: float(pe.set_f1[i].mean())) if m is None else (lambda i: float(pe.set_f1[m][i].mean()))
    e = lambda pe, m=None: (lambda i: float(pe.exact[i].mean())) if m is None else (lambda i: float(pe.exact[m][i].mean()))
    out = {"n": len(rows), "n_multi_role": int(multi.sum()), "paired_against": f"frozen {CONTROL[model_key]} {rk} (threshold {thr})",
           "softmax_argmax": {"all": sm.summary(), "multi_role": sm.summary(mi), "per_role_f1": sm.per_role_f1(),
                              "ci_all": {"set_f1": B.ci(docs, f(sm)), "exact_set_acc": B.ci(docs, e(sm))},
                              "ci_multi_role": {"set_f1": B.ci(mdocs, f(sm, mi)), "exact_set_acc": B.ci(mdocs, e(sm, mi))}},
           "reference_sigmoid_threshold": {"all": sg_thr.summary(), "multi_role": sg_thr.summary(mi)},
           "reference_sigmoid_argmax": {"all": sg_arg.summary(), "multi_role": sg_arg.summary(mi)},
           "paired_softmax_minus_sigmoid_threshold": {
               "multi_role": {"set_f1": B.paired_ci(mdocs, f(sm, mi), f(sg_thr, mi)), "exact_set_acc": B.paired_ci(mdocs, e(sm, mi), e(sg_thr, mi))},
               "all": {"set_f1": B.paired_ci(docs, f(sm), f(sg_thr)), "exact_set_acc": B.paired_ci(docs, e(sm), e(sg_thr))}},
           "paired_softmax_minus_sigmoid_argmax": {
               "multi_role": {"set_f1": B.paired_ci(mdocs, f(sm, mi), f(sg_arg, mi)), "exact_set_acc": B.paired_ci(mdocs, e(sm, mi), e(sg_arg, mi))},
               "all": {"set_f1": B.paired_ci(docs, f(sm), f(sg_arg)), "exact_set_acc": B.paired_ci(docs, e(sm), e(sg_arg))}}}
    return out


def _dump(out: Path, model_key, runs, selection, revision, chosen_lr):
    (out / "runs.json").write_text(json.dumps({"checkpoint": MODELS[model_key], "revision": revision, "control_of": CONTROL[model_key],
        "tokenizer": MODELS[model_key] + " (fast tokenizer, raw text, no Farasa segmentation)", "long_span_handling": CHUNK, "hyperparameters": HP,
        "lr_selection": selection, "chosen_lr": chosen_lr, "runs": runs}, indent=1))


def run_model(model_key: str, smoke: int = 0) -> None:
    from transformers import AutoTokenizer
    out = OUT / model_key; out.mkdir(parents=True, exist_ok=True)
    logf = DATA_DIR / "logs" / f"s1_softmax_{model_key}.log"; logf.parent.mkdir(parents=True, exist_ok=True)
    def log(m):
        print(m, flush=True); logf.open("a").write(m + "\n")
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if smoke:
        train, val = train[:smoke], val[:max(64, smoke // 4)]; out = OUT / f"_smoke_{model_key}"; out.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODELS[model_key], local_files_only=True)
    tr_w, va_w, te_w = tokenize_examples(tok, train), tokenize_examples(tok, val), tokenize_examples(tok, test)
    log(f"{model_key}: train {len(train)} val {len(val)} test {len(test)}; device {device()}; soft targets: multi-role train regions {int(sum(len(r['labels'])>1 for r in train))}")
    revision = _revision(MODELS[model_key]); sel_seed = HP["selection_seed"]; selection = []; runs: Dict[str, dict] = {}
    if smoke:
        HP["epochs"] = 1
    for lr in HP["lr_grid"]:
        r = train_one(model_key, lr, sel_seed, train, val, test, tok, tr_w, va_w, te_w, log); runs[f"lr{lr:g}_seed{sel_seed}"] = r
        selection.append({"lr": lr, "seed": sel_seed, "best_epoch": r["best"]["epoch"], "val_set_f1": r["best"]["val_set_f1"], "val_exact_set_acc": r["best"]["val_exact_set_acc"]})
        _dump(out, model_key, runs, selection, revision, None)
        if smoke:
            break
    chosen = sorted(selection, key=lambda s: (-s["val_set_f1"], s["lr"]))[0]["lr"]; log(f"{model_key}: chosen lr = {chosen}")
    for seed in HP["seeds"]:
        if seed == sel_seed or smoke:
            continue
        runs[f"lr{chosen:g}_seed{seed}"] = train_one(model_key, chosen, seed, train, val, test, tok, tr_w, va_w, te_w, log); _dump(out, model_key, runs, selection, revision, chosen)
    _dump(out, model_key, runs, selection, revision, chosen); log(f"{model_key}: done")


if __name__ == "__main__":
    key = sys.argv[1]; smoke = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_model(key, smoke=smoke)
