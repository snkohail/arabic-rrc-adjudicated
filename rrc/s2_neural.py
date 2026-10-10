"""S2-M1 (AraBERTv2) and S2-M2 (CAMeLBERT-MSA): sentence classifiers with a 9-way softmax.

Same encoder, window policy (510 tokens, stride 128, mean of per-window [CLS]), optimiser and
schedule as the S1 implementation; the head produces 9 logits trained with cross-entropy
and decoded by argmax.  Learning rate in {2e-5, 3e-5} selected on VAL macro-F1 with seed 13, then
seeds 37 and 73; up to 5 epochs, best VAL-macro-F1 epoch kept; TEST evaluated once per seed.
"""
from __future__ import annotations

import copy
import json
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from .config import DATA_DIR, REPORTS_DIR, ROLES
from .s1_neural import HP, MODELS, SpanClassifier, _revision, collate, device, make_batches, predict, seed_all, tokenize_examples, CHUNK
from .s2_data import load_s2
from .s2_eval import evaluate, save_predictions

OUT = REPORTS_DIR / "s2"
PRIV = DATA_DIR / "predictions"


@torch.no_grad()
def predict_softmax(model, ex_windows, pad_id, dev) -> np.ndarray:
    model.eval()
    order = sorted(range(len(ex_windows)), key=lambda i: (len(ex_windows[i]), len(ex_windows[i][0])))
    P = np.zeros((len(ex_windows), len(ROLES)), dtype=np.float64)
    for b in make_batches(ex_windows, order):
        ids, att, owner, n = collate(ex_windows, b, pad_id, dev)
        P[b] = torch.softmax(model(ids, att, owner, n), dim=-1).float().cpu().numpy()
    return P


def _labels(P: np.ndarray) -> List[str]:
    return [ROLES[i] for i in P.argmax(1)]


def train_one(model_key, lr, seed, train, val, test, tok, tr_w, va_w, te_w, log) -> dict:
    from transformers import get_linear_schedule_with_warmup
    seed_all(seed); dev = device()
    model = SpanClassifier(MODELS[model_key], HP["dropout"]).to(dev)
    ytr = torch.tensor([ROLES.index(r["label"]) for r in train], dtype=torch.long)
    rng = random.Random(seed)
    n_batches = len(make_batches(tr_w, list(range(len(train))))); total = n_batches * HP["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HP["weight_decay"])
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup_ratio"] * total), total)
    lossf = nn.CrossEntropyLoss(); pad = tok.pad_token_id
    history, best = [], None
    for epoch in range(1, HP["epochs"] + 1):
        model.train(); t0 = time.time(); tot = 0.0
        order = list(range(len(train))); rng.shuffle(order); batches = make_batches(tr_w, order)
        for step, b in enumerate(batches):
            ids, att, owner, n = collate(tr_w, b, pad, dev)
            loss = lossf(model(ids, att, owner, n), ytr[b].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"]); opt.step(); sch.step(); tot += float(loss)
            if step % 100 == 0:
                log(f"    [S2 {model_key} lr={lr} seed={seed}] epoch {epoch} step {step}/{len(batches)} loss {float(loss):.4f}")
        Pva = predict_softmax(model, va_w, pad, dev)
        ev = evaluate(val, _labels(Pva), with_ci=False)
        rec = {"epoch": epoch, "train_loss": round(tot / len(batches), 4), "val_macro_f1": ev["macro_f1"], "val_micro_f1": ev["micro_f1"], "seconds": round(time.time() - t0)}
        history.append(rec); log(f"  [S2 {model_key} lr={lr} seed={seed}] {rec}")
        if best is None or ev["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": ev["macro_f1"], "val_micro_f1": ev["micro_f1"],
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), "P_val": Pva}
    model.load_state_dict(best.pop("state")); P_val = best.pop("P_val")
    tag = f"s2_{model_key}/lr{lr:g}_seed{seed}"
    run = {"model": model_key, "checkpoint": MODELS[model_key], "lr": lr, "seed": seed, "history": history, "best": best}
    save_predictions(val, _labels(P_val), PRIV / f"{tag}_val.jsonl", P_val)
    run["val"] = evaluate(val, _labels(P_val), with_ci=False)
    P_te = predict_softmax(model, te_w, pad, dev)
    save_predictions(test, _labels(P_te), PRIV / f"{tag}_test.jsonl", P_te)
    run["test"] = evaluate(test, _labels(P_te), with_ci=True)
    del model
    if dev.type == "mps":
        torch.mps.empty_cache()
    return run


def run_model(model_key: str) -> None:
    from transformers import AutoTokenizer
    out = OUT / model_key; out.mkdir(parents=True, exist_ok=True)
    logf = DATA_DIR / "logs" / f"s2_{model_key}.log"; logf.parent.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with logf.open("a") as f:
            f.write(m + "\n")

    train, val, test = load_s2("train"), load_s2("val"), load_s2("test")
    tok = AutoTokenizer.from_pretrained(MODELS[model_key], local_files_only=True)
    tr_w, va_w, te_w = tokenize_examples(tok, train), tokenize_examples(tok, val), tokenize_examples(tok, test)
    nw = [len(w) for w in tr_w]
    log(f"S2 {model_key}: train {len(train)} val {len(val)} test {len(test)}; multi-window train sentences {sum(n > 1 for n in nw)}, max windows {max(nw)}; device {device()}")
    revision = _revision(MODELS[model_key])
    runs: Dict[str, dict] = {}; selection = []
    sel_seed = HP["selection_seed"]

    def dump(chosen):
        (out / "runs.json").write_text(json.dumps({"checkpoint": MODELS[model_key], "revision": revision, "tokenizer": MODELS[model_key] + " (fast tokenizer, raw text, no Farasa segmentation)",
                                                   "long_span_handling": CHUNK, "hyperparameters": {k: v for k, v in HP.items() if k not in ("thresholds", "fallback")} | {"decoding": "argmax over 9-way softmax", "loss": "cross-entropy"},
                                                   "selection_criterion": "VAL macro-F1", "lr_selection": selection, "chosen_lr": chosen, "runs": runs}, indent=1))

    for lr in HP["lr_grid"]:
        r = train_one(model_key, lr, sel_seed, train, val, test, tok, tr_w, va_w, te_w, log)
        runs[f"lr{lr:g}_seed{sel_seed}"] = r
        selection.append({"lr": lr, "seed": sel_seed, "best_epoch": r["best"]["epoch"], "val_macro_f1": r["best"]["val_macro_f1"], "val_micro_f1": r["best"]["val_micro_f1"]})
        dump(None)
    chosen_lr = sorted(selection, key=lambda s: (-s["val_macro_f1"], s["lr"]))[0]["lr"]
    log(f"S2 {model_key}: chosen lr = {chosen_lr}")
    for seed in HP["seeds"]:
        if seed == sel_seed:
            continue
        runs[f"lr{chosen_lr:g}_seed{seed}"] = train_one(model_key, chosen_lr, seed, train, val, test, tok, tr_w, va_w, te_w, log)
        dump(chosen_lr)
    dump(chosen_lr); log(f"S2 {model_key}: done")
