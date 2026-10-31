"""S1-M1 (AraBERTv2) and S1-M2 (CAMeLBERT-MSA): span encoder + 9 sigmoid outputs.

Identical pipeline for both encoders.

Long-span handling (no silent truncation): a span is tokenised without special tokens;
if it exceeds 510 tokens it is cut into deterministic overlapping windows of 510 tokens
(stride 128, i.e. step 382; the last window always reaches the span end).  Every window
is encoded as [CLS] window [SEP]; the [CLS] hidden states of all windows are averaged into
one span representation, which feeds dropout + a linear layer with 9 logits.  Gradients
flow through every window (gradient checkpointing keeps memory bounded).  Spans <= 510
tokens are a single window, so the model is the usual [CLS] classifier for them.

Protocol: TRAIN fit; per epoch the VAL set-F1 at the best global threshold (grid as B0)
is computed and the best epoch is kept; hyper-parameter (learning rate) is selected on
VAL with seed 13 and then reused for seeds 37 and 73; each seed's model uses its own
VAL-selected global threshold; TEST is evaluated once per seed.
Fine-tuned weights are not saved; per-example probabilities are saved privately.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .config import DATA_DIR, DERIVED_DIR, REPORTS_DIR, ROLES, TOKENIZER_ARABERT, TOKENIZER_ARABERT_V02, TOKENIZER_CAMELBERT
from .metrics import PerExample, decode_multilabel, to_binary
from .s1_eval import evaluate_split, save_predictions

MODELS = {"m1_arabert": TOKENIZER_ARABERT, "m1_arabertv02": TOKENIZER_ARABERT_V02, "m2_camelbert": TOKENIZER_CAMELBERT}
CHUNK = {"window_tokens": 510, "stride_tokens": 128, "pooling": "mean of per-window [CLS] hidden states"}
HP = {"lr_grid": [2e-5, 3e-5], "epochs": 5, "batch_examples": 16, "max_windows_per_batch": 48, "warmup_ratio": 0.1,
      "weight_decay": 0.01, "dropout": 0.1, "grad_clip": 1.0, "optimizer": "AdamW", "schedule": "linear decay with warmup",
      "loss": "BCEWithLogits (mean)", "selection_seed": 13, "seeds": [13, 37, 73], "precision": "fp32",
      "thresholds": [round(0.05 * k, 2) for k in range(1, 20)], "fallback": "argmax when no role passes the threshold"}
OUT = REPORTS_DIR / "s1"
PRIV = DATA_DIR / "predictions"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_split(name: str) -> List[dict]:
    with (DERIVED_DIR / "s1" / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def windows(ids: List[int], window: int = CHUNK["window_tokens"], stride: int = CHUNK["stride_tokens"]) -> List[List[int]]:
    if len(ids) <= window:
        return [ids]
    step = window - stride
    out, s = [], 0
    while True:
        e = min(s + window, len(ids))
        out.append(ids[s:e])
        if e == len(ids):
            return out
        s += step


def tokenize_examples(tok, rows: List[dict]) -> List[List[List[int]]]:
    enc = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
    cls, sep = tok.cls_token_id, tok.sep_token_id
    return [[[cls] + w + [sep] for w in windows(ids)] for ids in enc]


def make_batches(ex_windows: List[List[List[int]]], order: List[int]) -> List[List[int]]:
    batches, cur, cur_w = [], [], 0
    for i in order:
        w = len(ex_windows[i])
        if cur and (cur_w + w > HP["max_windows_per_batch"] or len(cur) >= HP["batch_examples"]):
            batches.append(cur); cur, cur_w = [], 0
        cur.append(i); cur_w += w
    if cur:
        batches.append(cur)
    return batches


def collate(ex_windows, idx: List[int], pad_id: int, dev):
    seqs, owner = [], []
    for j, i in enumerate(idx):
        for w in ex_windows[i]:
            seqs.append(w); owner.append(j)
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
    return ids.to(dev), att.to(dev), torch.tensor(owner, dtype=torch.long).to(dev), len(idx)


class SpanClassifier(nn.Module):
    def __init__(self, name: str, dropout: float):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(name, local_files_only=True)
        self.encoder.gradient_checkpointing_enable()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(ROLES))

    def forward(self, ids, att, owner, n_examples):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]       # (windows, hidden)
        agg = torch.zeros((n_examples, h.shape[1]), device=h.device, dtype=h.dtype).index_add_(0, owner, h)
        cnt = torch.zeros(n_examples, device=h.device, dtype=h.dtype).index_add_(0, owner, torch.ones_like(owner, dtype=h.dtype))
        return self.head(self.drop(agg / cnt.unsqueeze(1)))


@torch.no_grad()
def predict(model, ex_windows, pad_id, dev) -> np.ndarray:
    model.eval()
    order = sorted(range(len(ex_windows)), key=lambda i: (len(ex_windows[i]), len(ex_windows[i][0])))
    P = np.zeros((len(ex_windows), len(ROLES)), dtype=np.float64)
    for b in make_batches(ex_windows, order):
        ids, att, owner, n = collate(ex_windows, b, pad_id, dev)
        P[b] = torch.sigmoid(model(ids, att, owner, n)).float().cpu().numpy()
    return P


def best_threshold(P: np.ndarray, Yg: np.ndarray) -> Tuple[float, dict]:
    best = None
    for thr in HP["thresholds"]:
        s = PerExample(decode_multilabel(P, thr), Yg).summary()
        if best is None or s["set_f1"] > best[1]["set_f1"]:
            best = (thr, s)
    return best


def train_one(model_key: str, lr: float, seed: int, train, val, test, tok, tr_w, va_w, te_w, log) -> dict:
    from transformers import get_linear_schedule_with_warmup
    seed_all(seed)
    dev = device()
    model = SpanClassifier(MODELS[model_key], HP["dropout"]).to(dev)
    Ytr = torch.tensor(to_binary([r["labels"] for r in train]), dtype=torch.float32)
    Yva = to_binary([r["labels"] for r in val])
    rng = random.Random(seed)
    n_batches = len(make_batches(tr_w, list(range(len(train)))))
    total = n_batches * HP["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=HP["weight_decay"])
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup_ratio"] * total), total)
    lossf = nn.BCEWithLogitsLoss()
    pad = tok.pad_token_id
    history, best = [], None
    for epoch in range(1, HP["epochs"] + 1):
        model.train(); t0 = time.time(); tot = 0.0
        order = list(range(len(train))); rng.shuffle(order)
        batches = make_batches(tr_w, order)
        for step, b in enumerate(batches):
            ids, att, owner, n = collate(tr_w, b, pad, dev)
            logits = model(ids, att, owner, n)
            loss = lossf(logits, Ytr[b].to(dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step(); sch.step(); tot += float(loss)
            if step % 100 == 0:
                log(f"    [{model_key} lr={lr} seed={seed}] epoch {epoch} step {step}/{len(batches)} loss {float(loss):.4f}")
        Pva = predict(model, va_w, pad, dev)
        thr, s = best_threshold(Pva, Yva)
        rec = {"epoch": epoch, "train_loss": round(tot / len(batches), 4), "val_threshold": thr, **{f"val_{k}": v for k, v in s.items() if k != "n"},
               "seconds": round(time.time() - t0)}
        history.append(rec); log(f"  [{model_key} lr={lr} seed={seed}] {rec}")
        if best is None or s["set_f1"] > best["val_set_f1"]:
            best = {"epoch": epoch, "val_threshold": thr, "val_set_f1": s["set_f1"], "val_macro_f1": s["macro_f1"], "val_exact_set_acc": s["exact_set_acc"],
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), "P_val": Pva}
    model.load_state_dict(best.pop("state"))
    P_val = best.pop("P_val")
    run = {"model": model_key, "checkpoint": MODELS[model_key], "lr": lr, "seed": seed, "history": history, "best": best}
    tag = f"{model_key}/lr{lr:g}_seed{seed}"
    save_predictions(val, P_val, PRIV / f"{tag}_val.jsonl")
    run["val"] = evaluate_split(val, P_val, best["val_threshold"], with_ci=False)
    if test is not None:
        P_te = predict(model, te_w, pad, dev)
        save_predictions(test, P_te, PRIV / f"{tag}_test.jsonl")
        run["test"] = evaluate_split(test, P_te, best["val_threshold"], with_ci=True)
    del model
    if dev.type == "mps":
        torch.mps.empty_cache()
    return run


def run_model(model_key: str, seeds=None, lrs=None, epochs=None, smoke: int = 0) -> None:
    from transformers import AutoTokenizer
    if epochs:
        HP["epochs"] = epochs
    out = OUT / model_key; out.mkdir(parents=True, exist_ok=True)
    logf = (DATA_DIR / "logs" / f"s1_{model_key}.log"); logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with logf.open("a") as f:
            f.write(msg + "\n")

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    if smoke:
        train, val, test = train[:smoke], val[:max(64, smoke // 4)], None
    tok = AutoTokenizer.from_pretrained(MODELS[model_key], local_files_only=True)
    tr_w, va_w = tokenize_examples(tok, train), tokenize_examples(tok, val)
    te_w = tokenize_examples(tok, test) if test is not None else None
    nw = [len(w) for w in tr_w]
    log(f"{model_key}: train {len(train)} val {len(val)} test {len(test) if test else 0}; multi-window train examples {sum(n > 1 for n in nw)}, max windows {max(nw)}; device {device()}")
    revision = _revision(MODELS[model_key])
    # 1) learning-rate selection on VAL with the selection seed
    sel_seed = HP["selection_seed"]
    lrs = lrs or HP["lr_grid"]
    selection = []
    runs: Dict[str, dict] = {}
    for lr in lrs:
        r = train_one(model_key, lr, sel_seed, train, val, test, tok, tr_w, va_w, te_w, log)
        runs[f"lr{lr:g}_seed{sel_seed}"] = r
        selection.append({"lr": lr, "seed": sel_seed, "best_epoch": r["best"]["epoch"], "val_threshold": r["best"]["val_threshold"],
                          "val_set_f1": r["best"]["val_set_f1"], "val_macro_f1": r["best"]["val_macro_f1"], "val_exact_set_acc": r["best"]["val_exact_set_acc"]})
        _dump(out, runs, selection, revision, chosen_lr=None)
    chosen_lr = sorted(selection, key=lambda s: (-s["val_set_f1"], s["lr"]))[0]["lr"]
    log(f"{model_key}: chosen lr = {chosen_lr}")
    # 2) remaining seeds with the chosen lr
    for seed in (seeds or HP["seeds"]):
        if seed == sel_seed:
            continue
        runs[f"lr{chosen_lr:g}_seed{seed}"] = train_one(model_key, chosen_lr, seed, train, val, test, tok, tr_w, va_w, te_w, log)
        _dump(out, runs, selection, revision, chosen_lr)
    _dump(out, runs, selection, revision, chosen_lr)
    log(f"{model_key}: done")


def _revision(name: str) -> str:
    """Commit hash of the cached checkpoint (refs/main of the local HF hub cache)."""
    import os
    cache = None
    try:
        from huggingface_hub import constants
        cache = getattr(constants, "HF_HUB_CACHE", None) or getattr(constants, "HUGGINGFACE_HUB_CACHE", None)
    except Exception:
        pass
    cache = Path(os.environ.get("HF_HUB_CACHE") or cache or Path.home() / ".cache" / "huggingface" / "hub")
    p = cache / ("models--" + name.replace("/", "--")) / "refs" / "main"
    return p.read_text().strip() if p.exists() else "unknown"


def _dump(out: Path, runs: dict, selection: list, revision: str, chosen_lr):
    summary = {"checkpoint": MODELS[out.name], "revision": revision, "tokenizer": MODELS[out.name] + " (fast tokenizer, raw text, no Farasa segmentation)",
               "long_span_handling": CHUNK, "hyperparameters": HP, "lr_selection": selection, "chosen_lr": chosen_lr,
               "runs": {k: {kk: vv for kk, vv in v.items() if kk != "history"} | {"history": v["history"]} for k, v in runs.items()}}
    (out / "runs.json").write_text(json.dumps(summary, indent=1))
