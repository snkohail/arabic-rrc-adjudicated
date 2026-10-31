"""Zero-shot LLM baseline (ALLaM-7B-Instruct-preview, open weights, run locally on MPS; no network at inference).

Protocol (pre-declared):
  * prompt v1 (S1 format example showed two slots -> model copied the shape, cardinality 1.96 on VAL) and
    prompt v2 (S1 format line asks for 'one or more codes' with no slot example); S2 prompt unchanged.
  * prompt = fixed system instruction + the nine role codes with the guideline definitions (configs/role_definitions.json)
    + the passage; S1 asks for ALL applicable roles, S2 asks for exactly one role; output = JSON list of codes.
  * greedy decoding, max 40 new tokens, one deterministic run (no seeds).
  * passages longer than 1,500 ALLaM tokens are truncated to the first 1,200 + last 300 tokens (reported).
  * VAL stage: 100 S1 regions + 100 S2 sentences (seed 13) to check parse rate and speed only; TEST run once.
  * unparsable output = empty prediction (S1) / most frequent TRAIN role (S2); the failure rate is reported.
  * scoring reuses the encoder scoring code: set-F1 / macro-F1 / exact-set, slices, document-clustered bootstrap (5,000, seed 13),
    paired contrasts against B0 and the encoder seeds on identical resamples.
Outputs: <data_dir>/predictions/llm_allam/{s1,s2}_{val,test}.jsonl (+ *_raw.jsonl) and results here.
"""
import json, re, sys, time, random, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrc import config as C
from rrc import bootstrap as B
from rrc.metrics import PerExample, decode_multilabel, slice_masks, to_binary, micro_f1
from rrc.s1_eval import load_predictions as s1_load
from rrc.s2_data import load_s2
from rrc.s2_eval import evaluate as s2_evaluate, paired_macro_f1, load_predictions as s2_load

ROLES = C.ROLES
MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"
HERE = C.REPORTS_DIR / "llm_baseline"
PRED = C.DATA_DIR / "predictions" / "llm_allam"
DEFS = json.loads((C.CONFIG_DIR / "role_definitions.json").read_text(encoding="utf-8"))
MAX_TOK, HEAD, TAIL, MAX_NEW = 1500, 1200, 300, 40
PROMPT = "v1"
CODE_RE = re.compile("|".join(sorted(ROLES, key=len, reverse=True)))

SYSTEM = ("أنت خبير في تحليل الأحكام القضائية العربية. مهمتك تصنيف الوظيفة البلاغية (الدور الخطابي) لمقطع من حكم قضائي "
          "وفق تعريفات محددة. أجب بالرموز الإنجليزية فقط داخل قائمة JSON، دون أي شرح.\n"
          "You are an expert in Arabic court judgments. Classify the rhetorical role of a passage using the given "
          "definitions. Answer with the English role codes only, as a JSON list, with no explanation.")

def defs_block():
    return "\n".join(f"- {r}: {DEFS[r]['ar']} / {DEFS[r]['en']}" for r in ROLES)

def user_msg(passage, multi):
    q = ("ما الوظيفة أو الوظائف البلاغية التي تنطبق فعلاً على المقطع التالي؟ اختر كل ما ينطبق من الرموز التسعة.\n"
         "Which rhetorical role or roles genuinely apply to the passage? Select every code that applies." if multi else
         "ما الوظيفة البلاغية الواحدة الأنسب للجملة التالية؟ اختر رمزاً واحداً فقط من الرموز التسعة.\n"
         "Which single rhetorical role best describes the sentence? Select exactly one code.")
    if multi and PROMPT == "v2":
        tail = "الإجابة: قائمة JSON بالرموز المنطبقة فقط (رمز واحد أو أكثر) / Answer: a JSON list containing one or more of the codes, nothing else:"
    else:
        fmt = '["CODE", "CODE"]' if multi else '["CODE"]'
        tail = f"الإجابة بصيغة {fmt} فقط:"
    return f"الأدوار / Roles:\n{defs_block()}\n\n{q}\n\nالمقطع / Passage:\n«{passage}»\n\n{tail}"

def parse(out, multi):
    m = re.search(r"\[[^\]]*\]", out)
    codes = CODE_RE.findall(m.group(0) if m else out)
    seen = []
    for c in codes:
        if c not in seen: seen.append(c)
    if not multi: seen = seen[:1]
    return [r for r in ROLES if r in seen]

class LLM:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, local_files_only=True).to("mps").eval()
        self.n_trunc = 0
    def truncate(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= MAX_TOK: return text, False
        self.n_trunc += 1
        return self.tok.decode(ids[:HEAD]) + " … " + self.tok.decode(ids[-TAIL:]), True
    def ask(self, passage, multi):
        passage, tr = self.truncate(passage)
        prompt = self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(passage, multi)}], tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("mps")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, parse(text, multi), tr

def rows_s1(split):
    return [json.loads(l) for l in (C.DERIVED_DIR / "s1" / f"{split}.jsonl").open(encoding="utf-8")]

def run(stage, items, multi, out_prefix, llm):
    """Resumable: appends one line per item to *_raw.jsonl; skips ids already done."""
    PRED.mkdir(parents=True, exist_ok=True)
    raw_p = PRED / f"{out_prefix}_raw.jsonl"
    done = {json.loads(l)["id"] for l in raw_p.open()} if raw_p.exists() else set()
    t0, n = time.time(), 0
    with raw_p.open("a", encoding="utf-8") as f:
        for it in items:
            if it["id"] in done: continue
            text, labels, tr = llm.ask(it["text"], multi)
            f.write(json.dumps({"id": it["id"], "case_id": it["case_id"], "output": text, "pred": labels, "truncated": tr}, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 25 == 0: print(f"[{stage}] {len(done)+n}/{len(items)}  {(time.time()-t0)/n:.1f}s/item", flush=True)
    raw = {json.loads(l)["id"]: json.loads(l) for l in raw_p.open(encoding="utf-8")}
    return [raw[it["id"]] for it in items]

def write_s1_pred(items, raw, path):
    with path.open("w") as f:
        for it, r in zip(items, raw):
            f.write(json.dumps({"id": it["id"], "case_id": it["case_id"], "gold": it["labels"], "probs": [1.0 if l in r["pred"] else 0.0 for l in ROLES]}) + "\n")

def write_s2_pred(items, raw, fallback, path):
    with path.open("w") as f:
        for it, r in zip(items, raw):
            f.write(json.dumps({"id": it["id"], "case_id": it["case_id"], "gold": it["label"], "pred": r["pred"][0] if r["pred"] else fallback}) + "\n")

def majority_train_role():
    from collections import Counter
    return Counter(r["label"] for r in load_s2("train")).most_common(1)[0][0]

def s1_metrics(items, P, with_ci=True):
    Yg = to_binary([r["labels"] for r in items]); docs = [r["case_id"] for r in items]
    pe = PerExample(decode_multilabel(P, 0.5, fallback_argmax=False), Yg)
    res = {"n": len(items), **pe.summary(), "micro_f1": round(micro_f1(pe), 4), "per_role_f1": pe.per_role_f1(),
           "slices": {k: pe.summary(np.where(m)[0]) for k, m in slice_masks(items).items()},
           "pred_cardinality_mean": round(float(P.sum(1).mean()), 3), "empty_predictions": int((P.sum(1) == 0).sum())}
    if with_ci:
        res["ci"] = {"set_f1": B.ci(docs, lambda i: float(pe.set_f1[i].mean())), "macro_f1": B.ci(docs, lambda i: pe.macro_f1(i)),
                     "exact_set_acc": B.ci(docs, lambda i: float(pe.exact[i].mean()))}
    return res, pe

def paired_s1(items, pe_a, P_b, thr_b):
    Yg = to_binary([r["labels"] for r in items]); docs = [r["case_id"] for r in items]
    pb = PerExample(decode_multilabel(P_b, thr_b), Yg)
    return B.paired_ci(docs, lambda i: float(pe_a.set_f1[i].mean()), lambda i: float(pb.set_f1[i].mean()))

def score():
    out = {"model": MODEL_ID, "protocol": __doc__}
    # ---- S1 (every prompt version that has a complete raw file)
    items = rows_s1("test"); pred_dir = C.DATA_DIR / "predictions"
    thr = {"m1_arabertv02": {13: .35, 37: .3, 73: .4}, "m2_camelbert": {13: .3, 37: .3, 73: .15}, "m1_arabert": {13: .2, 37: .3, 73: .3}}
    for ver, raw_name, pred_name in (("v1", "s1_test_raw.jsonl", "s1_test.jsonl"), ("v2", "s1_test_v2_raw.jsonl", "s1_test_v2.jsonl")):
        if not (PRED / raw_name).exists(): continue
        rawd = {json.loads(l)["id"]: json.loads(l) for l in (PRED / raw_name).open(encoding="utf-8")}
        if len(rawd) < len(items): print(f"S1 {ver}: incomplete ({len(rawd)}/{len(items)}), skipped"); continue
        raw = [rawd[it["id"]] for it in items]
        write_s1_pred(items, raw, PRED / pred_name); _, P = s1_load(PRED / pred_name)
        res, pe = s1_metrics(items, P)
        res["parse_failures"] = sum(1 for r in raw if not r["pred"]); res["truncated_inputs"] = sum(1 for r in raw if r["truncated"])
        _, Pb0 = s1_load(pred_dir / "s1_b0" / "test.jsonl"); res["paired_vs_b0"] = paired_s1(items, pe, Pb0, 0.3)
        for mk, seeds in thr.items():
            res[f"paired_vs_{mk}"] = {}; lr = json.loads((C.REPORTS_DIR / "s1" / mk / "runs.json").read_text())["chosen_lr"]
            for sd, t in seeds.items():
                _, Pm = s1_load(pred_dir / mk / f"lr{lr:g}_seed{sd}_test.jsonl"); res[f"paired_vs_{mk}"][sd] = paired_s1(items, pe, Pm, t)
        out[f"s1_{ver}"] = res
    res = out.get("s1_v2") or out.get("s1_v1")
    # ---- S2
    s2 = load_s2("test"); raw2 = {json.loads(l)["id"]: json.loads(l) for l in (PRED / "s2_test_raw.jsonl").open(encoding="utf-8")}
    raw2 = [raw2[it["id"]] for it in s2]; fb = majority_train_role()
    write_s2_pred(s2, raw2, fb, PRED / "s2_test.jsonl"); pred = s2_load(PRED / "s2_test.jsonl")
    r2 = s2_evaluate(s2, pred, with_ci=True); r2["parse_failures"] = sum(1 for r in raw2 if not r["pred"]); r2["fallback_role"] = fb
    for bk in ("s2_b0", "s2_b1", "s2_b2"):
        r2[f"paired_vs_{bk}"] = paired_macro_f1(s2, pred, s2_load(pred_dir / bk / "test.jsonl"))
    for mk in ("m1_arabertv02", "m2_camelbert", "m1_arabert"):
        lr = json.loads((C.REPORTS_DIR / "s2" / mk / "runs.json").read_text())["chosen_lr"]
        r2[f"paired_vs_s2_{mk}"] = {sd: paired_macro_f1(s2, pred, s2_load(pred_dir / f"s2_{mk}" / f"lr{lr:g}_seed{sd}_test.jsonl")) for sd in (13, 37, 73)}
    out["s2"] = r2
    (HERE / "results_allam.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"s1": {k: res[k] for k in ("set_f1", "macro_f1", "exact_set_acc", "micro_f1", "parse_failures", "pred_cardinality_mean")}, "s1_ci": res["ci"]["set_f1"],
                      "s1_vs_b0": res["paired_vs_b0"], "s2": {k: r2[k] for k in ("macro_f1", "micro_f1", "accuracy", "parse_failures")}, "s2_ci": r2["ci"]["macro_f1"]}, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["smoke", "val", "val2", "s1", "s2", "score"]); ap.add_argument("--prompt", choices=["v1", "v2"], default="v1"); a = ap.parse_args()
    PROMPT = a.prompt
    if a.stage == "score": score(); sys.exit()
    llm = LLM(); print("model loaded", flush=True)
    if a.stage == "smoke":
        for it in rows_s1("val")[:3]:
            t0 = time.time(); text, labels, tr = llm.ask(it["text"], True); print(f"{time.time()-t0:.1f}s gold={it['labels']} pred={labels} raw={text!r}")
        it = load_s2("val")[5]; text, labels, tr = llm.ask(it["text"], False); print(f"S2 gold={it['label']} pred={labels} raw={text!r}")
    elif a.stage == "val":
        rng = random.Random(13); v1 = rng.sample(rows_s1("val"), 100); v2 = rng.sample(load_s2("val"), 100)
        t0 = time.time(); r1 = run("val-s1", v1, True, "s1_val", llm); t1 = time.time(); r2 = run("val-s2", v2, False, "s2_val", llm); t2 = time.time()
        write_s1_pred(v1, r1, PRED / "s1_val.jsonl"); _, P = s1_load(PRED / "s1_val.jsonl"); m1, _ = s1_metrics(v1, P, with_ci=False)
        fb = majority_train_role(); write_s2_pred(v2, r2, fb, PRED / "s2_val.jsonl"); m2 = s2_evaluate(v2, s2_load(PRED / "s2_val.jsonl"), with_ci=False)
        print(json.dumps({"s1_parse_failures": sum(1 for r in r1 if not r["pred"]), "s1_sec_per_item": round((t1-t0)/100, 2), "s1_set_f1_info_only": m1["set_f1"], "s1_cardinality": m1["pred_cardinality_mean"],
                          "s2_parse_failures": sum(1 for r in r2 if not r["pred"]), "s2_sec_per_item": round((t2-t1)/100, 2), "s2_macro_f1_info_only": m2["macro_f1"], "truncated": llm.n_trunc}, indent=1))
    elif a.stage == "val2":
        rng = random.Random(13); v1 = rng.sample(rows_s1("val"), 100)
        t0 = time.time(); r1 = run("val2-s1", v1, True, f"s1_val_{PROMPT}", llm)
        write_s1_pred(v1, r1, PRED / f"s1_val_{PROMPT}.jsonl"); _, P = s1_load(PRED / f"s1_val_{PROMPT}.jsonl"); m1, _ = s1_metrics(v1, P, with_ci=False)
        print(json.dumps({"prompt": PROMPT, "s1_parse_failures": sum(1 for r in r1 if not r["pred"]), "s1_sec_per_item": round((time.time()-t0)/100, 2), "s1_set_f1_info_only": m1["set_f1"], "s1_cardinality": m1["pred_cardinality_mean"]}, indent=1))
    elif a.stage == "s1":
        run("s1-test", rows_s1("test"), True, "s1_test" if PROMPT == "v1" else f"s1_test_{PROMPT}", llm); print(f"S1 TEST done ({PROMPT})", flush=True)
    elif a.stage == "s2":
        run("s2-test", load_s2("test"), False, "s2_test", llm); print("S2 TEST done", flush=True)
