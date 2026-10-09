"""Blind expert validation audit of the task / annotation formulation (NOT a model-error audit).

TRAIN + VALIDATION judgments only; TEST is never touched; model predictions are not used.
Deterministic seed-13 stratified samples (20 each):
  1  single-role, non-nested adjudicated regions
  2  adjudicated multi-role regions
  3  nested structures (outer ⊃ inner, distinct label sets; pairs with single-role spans preferred)
  4  S2 AMBIGUOUS_PROJECTION sentences
Caps keep any judgment or role from dominating a stratum (≤ 2 items per judgment, role-key caps);
caps are relaxed step-wise only if a stratum cannot be filled.

Outputs: PRIVATE (text-bearing) items/key/HTML under <data_dir>/expert_audit/<version>/ and a
PUBLIC manifest (ids, offsets, stratum, hashes; no text, no labels) under reports/expert_audit/.
The HTML shows Arabic context with the span(s) highlighted and reveals no A/B/C labels, no
provenance, no predictions.  ``score`` computes the pre-declared summary from the expert's JSON.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import html
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from . import config as C
from .gold import doc_structure
from .io import load_c_only
from .metrics import PerExample, to_binary
from .split import load_manifest

VERSION = "v1"
SEED = 13
N_PER_STRATUM = 20
JUDGMENT_CAP = 2
ROLE_KEY_CAPS = {1: 4, 2: 6, 3: 5, 4: 6}
CTX = 300
STRATA = {1: "single_role_non_nested_region", 2: "multi_role_region", 3: "nested_structure", 4: "ambiguous_projection_sentence"}
DEFS_PATH = C.CONFIG_DIR / "role_definitions.json"
EXT_CTX = 1500  # "Show more context" window (characters each side)
PUB = C.REPORTS_DIR / "expert_audit" / VERSION
PRIV = C.DATA_DIR / "expert_audit" / VERSION


def role_definitions() -> dict:
    """Role definitions displayed to the expert (guideline-derived; status recorded in the file)."""
    return json.loads(DEFS_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- sampling
def _snap(text: str, i: int, forward: bool) -> int:
    n = len(text)
    i = max(0, min(n, i))
    if forward:
        while i < n and not text[i].isspace():
            i += 1
    else:
        while i > 0 and not text[i - 1].isspace():
            i -= 1
    return i


def _window(text: str, b: int, e: int, ctx: int = CTX) -> Tuple[str, str, str]:
    lb, le = _snap(text, b - ctx, False), _snap(text, e + ctx, True)
    return text[lb:b], text[b:e], text[e:le]


def _pick(cands: List[dict], n: int, role_key, role_cap: int, rng: random.Random) -> List[dict]:
    order = list(cands)
    rng.shuffle(order)
    chosen: List[dict] = []
    for jcap, rcap in ((JUDGMENT_CAP, role_cap), (JUDGMENT_CAP + 1, role_cap + 2), (10 ** 6, 10 ** 6)):
        jc, rc = Counter(x["case_id"] for x in chosen), Counter(role_key(x) for x in chosen)
        for x in order:
            if len(chosen) >= n:
                break
            if x in chosen or jc[x["case_id"]] >= jcap or rc[role_key(x)] >= rcap:
                continue
            chosen.append(x); jc[x["case_id"]] += 1; rc[role_key(x)] += 1
        if len(chosen) >= n:
            break
    return chosen[:n]


def _candidates(docs: Dict[str, "Document"]):
    s1, s2, s3 = [], [], []
    for cid in sorted(docs):
        d = docs[cid]
        st = doc_structure(d)
        nested = st.nested_indices
        for k, r in enumerate(st.regions):
            base = {"case_id": cid, "begin": r.begin, "end": r.end, "c_labels": list(r.labels)}
            if r.is_multi:
                s2.append(base)
            elif k not in nested:
                s1.append(base)
        for o, i in st.nesting:
            ro, ri = st.regions[o], st.regions[i]
            if set(ro.labels) == set(ri.labels):
                continue
            s3.append({"case_id": cid, "outer": [ro.begin, ro.end], "inner": [ri.begin, ri.end], "c_outer_labels": list(ro.labels),
                       "c_inner_labels": list(ri.labels), "both_single": (not ro.is_multi) and (not ri.is_multi)})
    return s1, s2, s3


def _ambiguous(split_ids: set) -> List[dict]:
    out = []
    for name in ("train", "val"):
        with (C.DERIVED_DIR / "s2" / f"{name}.jsonl").open(encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                if r["status"] == "AMBIGUOUS_PROJECTION" and r["case_id"] in split_ids:
                    out.append({"case_id": r["case_id"], "sent_idx": r["sent_idx"], "begin": r["begin"], "end": r["end"],
                                "c_max_roles": r["max_roles"], "c_roles_intersecting": r["roles_intersecting"], "ambiguity_kind": r["ambiguity_kind"]})
    return out


def build(force: bool = False) -> dict:
    if (PUB / "manifest.json").exists() and not force:
        raise SystemExit(f"{PUB / 'manifest.json'} exists and is frozen; refusing to resample (use `render` to regenerate the interface).")
    man = load_manifest(C.SPLITS_DIR / f"{C.SPLIT['name']}.json")
    tv = set(man["splits"]["train"]) | set(man["splits"]["val"])
    docs = {k: v for k, v in load_c_only(C.annotation_root()).items() if k in tv}
    rng = random.Random(SEED)
    s1, s2, s3 = _candidates(docs)
    s3_pref = [x for x in s3 if x["both_single"]]
    s4 = _ambiguous(tv)
    picks = {
        1: _pick(s1, N_PER_STRATUM, lambda x: x["c_labels"][0], ROLE_KEY_CAPS[1], rng),
        2: _pick(s2, N_PER_STRATUM, lambda x: "+".join(x["c_labels"]), ROLE_KEY_CAPS[2], rng),
        3: _pick(s3_pref if len(s3_pref) >= N_PER_STRATUM else s3, N_PER_STRATUM, lambda x: "+".join(x["c_outer_labels"]) + ">" + "+".join(x["c_inner_labels"]), ROLE_KEY_CAPS[3], rng),
        4: _pick(s4, N_PER_STRATUM, lambda x: "+".join(x["c_max_roles"]), ROLE_KEY_CAPS[4], rng),
    }
    items, key = [], []
    for s in (1, 2, 3, 4):
        # presentation order is shuffled across strata later; ids encode stratum only after shuffling is applied to the display order
        for k, x in enumerate(picks[s]):
            iid = f"E{s}-{k + 1:02d}"
            text = docs[x["case_id"]].text
            it = {"item_id": iid, "stratum": s, "stratum_name": STRATA[s], "case_id": x["case_id"]}
            kk = {"item_id": iid, "stratum": s, "case_id": x["case_id"]}
            if s in (1, 2):
                pre, span, post = _window(text, x["begin"], x["end"])
                it.update({"begin": x["begin"], "end": x["end"], "pre": pre, "span": span, "post": post})
                kk.update({"begin": x["begin"], "end": x["end"], "c_labels": x["c_labels"]})
            elif s == 3:
                ob, oe = x["outer"]; ib, ie = x["inner"]
                pre, _, post = _window(text, ob, oe, 200)
                it.update({"outer": [ob, oe], "inner": [ib, ie], "pre": pre, "post": post, "outer_text": text[ob:oe], "inner_offset_in_outer": [ib - ob, ie - ob]})
                kk.update({"outer": [ob, oe], "inner": [ib, ie], "c_outer_labels": x["c_outer_labels"], "c_inner_labels": x["c_inner_labels"]})
            else:
                pre, span, post = _window(text, x["begin"], x["end"])
                it.update({"begin": x["begin"], "end": x["end"], "sent_idx": x["sent_idx"], "pre": pre, "span": span, "post": post})
                kk.update({"begin": x["begin"], "end": x["end"], "c_max_roles": x["c_max_roles"], "c_roles_intersecting": x["c_roles_intersecting"], "ambiguity_kind": x["ambiguity_kind"]})
            items.append(it); key.append(kk)
    # blind presentation order: shuffled (seeded) so strata are interleaved
    order = list(range(len(items))); rng.shuffle(order)
    for pos, idx in enumerate(order):
        items[idx]["display_order"] = pos
    items.sort(key=lambda it: it["display_order"])
    PRIV.mkdir(parents=True, exist_ok=True); PUB.mkdir(parents=True, exist_ok=True)
    items_json = json.dumps(items, ensure_ascii=False, indent=1); key_json = json.dumps(key, ensure_ascii=False, indent=1)
    (PRIV / "items.json").write_text(items_json, encoding="utf-8")
    (PRIV / "key.json").write_text(key_json, encoding="utf-8")
    ext = build_context_ext(items, docs)
    (PRIV / "items_context_ext.json").write_text(json.dumps(ext, ensure_ascii=False, indent=1), encoding="utf-8")
    (PRIV / f"expert_audit_{VERSION}.html").write_text(render_html(items, ext, role_definitions()), encoding="utf-8")
    (PRIV / "responses_template.json").write_text(json.dumps({it["item_id"]: {} for it in items}, indent=1))
    manifest = {
        "version": VERSION, "seed": SEED, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "C adjudicated gold, TRAIN+VAL of split_v1 only; no TEST; no model predictions used",
        "n_per_stratum": N_PER_STRATUM, "caps": {"per_judgment": JUDGMENT_CAP, "role_key": ROLE_KEY_CAPS}, "strata": STRATA,
        "candidate_pool": {"1": len(s1), "2": len(s2), "3": len(s3), "3_both_single": len(s3_pref), "4": len(s4)},
        "items": [{k: v for k, v in it.items() if k in ("item_id", "stratum", "case_id", "begin", "end", "outer", "inner", "sent_idx", "display_order")} for it in items],
        "judgments_per_stratum": {str(s): len({it["case_id"] for it in items if it["stratum"] == s}) for s in STRATA},
        "items_sha256": hashlib.sha256(items_json.encode("utf-8")).hexdigest(),
        "key_sha256": hashlib.sha256(key_json.encode("utf-8")).hexdigest(),
        "blind": "interface shows Arabic context and highlighted span(s) only; no A/B/C labels, provenance, or predictions",
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "created_utc"}, sort_keys=True).encode()).hexdigest()
    (PUB / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (PUB / "manifest.sha256").write_text(manifest["manifest_sha256"] + "  manifest.json\n")
    return manifest


def build_context_ext(items: List[dict], docs: Dict[str, "Document"]) -> Dict[str, dict]:
    """Extended (+-EXT_CTX chars) context per frozen item, derived from offsets only; sample membership unchanged."""
    ext = {}
    for it in items:
        text = docs[it["case_id"]].text
        if it["stratum"] == 3:
            ob, oe = it["outer"]
            pre, _, post = _window(text, ob, oe, EXT_CTX)
        else:
            pre, _, post = _window(text, it["begin"], it["end"], EXT_CTX)
        ext[it["item_id"]] = {"pre_ext": pre, "post_ext": post}
    return ext


def render_from_frozen() -> dict:
    """Regenerate the blind interface from the FROZEN items (no resampling); verifies the manifest hashes first."""
    man = json.loads((PUB / "manifest.json").read_text())
    items_json = (PRIV / "items.json").read_text(encoding="utf-8")
    if hashlib.sha256(items_json.encode("utf-8")).hexdigest() != man["items_sha256"]:
        raise SystemExit("items.json does not match the frozen manifest hash")
    items = json.loads(items_json)
    docs = {k: v for k, v in load_c_only(C.annotation_root()).items() if k in {it["case_id"] for it in items}}
    ext = build_context_ext(items, docs)
    (PRIV / "items_context_ext.json").write_text(json.dumps(ext, ensure_ascii=False, indent=1), encoding="utf-8")
    defs = role_definitions()
    html_text = render_html(items, ext, defs)
    (PRIV / f"expert_audit_{VERSION}.html").write_text(html_text, encoding="utf-8")
    return {"items": len(items), "manifest_sha256": man["manifest_sha256"], "html_sha256": hashlib.sha256(html_text.encode("utf-8")).hexdigest(),
            "definitions_status": defs.get("_status")}


# ----------------------------------------------------------------------------- blind interface
def _esc(t: str) -> str:
    return html.escape(t).replace("\r\n", "<br>").replace("\n", "<br>")


def _elide_outer(outer: str, ib: int, ie: int, keep: int = 350, around: int = 250) -> str:
    """Outer span with the inner span marked; long stretches far from both boundaries and the inner span are elided."""
    n = len(outer)
    segs = [(0, min(n, keep)), (max(0, ib - around), min(n, ie + around)), (max(0, n - keep), n)]
    segs.sort(); merged = []
    for a, b in segs:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    out, pos = [], 0
    for a, b in merged:
        if a > pos:
            out.append('<span class="elide"> […] </span>')
        for x, y in ((a, min(b, ib)), (max(a, ib), min(b, ie)), (max(a, ie), b)):
            if y > x:
                piece = _esc(outer[x:y])
                out.append(f'<mark class="inner">{piece}</mark>' if (x >= ib and y <= ie) else piece)
        pos = b
    return "".join(out)


def _role_boxes(name: str, defs: dict, with_uncertain: bool = True) -> str:
    rows = "".join(f'<label><input type="checkbox" name="{name}" value="{r}"> <b>{r}</b> <span class="gloss">{defs[r]["ar"]}</span></label>' for r in C.ROLES)
    if with_uncertain:
        rows += f'<label class="unc"><input type="checkbox" name="{name}" value="UNCERTAIN"> <b>UNCERTAIN</b> <span class="gloss">غير متأكد</span></label>'
    return f'<div class="roles">{rows}</div>'


def _ynu(name: str) -> str:
    return "".join(f'<label><input type="radio" name="{name}" value="{v}"> {v}</label>' for v in ("YES", "NO", "UNCERTAIN"))


def _ctx_blocks(dom: str, compact_html: str, ext_html: str, extra: str = "") -> str:
    """Compact context by default; 'Show more context' reveals the extended window (same highlighting, still blind).

    ``dom`` is a positional, stratum-free DOM id (display only); internal item ids appear solely in form input names.
    """
    return (f'<div class="txt" dir="rtl" id="{dom}-compact">{compact_html}</div>'
            f'<div class="txt ext" dir="rtl" id="{dom}-ext" style="display:none">{ext_html}</div>'
            f'<p class="ctl"><button type="button" onclick="toggle(\'{dom}\')" id="{dom}-btn">Show more context / عرض سياق أوسع</button>{extra}</p>')


def render_html(items: List[dict], ext: Dict[str, dict], defs: dict) -> str:
    parts = []
    for it in items:
        iid = it["item_id"]; s = it["stratum"]; e = ext[iid]
        dom = f"item{it['display_order'] + 1}"   # visible/DOM identity is positional only; the stratum-coded id stays internal
        if s in (1, 2, 4):
            compact = f'{_esc(it["pre"])}<mark class="span">{_esc(it["span"])}</mark>{_esc(it["post"])}'
            wide = f'{_esc(e["pre_ext"])}<mark class="span">{_esc(it["span"])}</mark>{_esc(e["post_ext"])}'
            blocks = _ctx_blocks(dom, compact, wide)
        if s in (1, 2):
            body = (blocks + f'<p class="q">Which rhetorical role or roles genuinely apply to the highlighted passage? / ما الوظيفة أو الوظائف البلاغية التي تنطبق فعلاً على المقطع المظلّل؟</p>'
                    + _role_boxes(f"{iid}|roles", defs))
        elif s == 3:
            ib, ie = it["inner_offset_in_outer"]
            elided = f'<span class="outer">{_elide_outer(it["outer_text"], ib, ie)}</span>'
            compact = f'{_esc(it["pre"])}{elided}{_esc(it["post"])}'
            wide = f'{_esc(e["pre_ext"])}{elided}{_esc(e["post_ext"])}'
            full_outer = (f'<mark class="inner">' .join([_esc(it["outer_text"][:ib]), ""]) if False else
                          f'{_esc(it["outer_text"][:ib])}<mark class="inner">{_esc(it["outer_text"][ib:ie])}</mark>{_esc(it["outer_text"][ie:])}')
            extra = (f' <button type="button" onclick="toggleFull(\'{dom}\')" id="{dom}-fbtn">Show full outer span ({len(it["outer_text"]):,} chars) / عرض المقطع الخارجي كاملاً</button>')
            body = (_ctx_blocks(dom, compact, wide, extra)
                    + f'<div class="txt full" dir="rtl" id="{dom}-full" style="display:none"><span class="outer">{full_outer}</span></div>'
                    f'<p class="legend"><span class="outer">&nbsp;OUTER span (green border)&nbsp;</span> &nbsp; <mark class="inner">INNER span (orange highlight)</mark> &nbsp; <span class="elide">[…] = omitted part of the outer span (use the full-span button)</span></p>'
                    f'<p class="q">Q1. Are both highlighted spans valid <b>distinct</b> rhetorical units in this context? / هل المقطعان المظلّلان وحدتان بلاغيتان صحيحتان و<b>متمايزتان</b> في هذا السياق؟ {_ynu(f"{iid}|valid_units")}</p>'
                    f'<p class="q">Q2. Which rhetorical role or roles apply to the <b>OUTER</b> span? / ما الوظيفة أو الوظائف التي تنطبق على المقطع <b>الخارجي</b>؟</p>{_role_boxes(f"{iid}|outer_roles", defs)}'
                    f'<p class="q">Q3. Which rhetorical role or roles apply to the <b>INNER</b> span? / ما الوظيفة أو الوظائف التي تنطبق على المقطع <b>الداخلي</b>؟</p>{_role_boxes(f"{iid}|inner_roles", defs)}')
        else:
            single = "".join(f'<option value="{r}">{r} — {defs[r]["ar"]}</option>' for r in C.ROLES)
            body = (blocks + f'<p class="q">Can this sentence be represented faithfully by <b>exactly one</b> rhetorical role? / هل يمكن تمثيل هذه الجملة بأمانة بوظيفة بلاغية <b>واحدة فقط</b>؟ {_ynu(f"{iid}|single_role_faithful")}</p>'
                    f'<p class="q">If YES: which ONE role is sufficient? / إذا كانت الإجابة نعم: ما الوظيفة الواحدة الكافية؟ <select name="{iid}|single_role"><option value="">—</option>{single}</select></p>'
                    f'<p class="q">If NO: which rhetorical roles genuinely apply? (select all) / إذا كانت الإجابة لا: ما الوظائف التي تنطبق فعلاً؟ (اختر الجميع)</p>{_role_boxes(f"{iid}|roles", defs)}')
        parts.append(f'<section class="item" id="{dom}"><h3>Item {it["display_order"] + 1}</h3>{body}</section>')
    defs_rows = "".join(f'<tr><td><b>{r}</b></td><td>{html.escape(defs[r]["en"])}</td><td dir="rtl">{html.escape(defs[r]["ar"])}</td></tr>' for r in C.ROLES)
    defs_status = html.escape(defs.get("_status", ""))
    gloss_note = (f'<details id="defs" open><summary><b>Rhetorical role definitions / تعريفات الوظائف البلاغية</b> (click to collapse)</summary>'
                  f'<table class="defs">{defs_rows}</table><p class="small">{defs_status}</p></details>')
    return f'''<!doctype html><html lang="ar"><head><meta charset="utf-8"><title>Blind expert validation — {VERSION}</title>
<style>
body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.6}}
.txt{{font-size:19px;line-height:2;background:#fafafa;border:1px solid #ddd;padding:14px;border-radius:6px;text-align:right;white-space:normal}}
mark.span{{background:#ffe680;padding:2px 0}} mark.inner{{background:#ffd27f;padding:2px 0}} .outer{{border:2px solid #4a7;padding:2px;border-radius:3px}}
.elide{{color:#888;font-style:italic}} .roles{{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 16px;margin:6px 0 12px}}
.ext{{border-color:#9ab}} .full{{max-height:420px;overflow:auto;border-color:#4a7}} .ctl button{{font-size:0.9em}} .defs{{border-collapse:collapse;width:100%}} .defs td{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}} .small{{color:#a50;font-size:0.9em}}
.roles label{{display:block}} .gloss{{color:#666;font-size:0.9em}} .unc{{grid-column:1/-1;border-top:1px dashed #bbb;padding-top:4px}}
.item{{border-bottom:2px solid #ccc;padding:16px 0 24px}} .q{{margin:10px 0 4px}} .legend{{font-size:0.9em;color:#444}}
#export{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 0;z-index:2}} textarea{{width:100%;height:120px}}
</style></head><body>
<h1>Blind independent expert validation of annotation structure ({VERSION})</h1>
<p>80 items in fixed random order. For each item read the Arabic passage (use <i>Show more context</i> whenever the rhetorical function depends on the surroundings), then answer.
No labels or other annotations are shown anywhere; please do not consult other files. UNCERTAIN is always acceptable.
Answers are saved in this browser as you go; when finished click <b>Export responses</b> and send the downloaded JSON file.</p>
{gloss_note}
<div id="export"><button onclick="exportJSON()">Export responses (JSON)</button> <button onclick="showJSON()">Show JSON</button> <span id="progress"></span><textarea id="out" style="display:none"></textarea></div>
<form id="f">{"".join(parts)}</form>
<script>
const KEY="expert_audit_{VERSION}";
function collect(){{const o={{}};const f=document.getElementById("f");
 for(const el of f.elements){{if(!el.name)continue;const [iid,q]=el.name.split("|");o[iid]=o[iid]||{{}};
  if(el.type==="checkbox"){{o[iid][q]=o[iid][q]||[];if(el.checked)o[iid][q].push(el.value);}}
  else if(el.type==="radio"){{if(el.checked)o[iid][q]=el.value;else if(!(q in o[iid]))o[iid][q]=null;}}
  else{{o[iid][q]=el.value||null;}}}}
 return o;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(collect()));}}catch(e){{}} progress();}}
function restore(){{try{{const o=JSON.parse(localStorage.getItem(KEY)||"{{}}");const f=document.getElementById("f");
 for(const el of f.elements){{if(!el.name)continue;const [iid,q]=el.name.split("|");const v=(o[iid]||{{}})[q];if(v==null)continue;
  if(el.type==="checkbox")el.checked=Array.isArray(v)&&v.includes(el.value);else if(el.type==="radio")el.checked=(v===el.value);else el.value=v;}}}}catch(e){{}} progress();}}
function progress(){{const o=collect();let done=0,n=0;for(const iid in o){{n++;const a=o[iid];
 if((a.roles&&a.roles.length)||a.valid_units||a.single_role_faithful)done++;}}document.getElementById("progress").textContent=" answered: "+done+" / "+n;}}
function toggle(iid){{const c=document.getElementById(iid+"-compact"),e=document.getElementById(iid+"-ext"),b=document.getElementById(iid+"-btn");
 const more=e.style.display==="none";e.style.display=more?"block":"none";c.style.display=more?"none":"block";b.textContent=more?"Show compact context / عرض السياق المختصر":"Show more context / عرض سياق أوسع";}}
function toggleFull(iid){{const f=document.getElementById(iid+"-full"),b=document.getElementById(iid+"-fbtn");const on=f.style.display==="none";f.style.display=on?"block":"none";b.textContent=(on?"Hide full outer span / إخفاء":"Show full outer span / عرض")+" المقطع الخارجي كاملاً";}}
function exportJSON(){{const s=JSON.stringify(collect(),null,1);const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{{type:"application/json"}}));a.download="expert_audit_{VERSION}_responses.json";a.click();}}
function showJSON(){{const t=document.getElementById("out");t.style.display="block";t.value=JSON.stringify(collect(),null,1);}}
document.getElementById("f").addEventListener("change",save);document.getElementById("f").addEventListener("input",save);restore();
</script></body></html>'''


# ----------------------------------------------------------------------------- scoring
def _roles(v) -> Tuple[List[str], bool]:
    v = v or []
    unc = "UNCERTAIN" in v
    return [r for r in v if r in C.ROLES], unc


def _setf1(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(set(a) & set(b))
    return 2 * inter / (len(set(a)) + len(set(b))) if (a or b) else 0.0


def score(responses_path: Path) -> dict:
    man = json.loads((PUB / "manifest.json").read_text())
    key_json = (PRIV / "key.json").read_text(encoding="utf-8")
    items_json = (PRIV / "items.json").read_text(encoding="utf-8")
    if hashlib.sha256(key_json.encode("utf-8")).hexdigest() != man["key_sha256"] or hashlib.sha256(items_json.encode("utf-8")).hexdigest() != man["items_sha256"]:
        raise SystemExit("frozen items/key do not match the manifest hashes; scoring refused")
    key = {k["item_id"]: k for k in json.loads(key_json)}
    raw = Path(responses_path).read_bytes()
    archive = PRIV / "responses" / f"{Path(responses_path).stem}__{hashlib.sha256(raw).hexdigest()[:12]}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(raw)   # original response file preserved unchanged
    resp = json.loads(raw.decode("utf-8"))
    unknown = sorted(set(resp) - set(key))
    out = {"version": VERSION, "manifest_sha256": man["manifest_sha256"], "responses_sha256": hashlib.sha256(raw).hexdigest(),
           "n_items": len(key), "n_answered": 0, "unknown_item_ids": unknown, "strata": {}}

    def summarize(pairs: List[Tuple[List[str], List[str]]]) -> dict:
        if not pairs:
            return {"n_scored": 0}
        f1 = [_setf1(a, b) for a, b in pairs]
        return {"n_scored": len(pairs), "set_f1_vs_C": round(sum(f1) / len(f1), 4), "exact_set_agreement": round(sum(1 for a, b in pairs if set(a) == set(b)) / len(pairs), 4)}

    for s in (1, 2, 3, 4):
        ks = [k for k in key.values() if k["stratum"] == s]
        R = {k["item_id"]: resp.get(k["item_id"], {}) for k in ks}
        st: dict = {"n": len(ks)}
        if s in (1, 2):
            pairs, unc, empty, multi = [], 0, 0, 0
            for k in ks:
                roles, u = _roles(R[k["item_id"]].get("roles"))
                unc += u
                if roles:
                    pairs.append((roles, k["c_labels"])); multi += len(roles) >= 2
                elif not u:
                    empty += 1
            st.update(summarize(pairs)); st["UNCERTAIN"] = unc; st["unanswered"] = empty
            if s == 2:
                st["multi_role_confirmed_n"] = multi; st["multi_role_confirmed"] = f"{multi}/{len(ks)}"
        elif s == 3:
            yes = no = unc_q1 = empty = 0; po, pi = [], []; unc_o = unc_i = 0
            for k in ks:
                a = R[k["item_id"]]; v = a.get("valid_units")
                yes += v == "YES"; no += v == "NO"; unc_q1 += v == "UNCERTAIN"; empty += v is None
                ro, uo = _roles(a.get("outer_roles")); ri, ui = _roles(a.get("inner_roles")); unc_o += uo; unc_i += ui
                if ro: po.append((ro, k["c_outer_labels"]))
                if ri: pi.append((ri, k["c_inner_labels"]))
            st.update({"nested_structure_confirmed_n": yes, "nested_structure_confirmed": f"{yes}/{len(ks)}", "valid_units_NO": no, "UNCERTAIN_valid_units": unc_q1,
                       "unanswered": empty, "outer_span": summarize(po), "inner_span": summarize(pi), "UNCERTAIN_outer_roles": unc_o, "UNCERTAIN_inner_roles": unc_i})
        else:
            yes = no = unc_q = empty = 0; pairs = []; unc_roles = 0
            for k in ks:
                a = R[k["item_id"]]; v = a.get("single_role_faithful")
                yes += v == "YES"; no += v == "NO"; unc_q += v == "UNCERTAIN"; empty += v is None
                if v == "YES" and a.get("single_role") in C.ROLES:
                    pairs.append(([a["single_role"]], k["c_max_roles"]))
                elif v == "NO":
                    roles, u = _roles(a.get("roles")); unc_roles += u
                    if roles:
                        pairs.append((roles, k["c_max_roles"]))
            st.update({"single_label_insufficient_n": no, "single_label_insufficient": f"{no}/{len(ks)}", "single_label_sufficient_YES": yes,
                       "UNCERTAIN": unc_q, "UNCERTAIN_roles": unc_roles, "unanswered": empty, "roles_vs_C_tied_set": summarize(pairs)})
        out["strata"][str(s)] = st
    out["n_answered"] = sum(1 for k in key if resp.get(k))
    out["note"] = "stratified sample; metrics reported per stratum only; no pooled agreement, no prevalence estimate; C is not modified"
    PUB.mkdir(parents=True, exist_ok=True)
    (PUB / "results.json").write_text(json.dumps(out, indent=1))
    (PUB / "results.md").write_text(results_markdown(out))
    return out


def results_markdown(out: dict) -> str:
    S = out["strata"]
    L = [f"# Blind independent expert validation audit — results ({out['version']})", "",
         f"Manifest sha256 `{out['manifest_sha256']}`; responses sha256 `{out['responses_sha256']}`; items answered {out['n_answered']}/{out['n_items']}.", "",
         "Stratified sample (20 per stratum, TRAIN+VAL only). Results are reported per stratum; no pooled agreement statistic and no corpus prevalence estimate.", ""]
    s1, s2, s3, s4 = S["1"], S["2"], S["3"], S["4"]
    L += ["| stratum | primary result | set-F1 vs C | exact-set agreement | UNCERTAIN |", "|---|---|---:|---:|---:|",
          f"| 1 single-role, non-nested (control) | — | {s1.get('set_f1_vs_C', '—')} (n={s1.get('n_scored', 0)}) | {s1.get('exact_set_agreement', '—')} | {s1['UNCERTAIN']}/20 |",
          f"| 2 adjudicated multi-role | expert selected >1 role: **{s2.get('multi_role_confirmed', '—')}** | {s2.get('set_f1_vs_C', '—')} (n={s2.get('n_scored', 0)}) | {s2.get('exact_set_agreement', '—')} | {s2['UNCERTAIN']}/20 |",
          f"| 3 nested structure | both units confirmed: **{s3['nested_structure_confirmed']}** (NO {s3['valid_units_NO']}) | outer {s3['outer_span'].get('set_f1_vs_C', '—')} / inner {s3['inner_span'].get('set_f1_vs_C', '—')} | outer {s3['outer_span'].get('exact_set_agreement', '—')} / inner {s3['inner_span'].get('exact_set_agreement', '—')} | Q1 {s3['UNCERTAIN_valid_units']}/20; roles outer {s3['UNCERTAIN_outer_roles']}, inner {s3['UNCERTAIN_inner_roles']} |",
          f"| 4 ambiguous projection | one role insufficient: **{s4['single_label_insufficient']}**; sufficient (YES) {s4['single_label_sufficient_YES']}/20 | vs C tied set {s4['roles_vs_C_tied_set'].get('set_f1_vs_C', '—')} (n={s4['roles_vs_C_tied_set'].get('n_scored', 0)}) | {s4['roles_vs_C_tied_set'].get('exact_set_agreement', '—')} | {s4['UNCERTAIN']}/20 |", ""]
    L += ["Unanswered items per stratum: " + ", ".join(f"{k}: {v.get('unanswered', 0)}" for k, v in S.items()) + ".", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- pre-audit ratification of two guideline conventions
def record_ratification(responses_path: Path) -> dict:
    """Archive the expert's ratification answers separately from the audit responses; never touches C or any experiment."""
    raw = Path(responses_path).read_bytes()
    resp = json.loads(raw.decode("utf-8"))
    priv = PRIV / "ratification"; priv.mkdir(parents=True, exist_ok=True)
    (priv / f"ratification_v1_responses__{hashlib.sha256(raw).hexdigest()[:12]}.json").write_bytes(raw)
    summary = {"form": "ratification_v1", "responses_sha256": hashlib.sha256(raw).hexdigest(),
               "Q1_appeal_mapping_appellant_DEFENDANT_respondent_PLAINTIFF": (resp.get("Q1") or {}).get("answer"),
               "Q2_orders_outside_merits_under_DECISION_APPEAL": (resp.get("Q2") or {}).get("answer"),
               "qualifications_recorded": {q: bool((resp.get(q) or {}).get("qualification")) for q in ("Q1", "Q2")},
               "note": "qualification text kept in the private copy; recorded for the guideline description and reviewed before any displayed definition changes; C unchanged"}
    (PUB / "ratification_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


RATIFICATION_Q1_EN = ("For the purposes of this annotation scheme, is this appellant/respondent mapping a coherent and "
                      "legally/rhetorically defensible operational convention?")
RATIFICATION_Q1_AR = "لأغراض مخطط الترميز هذا، هل يُعد هذا الربط بين المستأنف والمستأنف ضده والوظيفتين البلاغيتين اتفاقية تشغيلية متسقة ومبررة قانونياً وبلاغياً؟"
RATIFICATION_Q2_EN = ("For the purposes of this annotation scheme, is grouping procedural or prosecutorial orders outside the merits under "
                      "DECISION_APPEAL, including when they occur in a first-instance file, a coherent and defensible operational convention?")
RATIFICATION_Q2_AR = "لأغراض مخطط الترميز هذا، هل يُعد إدراج الأوامر الإجرائية أو أوامر النيابة الواقعة خارج الفصل في الموضوع ضمن DECISION_APPEAL، بما في ذلك عند ورودها في ملف أول درجة، اتفاقية تشغيلية متسقة ومبررة؟"


def render_ratification_form() -> str:
    """Two-question pre-audit ratification of existing project conventions (no judgment text).

    The expert is asked whether each convention is coherent and legally/rhetorically defensible as an
    operational annotation convention, not whether it reflects the project's historical intent.
    """
    def ynq(q):
        return "".join(f'<label><input type="radio" name="{q}|answer" value="{v}"> {t}</label>'
                       for v, t in (("YES", "YES"), ("NO", "NO"), ("NEEDS_QUALIFICATION", "NEEDS QUALIFICATION")))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Pre-audit ratification of two annotation conventions (v1.1)</title>
<style>body{{font-family:-apple-system,"Segoe UI",Arial,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;line-height:1.6}} .q{{border:1px solid #ccc;border-radius:6px;padding:14px;margin:16px 0}} label{{display:inline-block;margin-right:16px}} textarea{{width:100%;height:90px}} .ar{{direction:rtl;text-align:right;color:#333}} .conv{{background:#f6f6f6;padding:8px 12px;border-radius:4px}}</style></head><body>
<h1>Pre-audit ratification of two annotation conventions</h1>
<p>The annotation scheme used for this corpus contains two operational conventions. Please assess each one <b>before</b> opening the 80-item validation audit. No judgment text is involved. Your answers are recorded separately from the audit responses and do not change any annotation.</p>
<form id="f">
<div class="q"><h3>Q1. Appeal submissions</h3>
<p class="conv">Project convention: <b>appellant → ARGUMENT_DEFENDANT</b>; <b>respondent → ARGUMENT_PLAINTIFF</b>.<br>
<span class="ar">اتفاقية المشروع: دفوع <b>المستأنف</b> ← ARGUMENT_DEFENDANT؛ دفوع <b>المستأنف ضده</b> ← ARGUMENT_PLAINTIFF.</span></p>
<p>{html.escape(RATIFICATION_Q1_EN)}</p>
<p class="ar">{html.escape(RATIFICATION_Q1_AR)}</p>
{ynq("Q1")}
<p>Optional qualification / توضيح اختياري</p><textarea name="Q1|qualification"></textarea></div>
<div class="q"><h3>Q2. Procedural / prosecutorial orders outside the merits</h3>
<p class="conv">Project convention: procedural or prosecutorial orders outside the merits are categorised under <b>DECISION_APPEAL</b>, including such orders when they occur in a first-instance file.<br>
<span class="ar">اتفاقية المشروع: تُدرج الأوامر الإجرائية أو أوامر النيابة الواقعة خارج الفصل في الموضوع ضمن <b>DECISION_APPEAL</b>، بما في ذلك عند ورودها في ملف أول درجة.</span></p>
<p>{html.escape(RATIFICATION_Q2_EN)}</p>
<p class="ar">{html.escape(RATIFICATION_Q2_AR)}</p>
{ynq("Q2")}
<p>Optional qualification / توضيح اختياري</p><textarea name="Q2|qualification"></textarea></div>
</form>
<p><button onclick="exportJSON()">Export answers (JSON)</button> <button onclick="showJSON()">Show JSON</button></p><textarea id="out" style="display:none"></textarea>
<script>
const KEY="ratification_v1.1";
function collect(){{const o={{"form":"ratification_v1.1"}};for(const el of document.getElementById("f").elements){{if(!el.name)continue;const [q,k]=el.name.split("|");o[q]=o[q]||{{}};if(el.type==="radio"){{if(el.checked)o[q][k]=el.value;else if(!(k in o[q]))o[q][k]=null;}}else o[q][k]=el.value||null;}}return o;}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(collect()));}}catch(e){{}}}}
function restore(){{try{{const o=JSON.parse(localStorage.getItem(KEY)||"{{}}");for(const el of document.getElementById("f").elements){{if(!el.name)continue;const [q,k]=el.name.split("|");const v=(o[q]||{{}})[k];if(v==null)continue;if(el.type==="radio")el.checked=(v===el.value);else el.value=v;}}}}catch(e){{}}}}
function exportJSON(){{const s=JSON.stringify(collect(),null,1);const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{{type:"application/json"}}));a.download="ratification_v1.1_responses.json";a.click();}}
function showJSON(){{const t=document.getElementById("out");t.style.display="block";t.value=JSON.stringify(collect(),null,1);}}
document.getElementById("f").addEventListener("change",save);document.getElementById("f").addEventListener("input",save);restore();
</script></body></html>'''


def write_ratification_form() -> Path:
    PUB.mkdir(parents=True, exist_ok=True)
    out = PUB / "ratification_form_v1.html"
    out.write_text(render_ratification_form(), encoding="utf-8")
    return out
