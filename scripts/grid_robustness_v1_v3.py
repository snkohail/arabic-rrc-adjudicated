"""Projection robustness: grid_v1 (frozen regex grid) vs grid_v3 (frozen Stanza grid).

TRAIN+VAL judgments only; TEST is never loaded.  Writes reports/grid_robustness/grid_v1_v3.{json,md}.
Usage: python scripts/grid_robustness_v1_v3.py [--annotation-root PATH]
"""
import json, sys, collections, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rrc import config as C
from rrc.io import load_c_only
from rrc.sentences import sentence_grid
from rrc.grid_v3 import load_grid_v3, SPLIT_PATH
from rrc.projection import project_document
OUT = C.REPORTS_DIR / "grid_robustness"; OUT.mkdir(parents=True, exist_ok=True)
man = json.loads(SPLIT_PATH.read_text()); root = C.annotation_root(sys.argv[sys.argv.index("--annotation-root") + 1] if "--annotation-root" in sys.argv else None)
docs = load_c_only(root, normalize=True); v3 = load_grid_v3()
grids = {"grid_v1": lambda d: sentence_grid(d.text), "grid_v3": lambda d: v3[d.case_id]}
res, per, lens = {}, {}, {g: [] for g in grids}
for g, gf in grids.items():
    res[g], per[g] = {}, {}
    for split in ("train", "val"):
        tot, kinds = collections.Counter(), collections.Counter()
        for cid in man["splits"][split]:
            d = docs[cid]; G = gf(d); P = project_document(d, G); lens[g] += [e - b for b, e in G]
            amb = [p for p in P if p.status == "AMBIGUOUS_PROJECTION"]; unl = sum(p.status == "UNLABELED" for p in P); lab = sum(p.status == "LABELED" for p in P)
            tot.update(grid=len(P), unlabeled=unl, ambiguous=len(amb), eligible=lab, covered=len(P)-unl)
            for p in amb: kinds[p.ambiguity_kind] += 1
            per[g][cid] = {"split": split, "grid": len(P), "ambiguous": len(amb), "amb_rate": round(100*len(amb)/len(P), 2) if P else 0.0}
        res[g][split] = {**tot, "ambiguous_kind": dict(kinds), "identity": tot["unlabeled"]+tot["ambiguous"]+tot["eligible"] == tot["grid"]}
    tv, kv = collections.Counter(), collections.Counter()
    for split in ("train", "val"):
        tv.update({k: res[g][split][k] for k in ("grid","unlabeled","covered","ambiguous","eligible")}); kv.update(res[g][split]["ambiguous_kind"])
    res[g]["train+val"] = {**tv, "ambiguous_kind": dict(kv), "identity": tv["unlabeled"]+tv["ambiguous"]+tv["eligible"] == tv["grid"]}
    res[g]["median_sentence_chars"] = statistics.median(lens[g])
moved = sorted([{"case_id": c, "split": per["grid_v1"][c]["split"], "v1_rate": per["grid_v1"][c]["amb_rate"], "v3_rate": per["grid_v3"][c]["amb_rate"],
                 "delta": round(per["grid_v3"][c]["amb_rate"]-per["grid_v1"][c]["amb_rate"], 2), "v1_n": per["grid_v1"][c]["grid"], "v3_n": per["grid_v3"][c]["grid"]}
                for c in per["grid_v1"] if abs(per["grid_v3"][c]["amb_rate"]-per["grid_v1"][c]["amb_rate"]) > 5], key=lambda m: -abs(m["delta"]))
json.dump({"results": res, "moved_gt_5_points": moved, "per_judgment": per}, open(OUT/"grid_v1_v3.json", "w"), indent=1)
R1, R3 = res["grid_v1"]["train+val"], res["grid_v3"]["train+val"]
pc = lambda k, R: f"{R[k]:,} ({100*R[k]/R['grid']:.1f}%)"
rows = [("grid sentences", f"{R1['grid']:,}", f"{R3['grid']:,}"), ("median sentence length (chars)", f"{R1['grid'] and res['grid_v1']['median_sentence_chars']:.0f}", f"{res['grid_v3']['median_sentence_chars']:.0f}"),
        ("unlabeled (no C region)", pc("unlabeled", R1), pc("unlabeled", R3)), ("covered", pc("covered", R1), pc("covered", R3)),
        ("AMBIGUOUS_PROJECTION", pc("ambiguous", R1), pc("ambiguous", R3)),
        ("  co-extensive multi-role", f"{R1['ambiguous_kind'].get('co_extensive_multi_role',0):,}", f"{R3['ambiguous_kind'].get('co_extensive_multi_role',0):,}"),
        ("  nested", f"{R1['ambiguous_kind'].get('nested',0):,}", f"{R3['ambiguous_kind'].get('nested',0):,}"), ("  other", f"{R1['ambiguous_kind'].get('other',0):,}", f"{R3['ambiguous_kind'].get('other',0):,}"),
        ("ambiguous as share of covered", f"{100*R1['ambiguous']/R1['covered']:.1f}%", f"{100*R3['ambiguous']/R3['covered']:.1f}%"),
        ("eligible single-label", pc("eligible", R1), pc("eligible", R3)),
        ("identity holds (train, val, pooled)", str(all(res['grid_v1'][s]['identity'] for s in ('train','val','train+val'))), str(all(res['grid_v3'][s]['identity'] for s in ('train','val','train+val')))),
        ("judgments with |Δ ambiguity rate| > 5 pts vs grid_v1", "—", f"{len(moved)} / {len(per['grid_v1'])}")]
md = ["# Projection robustness: grid_v1 vs grid_v3 (train+val, 169 judgments)", "", "| | grid_v1 | grid_v3 (Stanza) |", "|---|---:|---:|"] + [f"| {a} | {b} | {c} |" for a,b,c in rows] + ["", "## Per split", ""]
for s in ("train","val"):
    for g in grids:
        r = res[g][s]; md.append(f"- **{s} / {g}**: grid {r['grid']:,}; unlabeled {r['unlabeled']:,}; covered {r['covered']:,}; ambiguous {r['ambiguous']:,} {r['ambiguous_kind']}; eligible {r['eligible']:,}; identity {r['identity']}")
md += ["", f"## Moved > 5 points ({len(moved)}): direction v3 lower {sum(m['delta']<0 for m in moved)}, higher {sum(m['delta']>0 for m in moved)}; of which v3 grid < 20 sentences: {sum(m['v3_n']<20 for m in moved)}", "", "| case | split | v1 | v3 | Δ | v1 n | v3 n |", "|---|---|---:|---:|---:|---:|---:|"] + [f"| {m['case_id']} | {m['split']} | {m['v1_rate']} | {m['v3_rate']} | {m['delta']:+} | {m['v1_n']} | {m['v3_n']} |" for m in moved]
(OUT/"grid_v1_v3.md").write_text("\n".join(md)+"\n"); print("\n".join(md[:26]))
