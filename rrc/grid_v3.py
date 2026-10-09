"""grid_v3 — annotation-independent Arabic sentence grid from a released neural segmenter.

TEXT-ONLY INPUT.  This module reads exactly one field, ``text`` (plus ``text_sha256`` for an
integrity check), from the judgment files.  It never reads ``spans`` or any annotation layer,
adjudication output, or projection file, and it does not import ``rrc.io``/``rrc.gold``/
``rrc.projection``.  The segmenter is Stanza's Arabic tokenize processor (PADT model), run with
sentence splitting enabled; a sentence's offsets are the start of its first token and the end of
its last token, in the original text's character space (identical offset semantics to grid_v1).

Unlike grid_v1 (a deterministic regex recomputed from text on load), grid_v3 depends on a neural
model, so the frozen manifest stores the offsets themselves under ``grids``; the ``grid_sha256``
is the same fingerprint procedure as grid_v1 (``rrc.sentences.grid_fingerprint`` over all offsets).
Selection order: (a) a released model from the corpus segmentation work — none public; (b) CAMeL Tools
sentence splitter — none exists in camel_tools 1.5.7; (c) Stanza Arabic — used.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .config import REPO_ROOT

SPLIT_PATH = REPO_ROOT / "splits" / "split_v1.json"
from .sentences import grid_fingerprint  # pure function over offsets; the module is text-only

GRID_V3_PATH = REPO_ROOT / "splits" / "grid_v3.json"
HAS_CONTENT = re.compile(r"[^\W_]", re.UNICODE)
Grid = List[Tuple[int, int]]


def grid_params() -> dict:
    import stanza
    return {"version": "grid_v3", "segmenter": "stanza", "stanza_version": stanza.__version__, "language": "ar",
            "processors": "tokenize", "package": "padt", "sentence_split": True, "use_gpu": False,
            "offsets": "first token start_char .. last token end_char (original text character space)",
            "keep_if": "contains at least one letter or digit", "input": "text only"}


def _texts(annotation_root: Path, split_manifest: dict) -> Dict[str, Tuple[str, str]]:
    """case_id -> (text, text_sha256) for the modelling judgments named in the split manifest.
    Only the ``text`` and ``text_sha256`` keys are read."""
    out = {}
    for cid, expected_sha in sorted(split_manifest["text_sha256"].items()):
        obj = json.loads((annotation_root / "Annotator_C" / f"{cid}.json").read_text(encoding="utf-8"))
        text = obj["text"]
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha != expected_sha or obj.get("text_sha256", sha) != expected_sha:
            raise SystemExit(f"text hash mismatch for {cid}")
        out[cid] = (text, sha)
    return out


def sentence_grid(nlp, text: str) -> Grid:
    doc = nlp(text)
    grid: Grid = []
    for s in doc.sentences:
        b, e = s.tokens[0].start_char, s.tokens[-1].end_char
        if e > b and HAS_CONTENT.search(text[b:e]):
            grid.append((b, e))
    return grid


def build(annotation_root: Path, out_path: Path = GRID_V3_PATH, force: bool = False) -> dict:
    import stanza
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists and is FROZEN; refusing to regenerate.")
    split = json.loads(SPLIT_PATH.read_text())
    texts = _texts(annotation_root, split)
    nlp = stanza.Pipeline("ar", processors="tokenize", tokenize_no_ssplit=False, use_gpu=False, verbose=False, download_method=None)
    grids: Dict[str, Grid] = {}
    for i, (cid, (text, _)) in enumerate(texts.items(), 1):
        grids[cid] = sentence_grid(nlp, text)
        print(f"[{i}/{len(texts)}] {cid}: {len(grids[cid])} sentences", flush=True)
    man = {"name": "grid_v3", "status": "FROZEN", "params": grid_params(), "n_judgments": len(grids),
           "n_sentences": sum(len(g) for g in grids.values()),
           "sentences_per_judgment": {k: len(v) for k, v in grids.items()},
           "text_sha256": {cid: sha for cid, (_, sha) in texts.items()},
           "split_manifest_sha256": split.get("manifest_sha256"),
           "grid_sha256": grid_fingerprint(grids),
           "grids": {k: [list(x) for x in v] for k, v in grids.items()}}
    out_path.write_text(json.dumps(man, ensure_ascii=False) + "\n")
    out_path.with_suffix(".sha256").write_text(man["grid_sha256"] + "  " + out_path.name + "\n")
    print(f"frozen grid_v3: {man['n_sentences']} sentences over {man['n_judgments']} judgments; grid_sha256={man['grid_sha256']}")
    return man


def load_grid_v3(path: Path = GRID_V3_PATH) -> Dict[str, Grid]:
    man = json.loads(path.read_text())
    grids = {k: [tuple(x) for x in v] for k, v in man["grids"].items()}
    if grid_fingerprint(grids) != man["grid_sha256"]:
        raise SystemExit("grid_v3 sha256 mismatch")
    return grids


if __name__ == "__main__":
    from .config import annotation_root
    build(annotation_root(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None), force="--force" in sys.argv)
