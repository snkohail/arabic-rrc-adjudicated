"""Deterministic, annotation-independent Arabic sentence grid (``grid_v1``).

Rule (text only, no annotation input):

1. A *paragraph break* is a blank line: a line break followed by optional
   horizontal whitespace and another line break.  Single line breaks are treated
   as soft wraps (the source judgments are PDF extractions with hard-wrapped
   lines) and count as ordinary whitespace.
2. Inside a paragraph, a sentence ends after ``.`` ``؟`` ``?`` ``!`` when the
   terminator is followed by whitespace or the paragraph end.  A terminator
   glued to a following character (e.g. ``12/3/2023``, ``3.5``) does not split.
3. Each candidate is trimmed of surrounding whitespace and kept only if it
   contains at least one letter or digit.

Offsets are ``[begin, end)`` into the original text.  The grid over the corpus is
fingerprinted (sha256 over the canonical JSON of all offsets) and recorded in
``splits/grid_v1.json`` together with the construction parameters below.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Tuple

from .config import GRID_VERSION
from .io import Document

PARA_BREAK = re.compile(r"\r?\n[ \t\u00a0\u200f\u200e]*\r?\n")
TERMINATOR = re.compile(r"[.؟?!]+(?=\s|$)")
HAS_CONTENT = re.compile(r"[^\W_]", re.UNICODE)

Grid = List[Tuple[int, int]]


def grid_params() -> dict:
    return {
        "version": GRID_VERSION,
        "paragraph_break_regex": PARA_BREAK.pattern,
        "sentence_terminator_regex": TERMINATOR.pattern,
        "content_regex": HAS_CONTENT.pattern,
        "single_line_break": "soft wrap (whitespace)",
        "trim": "surrounding whitespace",
        "keep_if": "contains at least one letter or digit",
    }


def _trim(text: str, b: int, e: int) -> Tuple[int, int]:
    while b < e and text[b].isspace():
        b += 1
    while e > b and text[e - 1].isspace():
        e -= 1
    return b, e


def _paragraphs(text: str) -> Grid:
    out, pos = [], 0
    for m in PARA_BREAK.finditer(text):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    return out


def sentence_grid(text: str) -> Grid:
    grid: Grid = []
    for pb, pe in _paragraphs(text):
        para = text[pb:pe]
        start = 0
        cuts = [m.end() for m in TERMINATOR.finditer(para)]
        if not cuts or cuts[-1] != len(para):
            cuts.append(len(para))
        for c in cuts:
            b, e = _trim(text, pb + start, pb + c)
            if e > b and HAS_CONTENT.search(text[b:e]):
                grid.append((b, e))
            start = c
    return grid


def corpus_grid(docs: Dict[str, Document]) -> Dict[str, Grid]:
    return {cid: sentence_grid(d.text) for cid, d in sorted(docs.items())}


def grid_fingerprint(grids: Dict[str, Grid]) -> str:
    payload = json.dumps({k: [list(x) for x in v] for k, v in sorted(grids.items())}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def grid_manifest(docs: Dict[str, Document], status: str = "FROZEN") -> dict:
    grids = corpus_grid(docs)
    return {
        "name": GRID_VERSION,
        "status": status,
        "params": grid_params(),
        "n_judgments": len(grids),
        "n_sentences": sum(len(g) for g in grids.values()),
        "sentences_per_judgment": {k: len(v) for k, v in grids.items()},
        "text_sha256": {cid: d.text_sha256 for cid, d in sorted(docs.items())},
        "grid_sha256": grid_fingerprint(grids),
    }


def check_grid_manifest(man: dict, docs: Dict[str, Document]) -> List[str]:
    problems = []
    if man["params"] != grid_params():
        problems.append("grid construction parameters changed")
    if set(man["text_sha256"]) != set(docs):
        problems.append("judgment set differs")
    for cid, sha in man["text_sha256"].items():
        if cid in docs and docs[cid].text_sha256 != sha:
            problems.append(f"text changed for {cid}")
    if not problems and grid_fingerprint(corpus_grid(docs)) != man["grid_sha256"]:
        problems.append("grid sha256 mismatch")
    return problems
