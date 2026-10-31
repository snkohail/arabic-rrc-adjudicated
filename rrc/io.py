"""Loading the three annotator folders into a uniform in-memory representation.

Files: ``<root>/<annotator_dir>/C####.json``.  C and A carry the judgment text;
B is blind (no text) and shares the text via ``text_sha256``.  Only the
``spans`` of ``kind == "rhetorical"`` are used anywhere in this project.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import ANNOTATOR_DIRS


@dataclass(frozen=True)
class Span:
    begin: int
    end: int
    label: str
    kind: str
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def key(self):
        return (self.begin, self.end, self.label)


@dataclass
class Document:
    case_id: str
    text: str
    text_sha256: str
    spans: List[Span]
    meta: dict  # every top-level key except text/spans
    path: Optional[Path] = None

    def rhetorical(self) -> List[Span]:
        return [s for s in self.spans if s.kind == "rhetorical"]


@dataclass
class Corpus:
    C: Dict[str, Document]
    A: Dict[str, Document]
    B: Dict[str, Document]

    @property
    def case_ids(self) -> List[str]:
        return sorted(self.C)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_span(raw: dict) -> Span:
    meta = {k: v for k, v in raw.items() if k not in ("begin", "end", "label", "kind")}
    return Span(int(raw["begin"]), int(raw["end"]), str(raw["label"]), str(raw.get("kind", "rhetorical")), meta)


def load_documents(root: Path, annotator: str, texts: Optional[Dict[str, str]] = None) -> Dict[str, Document]:
    """Load one annotator folder. ``texts`` supplies text for blind files (B)."""
    folder = root / ANNOTATOR_DIRS[annotator]
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    docs: Dict[str, Document] = {}
    for p in sorted(folder.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        case_id = str(raw.get("canonical_case_id") or p.stem)
        text = raw.get("text")
        if text is None:
            if texts is None or case_id not in texts:
                raise ValueError(f"{p}: no text and no external text supplied")
            text = texts[case_id]
        declared = raw.get("text_sha256")
        actual = sha256_text(text)
        if declared and declared != actual:
            raise ValueError(f"{p}: text_sha256 mismatch (declared {declared[:12]}, actual {actual[:12]})")
        spans = [_parse_span(s) for s in raw.get("spans", [])]
        meta = {k: v for k, v in raw.items() if k not in ("text", "spans")}
        docs[case_id] = Document(case_id, text, actual, spans, meta, p)
    return docs


def load_corpus(root: Path) -> Corpus:
    C = load_documents(root, "C")
    texts = {k: d.text for k, d in C.items()}
    A = load_documents(root, "A", texts)
    B = load_documents(root, "B", texts)
    if not (set(A) == set(B) == set(C)):
        raise ValueError("annotator folders do not cover the same case ids")
    return Corpus(C=C, A=A, B=B)


def load_c_only(root: Path, normalize: bool = True) -> Dict[str, Document]:
    """Load C. With ``normalize=True`` (default) apply rrc.normalize (cnorm_v1)."""
    docs = load_documents(root, "C")
    if normalize:
        from .normalize import normalize_corpus
        docs, _ = normalize_corpus(docs)
    return docs
