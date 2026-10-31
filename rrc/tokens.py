"""Sub-word token counting with a persistent cache (keyed by text hash)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import CACHE_DIR, TOKENIZER_ARABERT
from .io import sha256_text

_TOK_CACHE = {}


def load_tokenizer(name: str = TOKENIZER_ARABERT):
    if name in _TOK_CACHE:
        return _TOK_CACHE[name]
    from transformers import AutoTokenizer  # heavy import kept local

    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # pragma: no cover - network path
        tok = AutoTokenizer.from_pretrained(name)
    _TOK_CACHE[name] = tok
    return tok


def _cache_path(name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return CACHE_DIR / f"token_lengths__{slug}.json"


def token_lengths(texts: Iterable[str], name: str = TOKENIZER_ARABERT, batch: int = 256,
                  cache: bool = True) -> List[int]:
    """Number of sub-word tokens (no special tokens) for each text, cached by sha256."""
    texts = list(texts)
    store: Dict[str, int] = {}
    path = _cache_path(name)
    if cache and path.exists():
        store = json.loads(path.read_text())
    keys = [sha256_text(t) for t in texts]
    missing = sorted({k for k, t in zip(keys, texts) if k not in store})
    if missing:
        tok = load_tokenizer(name)
        by_key = {k: t for k, t in zip(keys, texts)}
        for i in range(0, len(missing), batch):
            chunk = missing[i:i + batch]
            enc = tok([by_key[k] for k in chunk], add_special_tokens=False)
            for k, ids in zip(chunk, enc["input_ids"]):
                store[k] = len(ids)
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(store))
    return [store[k] for k in keys]
