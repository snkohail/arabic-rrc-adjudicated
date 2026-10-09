import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rrc.io import Document, Span, sha256_text  # noqa: E402


def make_doc(text, rows, case_id="T0001", meta=None):
    spans = [Span(b, e, l, "rhetorical", m or {}) for (b, e, l, *rest) in rows for m in [rest[0] if rest else {}]]
    return Document(case_id, text, sha256_text(text), spans, meta or {})


@pytest.fixture
def make():
    return make_doc


@pytest.fixture
def annotation_root():
    """Private annotation root from $RRC_ANNOTATION_ROOT or configs/local.json; skip if absent."""
    try:
        from rrc.config import annotation_root as _resolve
        return _resolve()
    except SystemExit:
        pytest.skip("annotation root not configured; integration test skipped")
