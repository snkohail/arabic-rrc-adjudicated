import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "mock"))

from make_mock_data import write  # noqa: E402

from rrc.gold import doc_structure
from rrc.io import load_c_only, load_corpus
from rrc.projection import project_document
from rrc.sentences import sentence_grid
from rrc.validate import validate_document


def test_mock_folder_has_four_examples_of_each_form(tmp_path):
    items = write(tmp_path / "annotation")
    assert Counter(i["form"] for i in items) == {1: 4, 2: 4, 3: 4, 4: 4}
    corpus = load_corpus(tmp_path / "annotation")
    docs = load_c_only(tmp_path / "annotation")
    assert len(corpus.C) == len(docs) == 4
    kinds = Counter()
    for cid, d in docs.items():
        assert validate_document(d).passed
        S = doc_structure(d)
        assert any(r.is_multi for r in S.regions) and S.nesting and S.crossing
        for p in project_document(d, sentence_grid(d.text)):
            if p.ambiguity_kind:
                kinds[p.ambiguity_kind] += 1
    assert kinds == {"co_extensive_multi_role": 4, "nested": 4, "other": 4}
    spec = json.loads((tmp_path / "mock_items.json").read_text(encoding="utf-8"))
    assert len(spec["items"]) == 16
