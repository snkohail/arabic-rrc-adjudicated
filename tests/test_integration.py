"""Runs only when RRC_ANNOTATION_ROOT (or configs/local.json) points at the private annotation folder."""
import json

import pytest

from rrc import config as C
from rrc.io import load_c_only
from rrc.sentences import check_grid_manifest
from rrc.split import check_manifest_against_docs, load_manifest, split_of

GRID = C.SPLITS_DIR / f"{C.GRID_VERSION}.json"
SPLIT = C.SPLITS_DIR / f"{C.SPLIT['name']}.json"


def _modeling(root):
    return {k: v for k, v in load_c_only(root).items() if k not in C.EXCLUDED_FROM_MODELING}


def test_c_loads_and_labels_are_in_inventory(annotation_root):
    docs = load_c_only(annotation_root)
    assert len(docs) == 200 and len(_modeling(annotation_root)) == 199
    for d in docs.values():
        for s in d.rhetorical():
            assert s.label in C.ROLES and 0 <= s.begin < s.end <= len(d.text)


def test_excluded_duplicate_has_identical_text(annotation_root):
    docs = load_c_only(annotation_root)
    assert docs["C0047"].text_sha256 == docs["C0019"].text_sha256


def test_frozen_grid_matches_data(annotation_root):
    if not GRID.exists():
        pytest.skip("no grid manifest yet")
    g = json.loads(GRID.read_text())
    assert g["status"] == "FROZEN" and check_grid_manifest(g, _modeling(annotation_root)) == []


def test_frozen_split_matches_data(annotation_root):
    if not SPLIT.exists():
        pytest.skip("no split manifest yet")
    man = load_manifest(SPLIT)
    docs = _modeling(annotation_root)
    assert man["status"] == "FROZEN"
    assert check_manifest_against_docs(man, docs) == []
    ids = [c for ids in man["splits"].values() for c in ids]
    assert len(ids) == len(set(ids)) == 199 and "C0047" not in ids
    assert {s: len(v) for s, v in man["splits"].items()} == {"train": 139, "val": 30, "test": 30}
    if GRID.exists():
        assert man["grid_sha256"] == json.loads(GRID.read_text())["grid_sha256"]


def test_s2_files_have_no_label_on_ambiguous_or_unlabeled(annotation_root):
    p = C.DERIVED_DIR / "s2" / "train.jsonl"
    if not p.exists():
        pytest.skip("datasets not built")
    for split in ("train", "val", "test"):
        for line in (C.DERIVED_DIR / "s2" / f"{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            assert (r["label"] is not None) == (r["status"] == "LABELED")
            if r["status"] == "AMBIGUOUS_PROJECTION":
                assert len(r["max_roles"]) >= 2 and r["ambiguity_kind"] in ("co_extensive_multi_role", "nested", "other")


def test_s1_examples_are_unique_regions_in_their_split(annotation_root):
    p = C.DERIVED_DIR / "s1" / "train.jsonl"
    if not p.exists():
        pytest.skip("datasets not built")
    man = load_manifest(SPLIT)
    so = split_of(man)
    seen = set()
    for split in ("train", "val", "test"):
        for line in (C.DERIVED_DIR / "s1" / f"{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            assert so[r["case_id"]] == split and r["id"] not in seen and len(r["labels"]) == len(set(r["labels"])) >= 1
            seen.add(r["id"])
