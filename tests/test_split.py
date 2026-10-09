import random

from rrc.config import ROLES
from rrc.split import (VOLUME_BINS, build_manifest, check_manifest_against_docs, load_manifest,
                       manifest_hash, stratified_group_split, volume_bin, volume_boundaries, write_manifest)


def _fake_indicators(n, seed=0):
    rng = random.Random(seed)
    ind = {}
    for i in range(n):
        cid = f"C{i:04d}"
        d = {r: int(rng.random() < 0.8) for r in ROLES}
        d["DECISION_APPEAL"] = int(rng.random() < 0.15)
        d["MULTI_ROLE_DOC"] = int(rng.random() < 0.5)
        vb = rng.choice(VOLUME_BINS)
        for b in VOLUME_BINS:
            d[b] = int(b == vb)
        ind[cid] = d
    return ind


def test_volume_bins_are_terciles():
    counts = {f"C{i:04d}": v for i, v in enumerate([5, 1, 9, 3, 7, 2, 8, 4, 6])}
    b1, b2 = volume_boundaries(counts)
    assert (b1, b2) == (4, 7)
    assert [volume_bin(v, (b1, b2)) for v in (1, 3, 4, 6, 7, 9)] == ["VOLUME_LOW", "VOLUME_LOW", "VOLUME_MED", "VOLUME_MED", "VOLUME_HIGH", "VOLUME_HIGH"]


def test_split_sizes_groups_and_determinism():
    ind = _fake_indicators(199)
    groups = [[c] for c in ind if c not in ("C0001", "C0002", "C0003", "C0010")] + [["C0001", "C0002"], ["C0003", "C0010"]]
    sizes = {"train": 139, "val": 30, "test": 30}
    a1 = stratified_group_split(ind, groups, sizes, 13)
    assert a1 == stratified_group_split(ind, groups, sizes, 13)
    assert {s: sum(v == s for v in a1.values()) for s in sizes} == sizes
    assert a1["C0001"] == a1["C0002"] and a1["C0003"] == a1["C0010"]
    assert stratified_group_split(ind, groups, sizes, 37) != a1


def test_split_balances_rare_indicator_and_volume():
    ind = _fake_indicators(199, seed=1)
    sizes = {"train": 139, "val": 30, "test": 30}
    a = stratified_group_split(ind, [[c] for c in ind], sizes, 13)
    for k in ("DECISION_APPEAL", "VOLUME_HIGH"):
        total = sum(d[k] for d in ind.values())
        for s, n in sizes.items():
            got = sum(ind[c][k] for c, v in a.items() if v == s)
            assert abs(got - total * n / 199) <= 2


def test_manifest_hash_roundtrip_and_exclusion(tmp_path, make, monkeypatch):
    from rrc import split as S
    monkeypatch.setattr(S, "EXCLUDED_FROM_MODELING", {"C0099": "dup"})
    docs = {}
    for i in range(10):
        cid = f"C{i:04d}"
        docs[cid] = make("x" * 100 + str(i), [(0, 50, "FACTS"), (50, 100, "ANALYSIS")] + [(j, j + 1, "ISSUE") for j in range(i)], case_id=cid)
    dedup = {"params": {}, "groups": [[c] for c in docs] + [["C0099"]], "multi_member_groups": []}
    man = build_manifest(docs, dedup, params={"name": "t", "seed": 13, "sizes": {"train": 6, "val": 2, "test": 2}, "volume_bins": VOLUME_BINS}, grid_sha256="abc")
    assert manifest_hash(man) == man["manifest_sha256"] and man["status"] == "FROZEN"
    assert man["excluded_from_modeling"] == {"C0099": "dup"} and "C0099" not in {c for v in man["splits"].values() for c in v}
    write_manifest(man, tmp_path / "t.json")
    back = load_manifest(tmp_path / "t.json")
    assert back == man and check_manifest_against_docs(back, docs) == []
    docs["C0000"].text_sha256 = "0" * 64
    assert check_manifest_against_docs(back, docs)
