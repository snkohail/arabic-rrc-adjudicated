from rrc.normalize import normalize_document


def test_r1_drops_tiny_and_letterless_rows(make):
    text = "له كلمة طويلة هنا 2004) ونص آخر"
    d = make(text, [(0, 2, "ANALYSIS"), (3, 18, "FACTS"), (19, 24, "ANALYSIS"), (25, 32, "ISSUE")])
    nd, log = normalize_document(d)
    assert [s.key for s in nd.rhetorical()] == [(3, 18, "FACTS"), (25, 32, "ISSUE")]
    assert {(x["begin"], x["rule"]) for x in log} == {(0, "R1_tiny"), (19, "R1_tiny")}


def test_r2_merges_near_identical_same_label_keeping_outer(make):
    text = "x" * 200
    d = make(text, [(10, 100, "FACTS"), (24, 100, "FACTS"), (24, 100, "ISSUE"), (10, 60, "FACTS")])
    nd, log = normalize_document(d)
    keys = sorted(s.key for s in nd.rhetorical())
    assert keys == [(10, 60, "FACTS"), (10, 100, "FACTS"), (24, 100, "ISSUE")]   # inner FACTS dup dropped; other label kept
    assert log == [{"case_id": "T0001", "begin": 24, "end": 100, "label": "FACTS", "rule": "R2_near_identical", "kept_begin": 10, "kept_end": 100}]


def test_inherited_labels_are_kept(make):
    d = make("x" * 200, [(0, 200, "ANALYSIS"), (50, 120, "ANALYSIS"), (50, 120, "LAW_REFERENCE")])
    nd, log = normalize_document(d)
    assert log == [] and len(nd.rhetorical()) == 3


def test_normalization_is_idempotent(make):
    d = make("x" * 200, [(0, 2, "FACTS"), (10, 100, "FACTS"), (24, 100, "FACTS")])
    nd, _ = normalize_document(d)
    nd2, log2 = normalize_document(nd)
    assert log2 == [] and [s.key for s in nd2.rhetorical()] == [s.key for s in nd.rhetorical()]
