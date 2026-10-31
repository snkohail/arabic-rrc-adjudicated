from rrc.gold import collapse_regions
from rrc.projection import project_sentence, role_coverage


def regs(make, rows, n=100):
    return collapse_regions(make("x" * n, rows).spans)


def test_coverage_unions_same_role_nested_regions(make):
    r = regs(make, [(0, 50, "FACTS"), (10, 20, "FACTS"), (40, 60, "ANALYSIS")])
    assert role_coverage(r, 0, 60) == {"FACTS": 50, "ANALYSIS": 20}


def test_unique_maximum_is_labeled(make):
    r = regs(make, [(0, 30, "FACTS"), (30, 100, "ANALYSIS")])
    p = project_sentence(r, 20, 60)   # 10 chars FACTS, 30 chars ANALYSIS
    assert p.status == "LABELED" and p.label == "ANALYSIS" and p.roles_intersecting == ["FACTS", "ANALYSIS"]


def test_no_overlap_is_unlabeled(make):
    r = regs(make, [(0, 10, "FACTS")])
    p = project_sentence(r, 50, 60)
    assert p.status == "UNLABELED" and p.label is None and p.roles_intersecting == []


def test_shared_maximum_is_ambiguous_not_tie_broken_nested(make):
    r = regs(make, [(0, 100, "FACTS"), (40, 60, "LAW_REFERENCE")])
    p = project_sentence(r, 45, 55)
    assert p.status == "AMBIGUOUS_PROJECTION" and p.label is None
    assert set(p.max_roles) == {"FACTS", "LAW_REFERENCE"} and p.ambiguity_kind == "nested"


def test_co_extensive_multi_role_region_is_ambiguous(make):
    r = regs(make, [(0, 100, "ANALYSIS"), (0, 100, "LAW_REFERENCE")])
    p = project_sentence(r, 10, 20)
    assert p.status == "AMBIGUOUS_PROJECTION" and p.ambiguity_kind == "co_extensive_multi_role"


def test_coincidental_partial_overlap_kind_other(make):
    r = regs(make, [(0, 15, "FACTS"), (25, 100, "ANALYSIS")])
    p = project_sentence(r, 10, 30)
    assert p.status == "AMBIGUOUS_PROJECTION" and p.ambiguity_kind == "other"
