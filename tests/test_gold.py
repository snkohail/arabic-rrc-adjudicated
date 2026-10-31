from rrc.gold import collapse_regions, crossing_pairs, doc_structure, nesting_pairs


def test_collapse_identical_boundaries_into_multihot(make):
    d = make("a" * 100, [(0, 50, "ANALYSIS"), (0, 50, "LAW_REFERENCE"), (50, 100, "DECISION")])
    regs = collapse_regions(d.spans)
    assert [(r.begin, r.end) for r in regs] == [(0, 50), (50, 100)]
    assert regs[0].labels == ("ANALYSIS", "LAW_REFERENCE") and regs[0].is_multi
    assert regs[1].labels == ("DECISION",) and not regs[1].is_multi


def test_collapse_ignores_non_rhetorical(make):
    d = make("a" * 10, [(0, 5, "FACTS")])
    d.spans.append(type(d.spans[0])(0, 5, "COURT", "entity", {}))
    assert len(collapse_regions(d.spans)) == 1


def test_nesting_strict_containment_only(make):
    d = make("a" * 100, [(0, 100, "FACTS"), (10, 20, "LAW_REFERENCE"), (10, 20, "FACTS"), (30, 60, "ANALYSIS"), (50, 70, "DECISION")])
    st = doc_structure(d)
    assert set(st.nesting) == {(0, 1), (0, 2), (0, 3)}  # regions: (0,100),(10,20),(30,60),(50,70)
    assert st.same_role_nesting == [(0, 1)]           # (0,100) FACTS ⊃ (10,20) {FACTS, LAW_REFERENCE}
    assert crossing_pairs(st.regions) == [(2, 3)]       # (30,60) x (50,70)
    assert st.nested_indices == {0, 1, 2, 3}
    assert st.has_multi and st.has_nesting


def test_identical_boundaries_are_not_nesting(make):
    d = make("a" * 10, [(0, 10, "FACTS"), (0, 10, "ISSUE")])
    assert nesting_pairs(collapse_regions(d.spans)) == []
