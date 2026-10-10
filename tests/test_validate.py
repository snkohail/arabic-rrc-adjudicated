from rrc.validate import find_defer, provenance_audit, validate_document


def test_validator_passes_clean_doc(make):
    d = make("abc def ghi", [(0, 3, "FACTS"), (4, 7, "ANALYSIS")], meta={"text_len": 11})
    r = validate_document(d)
    assert r.passed and not r.has_defer


def test_validator_catches_bad_offsets_labels_dups_defer(make):
    d = make("abc def", [(0, 3, "FACTS"), (0, 3, "FACTS"), (5, 20, "ANALYSIS"), (0, 3, "BOGUS"), (3, 4, "ISSUE")],
             meta={"adjudication": {"status": "DEFER to expert"}})
    r = validate_document(d)
    msgs = " ".join(r.errors)
    assert "duplicate" in msgs and "bad offsets" in msgs and "unknown label" in msgs and "whitespace-only" in msgs
    assert r.has_defer and not r.passed
    assert find_defer(d)


def test_provenance_audit_flags_union_and_unreviewed(make):
    rows = [
        (0, 10, "FACTS", {"provenance": "human_adjudicated"}),
        (0, 10, "ISSUE", {"provenance": "human_adjudicated"}),            # explicit multi-role
        (20, 30, "FACTS", {"provenance": "unadjudicated_A"}),
        (20, 30, "ANALYSIS", {"provenance": "unadjudicated_B"}),          # automatic union
        (40, 50, "FACTS", {"provenance": "llm_adjudicated", "candidate_source": "UNION"}),
        (40, 50, "ISSUE", {"provenance": "llm_adjudicated", "candidate_source": "UNION"}),
    ]
    d = make("x" * 60, rows, meta={"merge": {"NOT_human_adjudicated_gold": True}})
    p = provenance_audit({d.case_id: d}, trust_flags=True)
    assert p["multi_role_regions"] == 3
    assert p["multi_role_regions_all_rows_reviewed"] == 1
    assert p["multi_role_regions_with_unreviewed_rows"] == 1
    assert p["multi_role_regions_with_union_marker"] == 1
    assert p["cases_declaring_NOT_human_adjudicated_gold"] == 1
    assert p["every_multi_role_region_is_explicit_adjudicator_decision"] is False


def test_provenance_audit_confirms_clean_gold(make):
    rows = [(0, 10, "FACTS", {"provenance": "human_adjudicated"}), (0, 10, "ISSUE", {"provenance": "human_adjudicated"})]
    d = make("x" * 20, rows)
    assert provenance_audit({d.case_id: d}, trust_flags=True)["every_multi_role_region_is_explicit_adjudicator_decision"] is True


def test_provenance_audit_ignores_tool_artefact_flags_by_attestation(make):
    rows = [(0, 10, "FACTS", {"provenance": "unadjudicated_A"}), (0, 10, "ISSUE", {"provenance": "unadjudicated_B"})]
    d = make("x" * 20, rows, meta={"merge": {"NOT_human_adjudicated_gold": True}})
    p = provenance_audit({d.case_id: d}, trust_flags=False)
    assert p["confirmation_by_flags_alone"] is False
    assert p["every_multi_role_region_is_explicit_adjudicator_decision"] is True
    assert p["confirmation_basis"].startswith("declared gold status")
