from rrc.expert_audit import _elide_outer, _pick, _setf1, _window


def test_window_snaps_to_whitespace():
    text = "aaa bbb ccc ddd eee fff"
    pre, span, post = _window(text, 8, 11, ctx=5)
    assert span == "ccc" and pre.endswith("bbb ") and post.startswith(" ddd")
    assert not pre or pre[0] != "a" or pre.startswith("aaa")  # snapped to a word boundary


def test_pick_respects_caps_then_relaxes():
    import random
    cands = [{"case_id": f"d{i % 3}", "c_labels": ["FACTS" if i % 2 else "ISSUE"]} for i in range(30)]
    got = _pick(cands, 6, lambda x: x["c_labels"][0], 3, random.Random(13))
    assert len(got) == 6
    from collections import Counter
    assert max(Counter(x["case_id"] for x in got).values()) <= 2 and max(Counter(x["c_labels"][0] for x in got).values()) <= 3
    got2 = _pick(cands[:4], 6, lambda x: x["c_labels"][0], 1, random.Random(13))   # impossible under caps -> relaxed
    assert len(got2) == 4


def test_elide_marks_inner_and_keeps_boundaries():
    outer = "A" * 1000 + "INNER" + "B" * 1000
    h = _elide_outer(outer, 1000, 1005)
    assert h.startswith("A" * 350) and h.endswith("B" * 350) and '<mark class="inner">INNER</mark>' in h and "[…]" in h


def test_set_f1():
    assert _setf1(["A"], ["A"]) == 1.0 and _setf1(["A"], ["B"]) == 0.0 and abs(_setf1(["A", "B"], ["A"]) - 2 / 3) < 1e-9


def test_roles_parsing_and_results_markdown():
    from rrc.expert_audit import _roles, results_markdown
    roles, unc = _roles(["FACTS", "UNCERTAIN", "BOGUS"])
    assert roles == ["FACTS"] and unc
    out = {"version": "v1", "manifest_sha256": "x", "responses_sha256": "y", "n_answered": 80, "n_items": 80, "strata": {
        "1": {"n": 20, "n_scored": 19, "set_f1_vs_C": 0.9, "exact_set_agreement": 0.85, "UNCERTAIN": 1, "unanswered": 0},
        "2": {"n": 20, "n_scored": 18, "set_f1_vs_C": 0.8, "exact_set_agreement": 0.6, "multi_role_confirmed": "14/20", "UNCERTAIN": 2, "unanswered": 0},
        "3": {"n": 20, "nested_structure_confirmed": "16/20", "valid_units_NO": 3, "UNCERTAIN_valid_units": 1, "outer_span": {"n_scored": 19, "set_f1_vs_C": 0.7, "exact_set_agreement": 0.6},
              "inner_span": {"n_scored": 19, "set_f1_vs_C": 0.75, "exact_set_agreement": 0.7}, "UNCERTAIN_outer_roles": 1, "UNCERTAIN_inner_roles": 0, "unanswered": 0},
        "4": {"n": 20, "single_label_insufficient": "12/20", "single_label_sufficient_YES": 6, "UNCERTAIN": 2, "roles_vs_C_tied_set": {"n_scored": 18, "set_f1_vs_C": 0.66, "exact_set_agreement": 0.4}, "unanswered": 0}}}
    md = results_markdown(out)
    assert "14/20" in md and "16/20" in md and "12/20" in md and "no pooled agreement" in md


def test_ratification_form_versioning():
    from rrc.expert_audit import render_ratification_form, RATIFICATION_Q1_EN, RATIFICATION_Q2_EN
    import html, re
    f = render_ratification_form()
    assert 'const KEY="ratification_v1.1";' in f
    assert 'a.download="ratification_v1.1_responses.json"' in f
    assert '{"form":"ratification_v1.1"}' in f
    assert "ratification_v1\"" not in f and "ratification_v1_responses" not in f     # no stale v1 identifiers
    assert html.escape(RATIFICATION_Q1_EN) in f and html.escape(RATIFICATION_Q2_EN) in f
    assert f.count('value="NEEDS_QUALIFICATION"') == 2 and not re.search(r"[؀-ۿ][^<]{200,}", f)
