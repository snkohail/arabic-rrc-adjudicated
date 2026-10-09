from rrc.sentences import sentence_grid


def _check_offsets(text, grid):
    last = 0
    for b, e in grid:
        assert 0 <= b < e <= len(text)
        assert b >= last            # sorted, non-overlapping
        assert text[b:e] == text[b:e].strip()
        last = e


def test_blank_line_is_paragraph_break_and_single_newline_is_soft():
    text = "محاكم دبي\r\n\r\nحيث أن النيابة\r\nاتهمت المتهم.\r\n\r\nوقررت المحكمة"
    g = sentence_grid(text)
    _check_offsets(text, g)
    assert [text[b:e] for b, e in g] == ["محاكم دبي", "حيث أن النيابة\r\nاتهمت المتهم.", "وقررت المحكمة"]


def test_terminal_punctuation_splits_only_before_whitespace():
    text = "بتاريخ 12/3/2023 قرر. ثم قال 3.5 درهم؟ نعم! انتهى"
    g = sentence_grid(text)
    _check_offsets(text, g)
    assert [text[b:e] for b, e in g] == ["بتاريخ 12/3/2023 قرر.", "ثم قال 3.5 درهم؟", "نعم!", "انتهى"]


def test_empty_and_punctuation_only_fragments_dropped():
    text = "\r\n\r\n - \r\n\r\nنص.\r\n\r\n...\r\n\r\n"
    g = sentence_grid(text)
    assert [text[b:e] for b, e in g] == ["نص."]


def test_deterministic():
    text = "أ. ب\r\n\r\nج؟ د"
    assert sentence_grid(text) == sentence_grid(text)


def test_grid_fingerprint_is_deterministic_and_sensitive(make):
    from rrc.sentences import check_grid_manifest, grid_manifest
    d1 = make("أ. ب\r\n\r\nج؟ د", [(0, 2, "FACTS")], case_id="C0001")
    d2 = make("نص واحد. نص ثان", [(0, 2, "FACTS")], case_id="C0002")
    m = grid_manifest({"C0001": d1, "C0002": d2})
    assert m["n_sentences"] == 6 and m["status"] == "FROZEN"   # d1 -> 4 sentences, d2 -> 2
    assert m == grid_manifest({"C0002": d2, "C0001": d1})  # order-independent
    assert check_grid_manifest(m, {"C0001": d1, "C0002": d2}) == []
    d2b = make("نص واحد نص ثان", [(0, 2, "FACTS")], case_id="C0002")  # terminator removed -> 1 sentence
    assert check_grid_manifest(m, {"C0001": d1, "C0002": d2b})
