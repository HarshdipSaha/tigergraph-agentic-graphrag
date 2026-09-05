from agrag.normalize import match_flags, normalize


def test_normalize_strips_accents_case_and_dashes():
    assert normalize("Naim Süleymanoğlu") == "naim suleymanoglu"
    assert normalize("Fencing at the 2008 Summer Olympics – Men's épée") == "fencing at the 2008 summer olympics - men's epee"
    assert normalize("  5 ") == "5"


def test_match_flags_exact_normalized_contains():
    f = match_flags("Chen Ding", "Chen Ding")
    assert f == {"exact": True, "normalized": True, "contains": True}
    f = match_flags("chen ding", "Chen Ding")
    assert f == {"exact": False, "normalized": True, "contains": True}
    f = match_flags("The winner was Chen Ding.", "Chen Ding")
    assert f == {"exact": False, "normalized": False, "contains": True}
    f = match_flags("Wang Zhen", "Chen Ding")
    assert f == {"exact": False, "normalized": False, "contains": False}


def test_match_flags_handles_none_prediction():
    assert match_flags(None, "5") == {"exact": False, "normalized": False, "contains": False}
