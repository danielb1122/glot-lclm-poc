from glot_lclm.utils.text import best_em_f1, exact_match, f1_score


def test_exact_match_normalization():
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0


def test_f1_overlap():
    assert f1_score("blue red", "blue green") == 0.5


def test_best_em_f1():
    em, f1 = best_em_f1("Paris", ["London", "Paris"])
    assert em == 1.0
    assert f1 == 1.0

