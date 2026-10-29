"""Regression: paired quality ties must not depend on seed aggregation order."""
from analyse_final50k import paired_gap_difference


def test_exact_integer_ties():
    a = [101, 109, 103, 117, 101, 119, 107, 113, 127, 131]
    b = list(reversed(a))
    assert paired_gap_difference(a, b, 97) == 0.0
    b[0] += 1
    assert paired_gap_difference(a, b, 97) < 0
    assert paired_gap_difference(b, a, 97) > 0


if __name__ == '__main__':
    test_exact_integer_ties()
    print('Exact integer tie regression passed')
