import pytest

from mteb.leaderboard._pareto import _pareto_frontier


@pytest.mark.parametrize(
    ("scores", "sizes", "expected"),
    [
        ([], [], []),
        ([0.6], [100], [True]),
        ([0.6, 0.65, 0.63], [100, 200, 300], [True, True, False]),
        ([0.6, 0.65], [100, 100], [False, True]),
        ([0.6, 0.6], [100, 200], [True, False]),
        ([0.6, 0.6], [100, 100], [True, True]),
        ([0.7, 0.6, 0.6], [50, 100, 100], [True, False, False]),
        ([None, 0.6], [100, 200], [None, True]),
        ([0.6, 0.7], [None, 200], [None, True]),
        ([float("nan"), 0.6], [100, 200], [None, True]),
        ([0.6, 0.7], [float("inf"), 200], [None, True]),
        ([0.6, 0.7], [-1, 200], [None, True]),
        ([0.6, 0.7], [0, 200], [True, True]),
        ([0.63, 0.6, 0.65], [300, 100, 200], [False, True, True]),
    ],
)
def test_pareto_frontier(scores, sizes, expected):
    assert _pareto_frontier(scores, sizes) == expected


def test_pareto_frontier_requires_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        _pareto_frontier([0.6], [])
