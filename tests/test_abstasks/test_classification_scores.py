from __future__ import annotations

import pytest

from mteb.mocks.mock_tasks import MockClassificationTask


@pytest.mark.parametrize(
    ("negative", "positive"),
    [
        (0, 1),
        (0, 2),
        (1, 2),
        (-1, 1),
        ("negative", "positive"),
    ],
)
def test_binary_ap_does_not_depend_on_the_label_values(negative, positive):
    """The larger of the two labels is the positive class, whatever its value is."""
    y_test = [
        negative,
        positive,
        positive,
        negative,
        positive,
        negative,
        negative,
        positive,
    ]
    y_pred = [
        negative,
        positive,
        negative,
        negative,
        positive,
        positive,
        negative,
        positive,
    ]

    scores = MockClassificationTask()._calculate_scores(y_test, y_pred)

    # 4 samples are predicted positive, 3 of them are right, and 3 of the 4 positives are found
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["ap"] == pytest.approx(0.75 * 0.75 + 0.25 * 0.5)
    assert scores["ap_weighted"] == pytest.approx(0.75 * 0.75 + 0.25 * 0.5)


def test_binary_ap_of_perfect_predictions_with_labels_one_and_two():
    scores = MockClassificationTask()._calculate_scores([1, 2, 2, 1], [1, 2, 2, 1])

    assert scores["accuracy"] == 1.0
    assert scores["ap"] == 1.0
