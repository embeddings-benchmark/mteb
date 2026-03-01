import numpy as np
import pytest

from mteb.abstasks.multilabel_classification import hamming_score


class TestHammingScore:
    """Test cases for the hamming_score function."""

    def test_perfect_match(self):
        """Test hamming score with perfect predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        score = hamming_score(y_true, y_pred)
        assert score == 1.0

    def test_no_match(self):
        """Test hamming score with completely wrong predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0, 1, 0], [1, 0, 1]])
        score = hamming_score(y_true, y_pred)
        assert score == 0.0

    def test_partial_match(self):
        """Test hamming score with partial predictions."""
        y_true = np.array([[1, 1, 0], [0, 1, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        # Sample 1: intersection=1, union=2 -> 0.5
        # Sample 2: intersection=1, union=2 -> 0.5
        # Average: 0.5
        score = hamming_score(y_true, y_pred)
        assert abs(score - 0.5) < 1e-6

    def test_division_by_zero_handling(self):
        """Test hamming score handles division by zero (all zeros case)."""
        y_true = np.array([[0, 0, 0], [1, 0, 1]])
        y_pred = np.array([[0, 0, 0], [1, 1, 0]])
        # Sample 1: both all zeros -> score = 1.0
        # Sample 2: intersection=1, union=3 -> 1/3
        # Average: (1.0 + 1/3) / 2 = 2/3
        score = hamming_score(y_true, y_pred)
        expected = (1.0 + 1 / 3) / 2
        assert abs(score - expected) < 1e-6

    def test_all_zeros(self):
        """Test hamming score with all zero predictions and labels."""
        y_true = np.array([[0, 0, 0], [0, 0, 0]])
        y_pred = np.array([[0, 0, 0], [0, 0, 0]])
        score = hamming_score(y_true, y_pred)
        assert score == 1.0

    def test_shape_mismatch(self):
        """Test hamming score raises error on shape mismatch."""
        y_true = np.array([[1, 0]])
        y_pred = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            hamming_score(y_true, y_pred)

    def test_empty_arrays(self):
        """Test hamming score raises error on empty arrays."""
        y_true = np.array([]).reshape(0, 3)
        y_pred = np.array([]).reshape(0, 3)
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            hamming_score(y_true, y_pred)

    def test_non_binary_values(self):
        """Test hamming score raises error on non-binary values."""
        y_true = np.array([[1, 2, 0]])
        y_pred = np.array([[1, 0, 0]])
        with pytest.raises(ValueError, match="Arrays must contain only binary values"):
            hamming_score(y_true, y_pred)

    def test_wrong_dimensions(self):
        """Test hamming score raises error on wrong dimensions."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        with pytest.raises(ValueError, match="Arrays must be 2D"):
            hamming_score(y_true, y_pred)

    def test_type_conversion(self):
        """Test hamming score handles type conversion."""
        # Test with lists (should be converted to numpy arrays)
        y_true = [[1, 0, 1], [0, 1, 0]]
        y_pred = [[1, 0, 1], [0, 1, 0]]
        score = hamming_score(y_true, y_pred)
        assert score == 1.0

    def test_invalid_input_types(self):
        """Test hamming score raises error on invalid input types."""
        # Test with None values which cannot be converted to proper arrays
        with pytest.raises((TypeError, ValueError)):
            hamming_score(None, [[1, 0]])

        # Test with incompatible nested structure
        with pytest.raises(ValueError):
            hamming_score([["invalid", "data"]], [[1, 0]])

    def test_mixed_performance_case(self):
        """Test hamming score with mixed performance across samples."""
        y_true = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1]])
        y_pred = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]])
        # Sample 1: intersection=1, union=2 -> 1/2 = 0.5
        # Sample 2: intersection=2, union=2 -> 2/2 = 1.0
        # Sample 3: intersection=0, union=3 -> 0/3 = 0.0
        # Sample 4: intersection=0, union=1 -> 0/1 = 0.0
        # Average: (0.5 + 1.0 + 0.0 + 0.0) / 4 = 0.375
        score = hamming_score(y_true, y_pred)
        expected = 0.375
        assert abs(score - expected) < 1e-6
