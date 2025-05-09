import pytest
from src.llamasearch.trustworthiness.score import (
    calculate_trustworthiness_score
    )

# the standard test cases
@pytest.mark.parametrize("p1, p2, w1, w2, expected", [
    # Boundary tests
    (0, 0, 0.5, 0.5, 0),        # 0% → 0 (normalized score)
    (20, 20, 0.5, 0.5, 20),     # 20% → 20 (normalized score)
    (20.1, 20.1, 0.5, 0.5, 20.1),  # 20.1% → 20.1 (normalized score)
    (100, 100, 0.5, 0.5, 100),  # 100% → 100 (normalized score)

    # Mixed values
    (15, 25, 0.5, 0.5, 20),     # Weighted average → 20
    (35, 45, 0.5, 0.5, 40),     # Weighted average → 40
    (50, 70, 0.5, 0.5, 60),     # Weighted average → 60
    (65, 95, 0.5, 0.5, 80),     # Weighted average → 80
    (90, 100, 0.5, 0.5, 95),    # Weighted average → 95

    # Float precision
    (19.9, 20.1, 0.5, 0.5, 20),  # Weighted average → 20
    (39.9, 40.1, 0.5, 0.5, 40),  # Weighted average → 40

    # Uneven weights
    (50, 100, 0.3, 0.7, 85),    # Weighted average → 85
    (10, 90, 0.8, 0.2, 26),     # Weighted average → 26
])
def test_trust_score(p1, p2, w1, w2, expected):
    assert calculate_trustworthiness_score(p1, p2, w1, w2) == pytest.approx(expected, rel=1e-2)

# Error case tests
def test_invalid_inputs():
    # Negative percentage
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(-5, 50, 0.5, 0.5)

    # Percentage greater than 100
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(110, 50, 0.5, 0.5)

    # Invalid weight (negative)
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(50, 50, -0.5, 0.5)

    # Invalid weight (greater than 1)
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(50, 50, 1.5, 0.5)

    # String input for percentages
    with pytest.raises(TypeError):
        calculate_trustworthiness_score("50", 50, 0.5, 0.5)

    # String input for weights
    with pytest.raises(TypeError):
        calculate_trustworthiness_score(50, 50, "0.5", 0.5)