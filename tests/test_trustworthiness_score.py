import pytest
from llamasearch.trustworthiness.score import (
    calculate_trustworthiness_score
    )

# the standard test cases
@pytest.mark.parametrize("p1,p2,expected", [
    # Boundary tests
    (0, 0, 1),       # 0% → 1★
    (20, 20, 1),     # 20% → 1★
    (20.1, 20.1, 2), # 20.1% → 2★
    (100, 100, 5),   # 100% → 5★
    
    # Mixed values
    (15, 25, 1),     # 20% avg → 1★
    (35, 45, 2),     # 40% avg → 2★ 
    (50, 70, 3),     # 60% avg → 3★
    (65, 95, 4),     # 80% avg → 4★
    (90, 100, 5),    # 95% avg → 5★
    
    # Float precision
    (19.9, 20.1, 1), # 20% → 1★
    (39.9, 40.1, 2)  # 40% → 2★
])
def test_trust_score(p1, p2, expected):
    assert calculate_trustworthiness_score(p1, p2) == expected

# Error case tests
def test_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(-5, 50)  # Negative value
        
    with pytest.raises(ValueError):
        calculate_trustworthiness_score(110, 50) # >100%
        
    with pytest.raises(TypeError):
        calculate_trustworthiness_score("50", 50) # String input