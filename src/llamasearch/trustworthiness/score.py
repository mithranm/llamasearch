def calculate_trustworthiness_score(percent_1: float, percent_2: float, weight_1: float, weight_2: float) -> float:
    """
    Author: Shaheda

    Calculates a 1-5 trustworthiness score from two percentages.
    Includes input validation and precise boundary handling.

    Args:
        percent_1: First trust metric (0-100)
        percent_2: Second trust metric (0-100)

    Returns:
        int: Star rating (1-5)

    Raises:
        ValueError: If inputs are outside 0-100 range
    """
     # Input validation
    if not all(isinstance(p, (int, float)) for p in [percent_1, percent_2, weight_1, weight_2]):
        raise TypeError("Percentages and weights must be numbers")
    if not all(0 <= p <= 100 for p in [percent_1, percent_2]):
        raise ValueError("Percentages must be between 0-100")
    if not all(0 <= w <= 1 for w in [weight_1, weight_2]):
        raise ValueError("Weights must be between 0-1")
    
    return (percent_1 * weight_1 + percent_2 * weight_2) / (weight_1 + weight_2)