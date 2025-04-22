def calculate_trustworthiness_score(percent_1: float, percent_2: float) -> int:
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
    if not all(isinstance(p, (int, float)) for p in [percent_1, percent_2]):
        raise TypeError("Percentages must be numbers")
    if not all(0 <= p <= 100 for p in [percent_1, percent_2]):
        raise ValueError("Percentages must be between 0-100")

    avg = (percent_1 + percent_2) / 2
    
    # Precise boundary handling
    if avg <= 20:
        return 1
    if avg <= 40:
        return 2
    if avg <= 60:
        return 3
    if avg <= 80:
        return 4
    return 5