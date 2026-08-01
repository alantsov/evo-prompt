from evo_prompt.simple_gepa import _extract_scalar_score

def test_extract_scalar_score_standard_mapping():
    """Test the specific string-to-float mappings defined in the logic (with default weights)."""
    rubric = {
        "dim1": {"score": "2"},    # 1.0
        "dim2": {"score": "1"},    # 0.6
        "dim3": {"score": "0"},    # 0.0
        "dim4": {"score": "N/A"}   # 0.4
    }
    # Average: (1.0 + 0.6 + 0.0 + 0.4) / 4 = 0.5
    assert _extract_scalar_score(rubric, weights={}) == 0.5

def test_extract_scalar_score_weighted_logic():
    """Test that high weights correctly skew the final scalar score."""
    rubric = {
        "important": {"score": "2"}, # Score 1.0
        "unimportant": {"score": "0"} # Score 0.0
    }
    # Weight 'important' as 3.0 and 'unimportant' as 1.0
    # Expected: (1.0 * 3.0 + 0.0 * 1.0) / (3.0 + 1.0) = 3.0 / 4.0 = 0.75
    weights = {"important": 3.0, "unimportant": 1.0}
    assert _extract_scalar_score(rubric, weights=weights) == 0.75

def test_extract_scalar_score_raw_floats():
    """Test that actual float strings are divided by max_rubric_score (2.0)."""
    rubric = {
        "dim1": {"score": "1.2"} # 1.2 / 2.0 = 0.6
    }
    assert _extract_scalar_score(rubric, weights={}) == 0.6

def test_extract_scalar_score_empty_or_no_scores():
    """Test behavior when no valid scores are found."""
    assert _extract_scalar_score({}) == 0.0
    assert _extract_scalar_score({"dim1": {"note": "no score here"}}) == 0.0

def test_extract_scalar_score_mixed_types():
    """Test a mix of integer-style strings and float-style strings."""
    rubric = {
        "dim1": {"score": "2"},  # 1.0
        "dim2": {"score": "0.4"} # 0.4 / 2.0 = 0.2
    }
    # Average: (1.0 + 0.2) / 2 = 0.6
    assert _extract_scalar_score(rubric, weights={}) == 0.6
