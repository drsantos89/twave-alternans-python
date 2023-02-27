"""Integration tests."""
from twaextractor.main import TWAExtractor


def test_with_data() -> None:
    """Test with data."""
    twa = TWAExtractor(path="tests/data/100")
    twa.extract()
    assert twa.k_score == 0.0
