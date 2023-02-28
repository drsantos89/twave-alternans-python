"""Integration tests."""
from twaextractor.main import TWAExtractor


def test_with_data() -> None:
    """Test with data."""
    twa = TWAExtractor(path="tests/data/100")

    twa.extract()

    assert len(twa.k_score) == 2
    assert twa.k_score[0]["k_score"] < 0.0
    assert twa.k_score[1]["k_score"] < 0.0
