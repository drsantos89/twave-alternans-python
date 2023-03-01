"""Integration tests."""
import pytest

from twaextractor.main import TWAExtractor


@pytest.mark.parametrize(
    "path, n_sig", [("tests/data/100", 2), ("tests/data/twa34", 12)]
)
def test_with_data(path, n_sig) -> None:
    """Test with data."""
    twa = TWAExtractor(path=path)

    twa.extract()

    assert len(twa.k_score) == n_sig
    assert twa.k_score[0]["k_score"] < 0.0
    assert twa.k_score[1]["k_score"] < 0.0
