import pytest
import os

import trackintel as ti
from trackintel import Tours


@pytest.fixture
def test_tours():
    """Read test data from file"""
    tours_file = os.path.join("tests", "data", "tours.csv")
    tours = ti.read_tours_csv(tours_file, sep=";", index_col="id")
    return tours


class TestTours:
    """Tests for the Tours class."""

    def test_missing_column(self, test_tours):
        """Test if as_tours accessor check required columns"""
        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of tours"):
            test_tours.drop(columns="user_id").as_tours

    def test_check_succeeding(self, test_tours):
        """Test if check returns True on valid tours"""
        assert Tours._check(test_tours)

    def test_check_missing_columns(self, test_tours):
        """Test if check returns False if column is missing"""
        assert not Tours._check(test_tours.drop(columns="user_id"))

    def test_check_no_tz(self, test_tours):
        """Test if check returns False if datetime columns have no tz"""
        tmp = test_tours["started_at"]
        test_tours["started_at"] = test_tours["started_at"].dt.tz_localize(None)
        assert not Tours._check(test_tours)
        test_tours["started_at"] = tmp
        test_tours["finished_at"] = test_tours["finished_at"].dt.tz_localize(None)
        assert not Tours._check(test_tours)
