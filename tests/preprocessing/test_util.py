import datetime

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from trackintel.preprocessing.util import calc_temp_overlap, _explode_agg


@pytest.fixture
def time_1():
    return datetime.datetime(year=1, month=1, day=1, hour=0, minute=0, second=0)


@pytest.fixture
def one_hour():
    return datetime.timedelta(hours=1)


class TestCalc_temp_overlap:
    def test_same_interval(self, time_1, one_hour):
        """Two equal intervals should have 100 % overlap"""
        ratio = calc_temp_overlap(time_1, time_1 + one_hour, time_1, time_1 + one_hour)
        assert ratio == 1

    def test_1_in_2(self, time_1, one_hour):
        """If interval 1 is fully covered by interval 2 the overlap should be 100 %"""
        ratio = calc_temp_overlap(time_1, time_1 + one_hour, time_1, time_1 + 2 * one_hour)
        assert ratio == 1

    def test_2_in_1(self, time_1, one_hour):
        """If interval 1 is only covered half by interval 2 the overlap should be 50 %"""
        ratio = calc_temp_overlap(time_1, time_1 + 2 * one_hour, time_1, time_1 + one_hour)
        assert ratio == 0.5

    def test_no_overlap(self, time_1, one_hour):
        """If the two intervals do not overlap the ratio should be 0"""
        ratio = calc_temp_overlap(time_1, time_1 + one_hour, time_1 + one_hour, time_1 + 2 * one_hour)
        assert ratio == 0


class TestExplodeAgg:
    """Test util method _explode_agg"""

    def test_empty_agg(self):
        """Test function with empty agg DataFrame"""
        orig = [
            {"a": 1, "b": "i", "c": None},
            {"a": 2, "b": "i", "c": None},
        ]
        orig_df = pd.DataFrame(orig, columns=["a", "b"])
        agg_df = pd.DataFrame({}, columns=["id", "c"])
        returned_df = _explode_agg("id", "c", orig_df, agg_df)
        solution_df = pd.DataFrame(orig)
        assert_frame_equal(returned_df, solution_df)

    def test_list_column(self):
        """Test function with a column of lists."""
        orig = [
            {"a": 1, "c": 0},
            {"a": 2, "c": 0},
            {"a": 3, "c": None},
            {"a": 4, "c": 1},
        ]
        agg = [
            {"id": [0, 1], "c": 0},
            {"id": [3], "c": 1},
        ]
        orig_df = pd.DataFrame(orig, columns=["a"])
        agg_df = pd.DataFrame(agg)
        returned_df = _explode_agg("id", "c", orig_df, agg_df)
        solution_df = pd.DataFrame(orig)
        assert_frame_equal(returned_df, solution_df)

    def test_index_dtype_with_None(self):
        """Test if dtype of index isn't changed with None values."""
        orig = [
            {"a": 1, "c": 0},
        ]
        agg = [{"id": [0, 1], "c": 0}, {"id": [], "c": 1}]
        orig_df = pd.DataFrame(orig, columns=["a"])
        agg_df = pd.DataFrame(agg)
        returned_df = _explode_agg("id", "c", orig_df, agg_df)
        solution_df = pd.DataFrame(orig)
        assert_frame_equal(returned_df, solution_df)
