import datetime

import pytest

from trackintel.preprocessing.util import calc_temp_overlap


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
