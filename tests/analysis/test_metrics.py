import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from shapely.geometry import Point

import trackintel as ti
from trackintel.analysis import radius_gyration, jump_length
from trackintel.geogr import point_haversine_dist


@pytest.fixture
def staypoints():
    """Staypoints for two users.
    First user has 5 unique points all with the same degree distance from each other, but different timestamps.
    Second user has 2 unique points, but duration of one point is zero.
    """
    p1 = Point(0.0, 9.0)
    p2 = Point(3.0, 12.0)
    p3 = Point(6.0, 15.0)

    t = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    h = pd.Timedelta(hours=1)

    list_dict = [
        {"id": 1, "user_id": 0, "started_at": t + 0 * h, "finished_at": t + 1 * h, "geom": p1},
        {"id": 2, "user_id": 0, "started_at": t + 1 * h, "finished_at": t + 3 * h, "geom": p2},
        {"id": 3, "user_id": 0, "started_at": t + 3 * h, "finished_at": t + 4 * h, "geom": p3},
        {"id": 4, "user_id": 1, "started_at": t + 0 * h, "finished_at": t + 1 * h, "geom": p1},
        {"id": 7, "user_id": 1, "started_at": t + 1 * h, "finished_at": t + 2 * h, "geom": p1},
        {"id": 8, "user_id": 1, "started_at": t + 2 * h, "finished_at": t + 2 * h, "geom": p3},
    ]
    sp = ti.Staypoints(data=list_dict, geometry="geom", crs="EPSG:2056")
    return sp


class TestRadius_gyration:
    def test_unknown_method(self, staypoints):
        """Test if unknown method raises a ValueError"""
        with pytest.raises(
            ValueError, match='Method unknown. Should be on of {"count", "duration"}. You passed "unknown"'
        ):
            radius_gyration(staypoints, method="unknown")

    def test_count(self, staypoints):
        """Test count-method with planar crs"""
        s = radius_gyration(staypoints, method="count")
        v1 = np.sqrt((4 * 3**2) / 3)  # center is (3, 12)
        v2 = np.sqrt((2 * (2**3 + 4**2)) / 3)  # center is (2, 11)
        assert_series_equal(s, pd.Series([v1, v2]), check_index=False, check_names=False)

    def test_duration(self, staypoints):
        """Test duration-method with a planar crs."""
        staypoints["duration"] = staypoints["finished_at"] - staypoints["started_at"]
        s = radius_gyration(staypoints, method="duration")
        v1 = np.sqrt(4 * 3**2 / 4)  # center is (3, 12) weight is 4 due to duration
        v2 = 0  # center lies on p1 and only p1 remains -> 0 variance
        assert_series_equal(s, pd.Series([v1, v2]), check_index=False, check_names=False)

    def test_haversine(self, staypoints):
        """Test haversine distance calculations"""
        staypoints = staypoints.set_crs(4326, allow_override=True)
        s = radius_gyration(staypoints, method="count")
        x = staypoints.geometry.x
        y = staypoints.geometry.y
        f1 = staypoints["user_id"] == 0
        f2 = staypoints["user_id"] == 1
        d1 = point_haversine_dist(x[f1], y[f1], 3, 12)
        d2 = point_haversine_dist(x[f2], y[f2], 2, 11)
        v1 = np.sqrt(np.mean(d1**2))
        v2 = np.sqrt(np.mean(d2**2))
        assert_series_equal(s, pd.Series([v1, v2]), check_index=False, check_names=False)

    def test_tqdm(self, staypoints):
        """Test if tqdm works fine"""
        radius_gyration(staypoints, print_progress=True)

    def test_staypoints_method(self, staypoints):
        """Test if staypoint method returns same result"""
        sfunc = radius_gyration(staypoints)
        smeth = staypoints.radius_gyration()
        assert_series_equal(sfunc, smeth)


class TestJump_length:
    def test_planar(self, staypoints):
        """Test planar jump length implementation"""
        s = jump_length(staypoints)
        s_test = pd.Series([np.sqrt(18), np.sqrt(18), np.nan, 0, np.sqrt(72), np.nan])
        assert_series_equal(s, s_test, check_names=False)

    def test_haversine(self, staypoints):
        """Test haversine jump length implementation"""
        staypoints = staypoints.set_crs(4326, allow_override=True)
        s = jump_length(staypoints)
        g = staypoints.geometry
        j1 = point_haversine_dist(g.iloc[0].x, g.iloc[0].y, g.iloc[1].x, g.iloc[1].y, float_flag=True)
        j2 = point_haversine_dist(g.iloc[1].x, g.iloc[1].y, g.iloc[2].x, g.iloc[2].y, float_flag=True)
        j3 = point_haversine_dist(g.iloc[-2].x, g.iloc[-2].y, g.iloc[-1].x, g.iloc[-1].y, float_flag=True)
        s_test = pd.Series([j1, j2, np.nan, 0, j3, np.nan])
        assert_series_equal(s, s_test, check_names=False)

    def test_unordered(self, staypoints):
        """Test if staypoints get sorted correctly"""
        s1 = jump_length(staypoints)
        staypoints = staypoints.sample(frac=1)
        s2 = jump_length(staypoints)
        s2 = s2.sort_index()
        assert_series_equal(s1, s2)

    def test_random_index(self, staypoints):
        """Test if does not need consecutive index"""
        s1 = jump_length(staypoints)
        staypoints.index = [100, 1, 10, 1000, 1001, 999]
        s2 = jump_length(staypoints)
        s1.index = staypoints.index
        assert_series_equal(s1, s2)

    def test_staypoints_method(self, staypoints):
        """Test if staypoint method returns same result"""
        sfunc = jump_length(staypoints)
        smeth = staypoints.jump_length()
        assert_series_equal(sfunc, smeth)
