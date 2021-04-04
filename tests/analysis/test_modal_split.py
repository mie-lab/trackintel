import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from trackintel.analysis.modal_split import calculate_modal_split
from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs


@pytest.fixture
def read_geolife_with_modes():
    pfs, labels = read_geolife(os.path.join("tests", "data", "geolife_modes"))

    pfs, stps = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(stps, method="between_staypoints")

    tpls_with_modes = geolife_add_modes_to_triplegs(tpls, labels)
    return tpls_with_modes


@pytest.fixture
def ls_short():
    """a linestring that is short (two points within passau)"""
    return LineString([(13.476808430, 48.573711823), (13.4664690, 48.5706414)])


@pytest.fixture
def ls_long():
    """a linestring that is long (Passau - Munich)"""
    return LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459)])


@pytest.fixture
def test_triplegs_modal_split(ls_short, ls_long):
    """Triplegs with transport modes that can be aggregated over days and weeks.

    user 0: day 1:  2 triplegs (car + bike)
            day 2:  1 tripleg (walk)
            day 8: 2 triplegs (walk, walk)
    user 1: day 1: 1 tripleg (walk)
    """

    day_1_h1 = pd.Timestamp("1970-01-01 00:00:00", tz="utc")
    day_1_h2 = pd.Timestamp("1970-01-01 02:00:00", tz="utc")
    day_2_h1 = pd.Timestamp("1970-01-02 02:00:00", tz="utc")
    day_8_h1 = pd.Timestamp("1970-01-08 02:00:00", tz="utc")

    one_hour = datetime.timedelta(hours=1)
    one_min = datetime.timedelta(minutes=1)

    list_dict = [
        {
            "id": 0,
            "user_id": 0,
            "started_at": day_1_h1,
            "finished_at": day_1_h1 + one_hour,
            "geometry": ls_short,
            "mode": "car",
        },
        {
            "id": 1,
            "user_id": 0,
            "started_at": day_1_h2,
            "finished_at": day_1_h2 + one_min,
            "geometry": ls_long,
            "mode": "bike",
        },
        {
            "id": 2,
            "user_id": 0,
            "started_at": day_2_h1,
            "finished_at": day_2_h1 + one_hour,
            "geometry": ls_short,
            "mode": "walk",
        },
        {
            "id": 3,
            "user_id": 0,
            "started_at": day_8_h1,
            "finished_at": day_8_h1 + one_hour,
            "geometry": ls_short,
            "mode": "walk",
        },
        {
            "id": 4,
            "user_id": 1,
            "started_at": day_1_h1,
            "finished_at": day_1_h1 + one_hour,
            "geometry": ls_long,
            "mode": "walk",
        },
        {
            "id": 5,
            "user_id": 1,
            "started_at": day_1_h2,
            "finished_at": day_1_h2 + one_hour,
            "geometry": ls_long,
            "mode": "walk",
        },
    ]

    df = pd.DataFrame(list_dict)
    gdf = gpd.GeoDataFrame(df).set_geometry("geometry")
    gdf = gdf.set_crs("wgs84")
    return gdf


class TestModalSplit:
    def test_run_modal_split_with_geolife(self, read_geolife_with_modes):
        """check if we can run all possible combinations with a small sample of the geolife data"""

        tpls = read_geolife_with_modes
        metric_list = ["duration", "distance", "count"]
        freq_list = [None, "D", "W-MON"]
        per_user_list = [False, True]
        norm_list = [True, False]

        for metric in metric_list:
            for freq in freq_list:
                for per_user in per_user_list:
                    for norm in norm_list:
                        calculate_modal_split(tpls, metric=metric, freq=freq, per_user=per_user, norm=norm)

        # we only check if it runs through successfully
        assert True

    def test_run_modal_split_with_geolife_accessors(self, read_geolife_with_modes):
        """check if we can access `calculate_modal_split` via the tripelg accessor"""

        tpls = read_geolife_with_modes
        tpls.as_triplegs.calculate_modal_split(metric="count", freq="D")

        # we only check if it runs through successfully
        assert True

    def test_modal_split_total_count(self, test_triplegs_modal_split):
        """Check counts per user and mode without temporal binning"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="count", per_user=True)

        # if the user get merged, walk would be larger than bike
        assert modal_split.loc[0, "bike"] == 1
        assert modal_split.loc[0, "car"] == 1
        assert modal_split.loc[0, "walk"] == 2
        assert modal_split.loc[1, "walk"] == 2

    def test_modal_split_total_no_user_count(self, test_triplegs_modal_split):
        """Check counts without users and without temporal binning"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="count", per_user=False)

        # if the user get merged, walk would be larger than bike
        assert modal_split.loc[0, "bike"] == 1
        assert modal_split.loc[0, "car"] == 1
        assert modal_split.loc[0, "walk"] == 4

    def test_modal_split_total_distance(self, test_triplegs_modal_split):
        """Check distances per user and mode without temporal binning"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="distance", per_user=True)

        # if the users would get merged, walk would be larger than bike
        assert modal_split.loc[0, "bike"] > modal_split.loc[0, "walk"]
        assert modal_split.loc[0, "walk"] > modal_split.loc[0, "car"]

    def test_modal_split_total_duration(self, test_triplegs_modal_split):
        """Check duration per user and mode without temporal binning"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="duration", per_user=True)

        assert np.isclose(modal_split.loc[0, "bike"], datetime.timedelta(minutes=1).total_seconds())
        assert np.isclose(modal_split.loc[0, "car"], datetime.timedelta(hours=1).total_seconds())
        assert np.isclose(modal_split.loc[0, "walk"], datetime.timedelta(hours=2).total_seconds())
        assert np.isclose(modal_split.loc[1, "walk"], datetime.timedelta(hours=2).total_seconds())

    def test_modal_split_daily_count(self, test_triplegs_modal_split):
        """Check counts per user and mode binned by day"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="count", freq="D", per_user=True)

        t_1 = pd.Timestamp("1970-01-01 00:00:00", tz="utc")
        t_2 = pd.Timestamp("1970-01-02 00:00:00", tz="utc")
        t_8 = pd.Timestamp("1970-01-08 00:00:00", tz="utc")

        assert modal_split.loc[[(0, t_1)], "bike"][0] == 1
        assert modal_split.loc[[(0, t_1)], "car"][0] == 1
        assert modal_split.loc[[(0, t_2)], "walk"][0] == 1
        assert modal_split.loc[[(0, t_8)], "walk"][0] == 1
        assert modal_split.loc[[(1, t_1)], "walk"][0] == 2

    def test_modal_split_weekly_count(self, test_triplegs_modal_split):
        """Check counts per user and mode binned by week"""
        tpls = test_triplegs_modal_split
        modal_split = calculate_modal_split(tpls, metric="count", freq="W-MON", per_user=True)

        w_1 = pd.Timestamp("1970-01-05 00:00:00", tz="utc")  # data is aggregated to the next monday
        w_2 = pd.Timestamp("1970-01-12 00:00:00", tz="utc")

        assert modal_split.loc[[(0, w_1)], "bike"][0] == 1
        assert modal_split.loc[[(0, w_1)], "car"][0] == 1
        assert modal_split.loc[[(0, w_1)], "walk"][0] == 1
        assert modal_split.loc[[(0, w_2)], "walk"][0] == 1
        assert modal_split.loc[[(1, w_1)], "walk"][0] == 2
