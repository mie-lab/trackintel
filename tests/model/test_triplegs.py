import os
import pytest

from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
import datetime

import trackintel as ti


@pytest.fixture
def testdata_tpls():
    """Read triplegs test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    return tpls


@pytest.fixture
def example_triplegs():
    """Triplegs to load into the database."""
    # three linestring geometries that are only slightly different (last coordinate)
    g1 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.4)])
    g2 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.5)])
    g3 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.6)])

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {"id": 0, "user_id": 0, "started_at": t1, "finished_at": t2, "geom": g1},
        {"id": 1, "user_id": 0, "started_at": t2, "finished_at": t3, "geom": g2},
        {"id": 2, "user_id": 1, "started_at": t3, "finished_at": t3 + one_hour, "geom": g3},
    ]

    tpls = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    tpls.set_index("id", inplace=True)

    assert tpls.as_triplegs
    return tpls


class TestTriplegs:
    """Tests for the TriplegsAccessor."""

    def test_accessor_column(self, testdata_tpls):
        """Test if the as_triplegs accessor checks the required column for triplegs."""
        tpls = testdata_tpls.copy()

        assert tpls.as_triplegs

        # check user_id
        with pytest.raises(AttributeError):
            tpls.drop(["user_id"], axis=1).as_triplegs

    def test_accessor_geometry(self, testdata_tpls):
        """Test if the as_triplegs accessor requires geometry column."""
        tpls = testdata_tpls.copy()

        # check geometry
        with pytest.raises(AttributeError):
            tpls.drop(["geom"], axis=1).as_triplegs

    def test_accessor_geometry_type(self, testdata_tpls):
        """Test if the as_triplegs accessor requires LineString geometry."""
        tpls = testdata_tpls.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a LineString"):
            tpls["geom"] = Point([(13.476808430, 48.573711823)])
            tpls.as_triplegs

    def test_accessor_empty_geometry(self, example_triplegs):
        """The accessor should accept empty geometries"""
        tpls = example_triplegs.copy()
        tpls.loc[2, "geom"] = LineString()

        tpls.as_triplegs

    def test_accessor_empty_geometry_warn(self, example_triplegs):
        """The accessor should warn about empty geometries"""
        tpls = example_triplegs.copy()
        tpls.loc[2, "geom"] = LineString()

        with pytest.warns(UserWarning, match="Dataframe contains empty geometries.*"):
            tpls.as_triplegs

    def test_accessor_missing_geometry(self, example_triplegs):
        """The accessor should accept missing geometries"""
        tpls = example_triplegs.copy()
        tpls.loc[2, "geom"] = None

        tpls.as_triplegs

    def test_accessor_missing_geometry_warn(self, example_triplegs):
        """The accessor should warn about missing geometries"""
        tpls = example_triplegs.copy()
        tpls.loc[2, "geom"] = None

        with pytest.warns(UserWarning, match="Dataframe contains missing geometries.*"):
            tpls.as_triplegs
