import os
import pytest
import numpy as np

from shapely.geometry import LineString, Point

import trackintel as ti
import pandas as pd
import geopandas as gpd


@pytest.fixture
def testdata_sp():
    """Read sp test data from files."""
    sp_file = os.path.join("tests", "data", "staypoints.csv")
    sp = ti.read_staypoints_csv(sp_file, sep=";", index_col="id")
    return sp


@pytest.fixture
def example_staypoints():
    """Example Staypoints"""

    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    t4 = pd.Timestamp("1971-01-02 08:00:00", tz="utc")

    list_dict = [
        {"id": 1, "user_id": 0, "started_at": t1, "finished_at": t2, "geom": p1},
        {"id": 5, "user_id": 0, "started_at": t2, "finished_at": t3, "geom": p2},
        {"id": 2, "user_id": 0, "started_at": t3, "finished_at": t4, "geom": p3},
    ]
    sp = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    sp = sp.set_index("id")
    return sp


class TestStaypoints:
    """Tests for the StaypointsAccessor."""

    def test_accessor_columns(self, testdata_sp):
        """Test if the as_staypoints accessor checks the required column for staypoints."""
        sp = testdata_sp.copy()
        assert sp.as_staypoints

        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of staypoints"):
            sp.drop(["user_id"], axis=1).as_staypoints

    def test_accessor_geometry(self, testdata_sp):
        """Test if the as_staypoints accessor requires geometry column."""
        sp = testdata_sp.copy()

        # geometery
        with pytest.raises(AttributeError):
            sp.drop(["geom"], axis=1).as_staypoints

    def test_accessor_geometry_type(self, testdata_sp):
        """Test if the as_staypoints accessor requires Point geometry."""
        sp = testdata_sp.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a Point"):
            sp["geom"] = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
            sp.as_staypoints

    def test_staypoints_center(self, testdata_sp):
        """Check if sp has center method and returns (lat, lon) pairs as geometry."""
        sp = testdata_sp.copy()
        assert len(sp.as_staypoints.center) == 2

    def test_accessor_empty_geometry(self, example_staypoints):
        """The accessor should accept empty geometries"""
        sp = example_staypoints.copy()
        sp.loc[2, "geom"] = Point()

        sp.as_staypoints

    def test_accessor_empty_geometry_warn(self, example_staypoints):
        """The accessor should warn about empty geometries"""
        sp = example_staypoints.copy()
        sp.loc[2, "geom"] = Point()

        with pytest.warns(UserWarning, match="Dataframe contains empty geometries.*"):
            sp.as_staypoints

    def test_accessor_missing_geometry(self, example_staypoints):
        """The accessor should accept missing geometries"""
        sp = example_staypoints.copy()
        sp.loc[2, "geom"] = None

        sp.as_staypoints

    def test_accessor_missing_geometry_warn(self, example_staypoints):
        """The accessor should warn about missing geometries"""
        sp = example_staypoints.copy()
        sp.loc[2, "geom"] = None

        with pytest.warns(UserWarning, match="Dataframe contains missing geometries.*"):
            sp.as_staypoints
