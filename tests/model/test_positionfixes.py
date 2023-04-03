import os
import pytest
import numpy as np

from shapely.geometry import LineString, Point
import pandas as pd
import geopandas as gpd

import trackintel as ti


@pytest.fixture
def example_positionfixes():
    """Positionfixes for tests."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")

    list_dict = [
        {"user_id": 0, "tracked_at": t1, "geometry": p1},
        {"user_id": 0, "tracked_at": t2, "geometry": p2},
        {"user_id": 1, "tracked_at": t3, "geometry": p3},
    ]
    pfs = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    pfs.index.name = "id"

    return pfs


@pytest.fixture
def testdata_geolife():
    """Read geolife test data from files."""
    pfs, _ = ti.io.dataset_reader.read_geolife(os.path.join("tests", "data", "geolife"))
    return pfs


class TestPositionfixes:
    """Tests for the PositionfixesAccessor."""

    def test_accessor_column(self, testdata_geolife):
        """Test if the as_positionfixes accessor checks the required column for positionfixes."""
        pfs = testdata_geolife.copy()
        assert pfs.as_positionfixes

        # check user_id
        with pytest.raises(AttributeError, match="To process a DataFrame as a collection of positionfixes"):
            pfs.drop(["user_id"], axis=1).as_positionfixes

    def test_accessor_geometry(self, testdata_geolife):
        """Test if the as_positionfixes accessor requires geometry column."""
        pfs = testdata_geolife.copy()

        # check geometry
        with pytest.raises(AttributeError):
            pfs.drop(["geom"], axis=1).as_positionfixes

    def test_accessor_geometry_type(self, testdata_geolife):
        """Test if the as_positionfixes accessor requires Point geometry."""
        pfs = testdata_geolife.copy()

        # check geometry type
        with pytest.raises(AttributeError, match="The geometry must be a Point"):
            pfs["geom"] = LineString([(13.476808430, 48.573711823), (13.506804, 48.939008), (13.4664690, 48.5706414)])
            pfs.as_positionfixes

    def test_accessor_empty_geometry(self, example_positionfixes):
        """The accessor should accept empty geometries"""
        pfs = example_positionfixes.copy()
        pfs.loc[2, "geometry"] = Point()

        pfs.as_positionfixes

    def test_accessor_empty_geometry_warn(self, example_positionfixes):
        """The accessor should warn about empty geometries"""
        pfs = example_positionfixes.copy()
        pfs.loc[2, "geometry"] = Point()

        with pytest.warns(UserWarning, match="Dataframe contains empty geometries.*"):
            pfs.as_positionfixes

    def test_accessor_missing_geometry(self, example_positionfixes):
        """The accessor should accept missing geometries"""
        pfs = example_positionfixes.copy()
        pfs.loc[2, "geometry"] = None

        pfs.as_positionfixes

    def test_accessor_missing_geometry_warn(self, example_positionfixes):
        """The accessor should warn about missing geometries"""
        pfs = example_positionfixes.copy()
        pfs.loc[2, "geometry"] = None

        with pytest.warns(UserWarning, match="Dataframe contains positionfixes with missing geometries.*"):
            pfs.as_positionfixes

    def test_center(self, testdata_geolife):
        """Check if pfs has center method and returns (lat, lon) pairs as geometry."""
        pfs = testdata_geolife
        assert len(pfs.as_positionfixes.center) == 2

    def test_similarity_matrix(self, testdata_geolife):
        """Check the similarity_matrix function called through accessor runs as expected."""
        pfs = testdata_geolife

        accessor_result = pfs.as_positionfixes.calculate_distance_matrix(dist_metric="haversine", n_jobs=1)
        function_result = ti.geogr.distances.calculate_distance_matrix(pfs, dist_metric="haversine", n_jobs=1)
        assert np.allclose(accessor_result, function_result)
