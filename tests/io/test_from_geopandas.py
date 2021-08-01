import os

import geopandas as gpd
import pandas as pd
import pytest
import trackintel as ti
from pandas.testing import assert_frame_equal
from shapely.geometry import Point
from trackintel.io.from_geopandas import (
    _trackintel_model,
    read_locations_gpd,
    read_positionfixes_gpd,
    read_staypoints_gpd,
    read_tours_gpd,
    read_triplegs_gpd,
    read_trips_gpd,
)


@pytest.fixture()
def example_positionfixes():
    """Positionfixes to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 04:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")

    list_dict = [
        {"user_id": 0, "tracked_at": t1, "geom": p1},
        {"user_id": 0, "tracked_at": t2, "geom": p2},
        {"user_id": 1, "tracked_at": t3, "geom": p3},
    ]
    pfs = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    pfs.index.name = "id"
    assert pfs.as_positionfixes
    return pfs


class Test_Trackintel_Model:
    """Test `_trackintel_model()` function."""

    def test_renaming(self, example_positionfixes):
        """Test renaming of columns"""
        example_positionfixes["additional_col"] = [11, 22, 33]
        pfs = example_positionfixes.copy()
        columns = {"user_id": "_user_id", "tracked_at": "_tracked_at", "additional_col": "_additional_col"}
        columns_rev = {val: key for key, val in columns.items()}
        pfs.rename(columns=columns, inplace=True)
        pfs = _trackintel_model(pfs, columns_rev)
        assert_frame_equal(example_positionfixes, pfs)

    def test_setting_geometry(self, example_positionfixes):
        """Test the setting of the geometry."""
        pfs = pd.DataFrame(example_positionfixes[["user_id", "tracked_at"]], copy=True)
        pfs["geom"] = example_positionfixes.geometry
        pfs = _trackintel_model(pfs, geom_col="geom")
        assert_frame_equal(example_positionfixes, pfs)

    def test_set_crs(self, example_positionfixes):
        """Test if crs will be set."""
        pfs = example_positionfixes.copy()
        pfs.crs = None
        pfs = _trackintel_model(pfs, crs="EPSG:4326")
        assert_frame_equal(example_positionfixes, pfs)

    def test_already_set_geometry(self, example_positionfixes):
        """Test if default checks if GeoDataFrame already has a geometry."""
        pfs = _trackintel_model(example_positionfixes)
        assert_frame_equal(pfs, example_positionfixes)

    def test_error_no_set_geometry(self, example_positionfixes):
        """Test if AttributeError is risen if no geom_col is provided and GeoDataFrame has no geometry."""
        pfs = gpd.GeoDataFrame(example_positionfixes[["user_id", "tracked_at"]], copy=True)
        with pytest.raises(AttributeError):
            _trackintel_model(pfs)

    def test_tz_cols(self, example_positionfixes):
        """Test if columns get casted to datetimes."""
        pfs = example_positionfixes.copy()
        pfs["tracked_at"] = ["1971-01-01 04:00:00", "1971-01-01 05:00:00", "1971-01-02 07:00:00"]
        pfs = _trackintel_model(pfs, tz_cols=["tracked_at"], tz="UTC")
        assert_frame_equal(pfs, example_positionfixes)


class TestRead_Positionfixes_Gpd:
    """Test `read_positionfixes_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        gdf = gpd.read_file(os.path.join("tests", "data", "positionfixes.geojson"))
        gdf.set_index("id", inplace=True)
        pfs_from_gpd = read_positionfixes_gpd(gdf, user_id="User", geom_col="geometry", crs="EPSG:4326", tz="utc")

        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs_from_csv = ti.read_positionfixes_csv(pfs_file, sep=";", tz="utc", index_col="id")
        pfs_from_csv = pfs_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(pfs_from_gpd, pfs_from_csv, check_exact=False)


class TestRead_Triplegs_Gpd:
    """Test `read_triplegs_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        gdf = gpd.read_file(os.path.join("tests", "data", "triplegs.geojson"))
        gdf.set_index("id", inplace=True)
        tpls_from_gpd = read_triplegs_gpd(gdf, user_id="User", geom_col="geometry", crs="EPSG:4326", tz="utc")

        tpls_file = os.path.join("tests", "data", "triplegs.csv")
        tpls_from_csv = ti.read_triplegs_csv(tpls_file, sep=";", tz="utc", index_col="id")
        tpls_from_csv = tpls_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(tpls_from_gpd, tpls_from_csv, check_exact=False)


class TestRead_Staypoints_Gpd:
    """Test `read_staypoints_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        gdf = gpd.read_file(os.path.join("tests", "data", "staypoints.geojson"))
        gdf.set_index("id", inplace=True)
        stps_from_gpd = read_staypoints_gpd(
            gdf, "start_time", "end_time", geom_col="geometry", crs="EPSG:4326", tz="utc"
        )

        stps_file = os.path.join("tests", "data", "staypoints.csv")
        stps_from_csv = ti.read_staypoints_csv(stps_file, sep=";", tz="utc", index_col="id")
        stps_from_csv = stps_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(stps_from_gpd, stps_from_csv, check_exact=False)


class TestRead_Locations_Gpd:
    """Test `read_locations_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        # TODO: Problem with multiple geometry columns and csv format
        gdf = gpd.read_file(os.path.join("tests", "data", "locations.geojson"))
        gdf.set_index("id", inplace=True)
        locs_from_gpd = read_locations_gpd(gdf, user_id="User", center="geometry", crs="EPSG:4326")

        locs_file = os.path.join("tests", "data", "locations.csv")
        locs_from_csv = ti.read_locations_csv(locs_file, sep=";", index_col="id")

        # drop the second geometry column manually because not storable in GeoJSON (from Geopandas)
        locs_from_csv = locs_from_csv.drop(columns="extent")
        assert_frame_equal(locs_from_csv, locs_from_gpd, check_exact=False)


class TestRead_Trips_Gpd:
    """Test `read_trips_gpd()` function."""

    def test_cvs(self):
        df = pd.read_csv(os.path.join("tests", "data", "trips.csv"), sep=";")
        df.set_index("id", inplace=True)
        trips_from_gpd = read_trips_gpd(df, tz="utc")

        trips_file = os.path.join("tests", "data", "trips.csv")
        trips_from_csv = ti.read_trips_csv(trips_file, sep=";", tz="utc", index_col="id")

        assert_frame_equal(trips_from_gpd, trips_from_csv, check_exact=False)


class TestRead_Tours_Gpd:
    """Test `read_trips_gpd()` function."""

    def test_read_tours_gpd(self):
        # TODO: implement tests for reading tours from Geopandas
        pass
