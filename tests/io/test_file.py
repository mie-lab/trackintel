import filecmp
import os

import geopandas as gpd
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

import trackintel as ti


class TestPositionfixes:
    """Test for 'read_positionfixes_csv' and 'write_positionfixes_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "positionfixes.csv")
        mod_file = os.path.join("tests", "data", "positionfixes_mod_columns.csv")
        tmp_file = os.path.join("tests", "data", "positionfixes_test_1.csv")

        pfs = ti.read_positionfixes_csv(orig_file, sep=";", index_col="id")

        column_mapping = {"lat": "latitude", "lon": "longitude", "time": "tracked_at"}
        mod_pfs = ti.read_positionfixes_csv(mod_file, sep=";", index_col="id", columns=column_mapping)
        assert mod_pfs.equals(pfs)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        columns = ["user_id", "tracked_at", "latitude", "longitude", "elevation", "accuracy"]
        pfs.as_positionfixes.to_csv(tmp_file, sep=";", columns=columns, date_format=date_format)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_set_crs(self):
        """Test setting the crs when reading."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col="id")
        assert pfs.crs is None

        crs = "EPSG:2056"
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col="id", crs=crs)
        assert pfs.crs == crs

    def test_set_datatime_tz(self):
        """Test setting the timezone infomation when reading."""
        # check if tz is added to the datatime column
        file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col="id")
        assert isinstance(pfs["tracked_at"].dtype, pd.DatetimeTZDtype)

        # check if a timezone will be set without storing the timezone
        date_format = "%Y-%m-%d %H:%M:%S"
        tmp_file = os.path.join("tests", "data", "positionfixes_test_2.csv")
        pfs.as_positionfixes.to_csv(tmp_file, sep=";", date_format=date_format)
        pfs = ti.read_positionfixes_csv(tmp_file, sep=";", index_col="id", tz="utc")

        assert isinstance(pfs["tracked_at"].dtype, pd.DatetimeTZDtype)

        # check if a warning is raised if 'tz' is not provided
        with pytest.warns(UserWarning):
            ti.read_positionfixes_csv(tmp_file, sep=";", index_col="id")

        os.remove(tmp_file)

    def test_set_index_warning(self):
        """Test if a warning is raised when not parsing the index_col argument."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        with pytest.warns(UserWarning):
            ti.read_positionfixes_csv(file, sep=";")

    def test_set_index(self):
        """Test if `index_col` can be set."""
        file = os.path.join("tests", "data", "positionfixes.csv")
        ind_name = "id"
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col=ind_name)
        assert pfs.index.name == ind_name
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col=None)
        assert pfs.index.name is None

    def test_type(self):
        """Test if returned object is Positionfix"""
        file = os.path.join("tests", "data", "positionfixes.csv")
        ind_name = "id"
        pfs = ti.read_positionfixes_csv(file, sep=";", index_col=ind_name)
        assert isinstance(pfs, ti.Positionfixes)


class TestTriplegs:
    """Test for 'read_triplegs_csv' and 'write_triplegs_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "triplegs.csv")
        mod_file = os.path.join("tests", "data", "triplegs_mod_columns.csv")
        tmp_file = os.path.join("tests", "data", "triplegs_test_1.csv")
        tpls = ti.read_triplegs_csv(orig_file, sep=";", tz="utc", index_col="id")

        column_mapping = {"start_time": "started_at", "end_time": "finished_at", "tripleg": "geom"}
        mod_tpls = ti.read_triplegs_csv(mod_file, sep=";", columns=column_mapping, index_col="id")

        assert mod_tpls.equals(tpls)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        columns = ["user_id", "started_at", "finished_at", "geom"]
        tpls.as_triplegs.to_csv(tmp_file, sep=";", columns=columns, date_format=date_format)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_set_crs(self):
        """Test setting the crs when reading."""
        file = os.path.join("tests", "data", "triplegs.csv")
        crs = "EPSG:2056"
        tpls = ti.read_triplegs_csv(file, sep=";", tz="utc", index_col="id")
        assert tpls.crs is None

        tpls = ti.read_triplegs_csv(file, sep=";", tz="utc", index_col="id", crs=crs)
        assert tpls.crs == crs

    def test_set_datatime_tz(self):
        """Test setting the timezone infomation when reading."""
        # check if tz is added to the datatime column
        file = os.path.join("tests", "data", "triplegs.csv")
        tpls = ti.read_triplegs_csv(file, sep=";", index_col="id")
        assert isinstance(tpls["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a timezone will be set without storing the timezone
        tmp_file = os.path.join("tests", "data", "triplegs_test_2.csv")
        date_format = "%Y-%m-%d %H:%M:%S"
        tpls.as_triplegs.to_csv(tmp_file, sep=";", date_format=date_format)
        tpls = ti.read_triplegs_csv(tmp_file, sep=";", index_col="id", tz="utc")

        assert isinstance(tpls["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a warning is raised if 'tz' is not provided
        with pytest.warns(UserWarning):
            ti.read_triplegs_csv(tmp_file, sep=";", index_col="id")

        os.remove(tmp_file)

    def test_set_index_warning(self):
        """Test if a warning is raised when not parsing the index_col argument."""
        file = os.path.join("tests", "data", "triplegs.csv")
        with pytest.warns(UserWarning):
            ti.read_triplegs_csv(file, sep=";")

    def test_set_index(self):
        """Test if `index_col` can be set."""
        file = os.path.join("tests", "data", "triplegs.csv")
        ind_name = "id"
        pfs = ti.read_triplegs_csv(file, sep=";", index_col=ind_name)
        assert pfs.index.name == ind_name
        pfs = ti.read_triplegs_csv(file, sep=";", index_col=None)
        assert pfs.index.name is None

    def test_type(self):
        """Test if returned object is Triplegs"""
        file = os.path.join("tests", "data", "triplegs.csv")
        ind_name = "id"
        tpls = ti.read_triplegs_csv(file, sep=";", index_col=ind_name)
        assert isinstance(tpls, ti.Triplegs)


class TestStaypoints:
    """Test for 'read_staypoints_csv' and 'write_staypoints_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "staypoints.csv")
        mod_file = os.path.join("tests", "data", "staypoints_mod_columns.csv")
        tmp_file = os.path.join("tests", "data", "staypoints_test_1.csv")
        sp = ti.read_staypoints_csv(orig_file, sep=";", tz="utc", index_col="id")
        mod_sp = ti.read_staypoints_csv(mod_file, columns={"User": "user_id"}, sep=";", index_col="id")
        assert mod_sp.equals(sp)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        columns = ["user_id", "started_at", "finished_at", "elevation", "geom"]
        sp.as_staypoints.to_csv(tmp_file, sep=";", columns=columns, date_format=date_format)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_set_crs(self):
        """Test setting the crs when reading."""
        file = os.path.join("tests", "data", "staypoints.csv")
        crs = "EPSG:2056"
        sp = ti.read_staypoints_csv(file, sep=";", tz="utc", index_col="id")
        assert sp.crs is None

        sp = ti.read_staypoints_csv(file, sep=";", tz="utc", index_col="id", crs=crs)
        assert sp.crs == crs

    def test_set_datatime_tz(self):
        """Test setting the timezone infomation when reading."""
        # check if tz is added to the datatime column
        file = os.path.join("tests", "data", "staypoints.csv")
        sp = ti.read_staypoints_csv(file, sep=";", index_col="id")
        assert isinstance(sp["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a timezone will be without storing the timezone
        tmp_file = os.path.join("tests", "data", "staypoints_test_2.csv")
        date_format = "%Y-%m-%d %H:%M:%S"
        sp.as_staypoints.to_csv(tmp_file, sep=";", date_format=date_format)
        sp = ti.read_staypoints_csv(tmp_file, sep=";", index_col="id", tz="utc")

        assert isinstance(sp["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a warning is raised if 'tz' is not provided
        with pytest.warns(UserWarning):
            ti.read_staypoints_csv(tmp_file, sep=";", index_col="id")

        os.remove(tmp_file)

    def test_set_index_warning(self):
        """Test if a warning is raised when not parsing the index_col argument."""
        file = os.path.join("tests", "data", "staypoints.csv")
        with pytest.warns(UserWarning):
            ti.read_staypoints_csv(file, sep=";")

    def test_set_index(self):
        """Test if `index_col` can be set."""
        file = os.path.join("tests", "data", "staypoints.csv")
        ind_name = "id"
        pfs = ti.read_staypoints_csv(file, sep=";", index_col=ind_name)
        assert pfs.index.name == ind_name
        pfs = ti.read_staypoints_csv(file, sep=";", index_col=None)
        assert pfs.index.name is None

    def test_type(self):
        """Test if returned object is Staypoint"""
        file = os.path.join("tests", "data", "staypoints.csv")
        ind_name = "id"
        sp = ti.read_staypoints_csv(file, sep=";", index_col=ind_name)
        assert isinstance(sp, ti.Staypoints)


@pytest.fixture
def example_locations():
    """Locations to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    list_dict = [
        {"user_id": 0, "center": p1},
        {"user_id": 0, "center": p2},
        {"user_id": 1, "center": p3},
    ]
    locs = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    locs.index.name = "id"
    return ti.Locations(locs)


class TestLocations:
    """Test for 'read_locations_csv' and 'write_locations_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "locations.csv")
        mod_file = os.path.join("tests", "data", "locations_mod_columns.csv")
        tmp_file = os.path.join("tests", "data", "locations_test_1.csv")
        mod_locs = ti.read_locations_csv(mod_file, columns={"geom": "center"}, sep=";", index_col="id")
        locs = ti.read_locations_csv(orig_file, sep=";", index_col="id")
        assert mod_locs.equals(locs)
        locs.as_locations.to_csv(tmp_file, sep=";", columns=["user_id", "elevation", "center", "extent"])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_set_crs(self):
        """Test setting the crs when reading."""
        file = os.path.join("tests", "data", "locations.csv")
        crs = "EPSG:2056"
        locs = ti.read_locations_csv(file, sep=";", index_col="id")
        assert locs.crs is None

        locs = ti.read_locations_csv(file, sep=";", index_col="id", crs=crs)
        assert locs.crs == crs

    def test_set_index_warning(self):
        """Test if a warning is raised when not parsing the index_col argument."""
        file = os.path.join("tests", "data", "locations.csv")
        with pytest.warns(UserWarning):
            ti.read_locations_csv(file, sep=";")

    def test_set_index(self):
        """Test if `index_col` can be set."""
        file = os.path.join("tests", "data", "locations.csv")
        ind_name = "id"
        pfs = ti.read_locations_csv(file, sep=";", index_col=ind_name)
        assert pfs.index.name == ind_name
        pfs = ti.read_locations_csv(file, sep=";", index_col=None)
        assert pfs.index.name is None

    def test_without_extent(self, example_locations):
        """Test if without extent column data is correctly write/read."""
        locs = example_locations
        tmp_file = os.path.join("tests", "data", "locations_test.csv")
        locs.as_locations.to_csv(tmp_file, sep=";")
        locs_in = ti.read_locations_csv(tmp_file, sep=";", index_col="id", crs=4326)
        assert_geodataframe_equal(locs, locs_in)
        os.remove(tmp_file)

    def test_type(self):
        """Test if returned object is Locations"""
        file = os.path.join("tests", "data", "locations.csv")
        ind_name = "id"
        sp = ti.read_locations_csv(file, sep=";", index_col=ind_name)
        assert isinstance(sp, ti.Locations)


class TestTrips:
    """Test for 'read_trips_csv' and 'write_trips_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "trips.csv")
        mod_file = os.path.join("tests", "data", "trips_mod_columns.csv")
        tmp_file = os.path.join("tests", "data", "trips_test_1.csv")
        trips = ti.read_trips_csv(orig_file, sep=";", index_col="id")
        column_mapping = {"orig_stp": "origin_staypoint_id", "dest_stp": "destination_staypoint_id"}
        mod_trips = ti.read_trips_csv(mod_file, columns=column_mapping, sep=";", index_col="id")
        mod_trips_wo_geom = pd.DataFrame(mod_trips.drop(columns=["geom"]))
        assert mod_trips_wo_geom.equals(trips)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        columns = ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]
        trips.as_trips.to_csv(tmp_file, sep=";", columns=columns, date_format=date_format)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_set_datatime_tz(self):
        """Test setting the timezone infomation when reading."""
        # check if tz is added to the datatime column
        file = os.path.join("tests", "data", "trips.csv")
        trips = ti.read_trips_csv(file, sep=";", index_col="id")
        assert isinstance(trips["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a timezone will be set without storing the timezone
        tmp_file = os.path.join("tests", "data", "trips_test_2.csv")
        date_format = "%Y-%m-%d %H:%M:%S"
        trips.as_trips.to_csv(tmp_file, sep=";", date_format=date_format)
        trips = ti.read_trips_csv(tmp_file, sep=";", index_col="id", tz="utc")

        assert isinstance(trips["started_at"].dtype, pd.DatetimeTZDtype)

        # check if a warning is raised if 'tz' is not provided
        with pytest.warns(UserWarning):
            ti.read_trips_csv(tmp_file, sep=";", index_col="id")

        os.remove(tmp_file)

    def test_set_index_warning(self):
        """Test if a warning is raised when not parsing the index_col argument."""
        file = os.path.join("tests", "data", "trips.csv")
        with pytest.warns(UserWarning):
            ti.read_trips_csv(file, sep=";")

    def test_set_index(self):
        """Test if `index_col` can be set."""
        file = os.path.join("tests", "data", "trips.csv")
        ind_name = "id"
        gdf = ti.read_trips_csv(file, sep=";", index_col=ind_name)
        assert gdf.index.name == ind_name
        gdf = ti.read_trips_csv(file, sep=";", index_col=None)
        assert gdf.index.name is None

    def test_type(self):
        """Test if returned object is Trips"""
        file = os.path.join("tests", "data", "trips.csv")
        ind_name = "id"
        trips = ti.read_trips_csv(file, sep=";", index_col=ind_name)
        assert isinstance(trips, ti.TripsDataFrame)


@pytest.fixture
def example_tours():
    """Tours to load into the database."""
    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    h = pd.Timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t1 + h, "trips": [0, 1, 2]},
        {"user_id": 0, "started_at": t2, "finished_at": t2 + h, "trips": [2, 3, 4]},
        {"user_id": 1, "started_at": t3, "finished_at": t3 + h, "trips": [4, 5, 6]},
    ]
    tours = pd.DataFrame(data=list_dict)
    tours.index.name = "id"
    return ti.Tours(tours)


class TestTours:
    """Test for 'read_tours_csv' and 'write_tours_csv' functions."""

    def test_from_to_csv(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "geolife_long", "tours.csv")
        tmp_file = os.path.join("tests", "data", "tours_test.csv")
        tours = ti.read_tours_csv(orig_file, index_col="id")

        ti.io.write_tours_csv(tours, tmp_file)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_to_from_csv(self, example_tours):
        """Test writing then reading functionality."""
        tmp_file = os.path.join("tests", "data", "tours_test.csv")
        example_tours.as_tours.to_csv(tmp_file)
        read_tours = ti.read_tours_csv(tmp_file, index_col="id")
        os.remove(tmp_file)
        assert_frame_equal(example_tours, read_tours)

    def test_to_csv_accessor(self):
        """Test basic reading and writing functions."""
        orig_file = os.path.join("tests", "data", "geolife_long", "tours.csv")
        tmp_file = os.path.join("tests", "data", "tours_test.csv")
        tours = ti.read_tours_csv(orig_file, index_col="id")

        tours.as_tours.to_csv(tmp_file)
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    def test_type(self):
        """Test if returned object is Tours"""
        file = os.path.join("tests", "data", "tours.csv")
        ind_name = "id"
        sp = ti.read_tours_csv(file, sep=";", index_col=ind_name)
        assert isinstance(sp, ti.Tours)
