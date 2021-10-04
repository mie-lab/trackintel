import os

import geopandas as gpd
import pandas as pd
import pytest
import trackintel as ti
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal, assert_index_equal
from shapely.geometry import Point, Polygon, MultiPoint
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
    """Model conform positionfixes to test with."""
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
        """Test renaming of columns."""
        example_positionfixes["additional_col"] = [11, 22, 33]
        pfs = example_positionfixes.copy()
        # create new column mapping and revert it
        columns = {"user_id": "_user_id", "tracked_at": "_tracked_at", "additional_col": "_additional_col"}
        columns_rev = {val: key for key, val in columns.items()}
        # check if columns get renamed correctly
        pfs.rename(columns=columns, inplace=True)
        pfs = _trackintel_model(pfs, columns_rev)
        assert_geodataframe_equal(example_positionfixes, pfs)

    def test_setting_geometry(self, example_positionfixes):
        """Test the setting of the geometry."""
        # create pfs as dataframe
        pfs = pd.DataFrame(example_positionfixes[["user_id", "tracked_at"]], copy=True)
        pfs["geom"] = example_positionfixes.geometry
        # check if geom column gets assigned to geometry
        pfs = _trackintel_model(pfs, geom_col="geom")
        assert_geodataframe_equal(example_positionfixes, pfs)

    def test_set_crs(self, example_positionfixes):
        """Test if crs will be set."""
        pfs = example_positionfixes.copy()
        example_positionfixes.crs = "EPSG:2056"
        # check if the crs is correctly set
        pfs.crs = None
        pfs = _trackintel_model(pfs, crs="EPSG:2056")
        assert_geodataframe_equal(example_positionfixes, pfs)

    def test_already_set_geometry(self, example_positionfixes):
        """Test if default checks if GeoDataFrame already has a geometry."""
        pfs = _trackintel_model(example_positionfixes)
        assert_geodataframe_equal(pfs, example_positionfixes)

    def test_error_no_set_geometry(self, example_positionfixes):
        """Test if AttributeError will be raised if no geom_col is provided and GeoDataFrame has no geometry."""
        pfs = gpd.GeoDataFrame(example_positionfixes[["user_id", "tracked_at"]])
        with pytest.raises(AttributeError):
            _trackintel_model(pfs)

    def test_tz_cols(self, example_positionfixes):
        """Test if columns get casted to datetimes."""
        pfs = example_positionfixes.copy()
        pfs["tracked_at"] = ["1971-01-01 04:00:00", "1971-01-01 05:00:00", "1971-01-02 07:00:00"]
        pfs = _trackintel_model(pfs, tz_cols=["tracked_at"], tz="UTC")
        assert_geodataframe_equal(pfs, example_positionfixes)

    def test_multiple_timezones_in_col(self, example_positionfixes):
        """Test if datetimes in column don't have the same timezone get casted to UTC."""
        example_positionfixes["tracked_at"] = [
            pd.Timestamp("2021-08-01 16:00:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2021-08-01 16:00:00", tz="Asia/Muscat"),
            pd.Timestamp("2021-08-01 16:00:00", tz="Pacific/Niue"),
        ]
        pfs = _trackintel_model(example_positionfixes, tz_cols=["tracked_at"])
        example_positionfixes["tracked_at"] = pd.to_datetime(example_positionfixes["tracked_at"], utc=True)
        assert_geodataframe_equal(pfs, example_positionfixes)


class TestRead_Positionfixes_Gpd:
    """Test `read_positionfixes_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        # read from file and transform to trackintel format
        gdf = gpd.read_file(os.path.join("tests", "data", "positionfixes.geojson"))
        gdf.set_index("id", inplace=True)
        pfs_from_gpd = read_positionfixes_gpd(gdf, user_id="User", geom_col="geometry", crs="EPSG:4326", tz="utc")

        # read from csv file
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs_from_csv = ti.read_positionfixes_csv(pfs_file, sep=";", tz="utc", index_col="id", crs="EPSG:4326")
        pfs_from_csv = pfs_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(pfs_from_gpd, pfs_from_csv, check_exact=False)

    def test_mapper(self, example_positionfixes):
        """Test if mapper argument allows for additional renaming."""
        example_positionfixes["additional_col"] = [11, 22, 33]
        mapper = {"additional_col": "additional_col_renamed"}
        pfs = read_positionfixes_gpd(example_positionfixes, mapper=mapper)
        example_positionfixes.rename(columns=mapper, inplace=True)
        assert_geodataframe_equal(example_positionfixes, pfs)


class TestRead_Triplegs_Gpd:
    """Test `read_triplegs_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        # read from file and transform to trackintel format
        gdf = gpd.read_file(os.path.join("tests", "data", "triplegs.geojson"))
        gdf.set_index("id", inplace=True)
        tpls_from_gpd = read_triplegs_gpd(gdf, user_id="User", geom_col="geometry", crs="EPSG:4326", tz="utc")

        # read from csv file
        tpls_file = os.path.join("tests", "data", "triplegs.csv")
        tpls_from_csv = ti.read_triplegs_csv(tpls_file, sep=";", tz="utc", index_col="id")
        tpls_from_csv = tpls_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(tpls_from_gpd, tpls_from_csv, check_exact=False)

    def test_mapper(self):
        """Test if mapper argument allows for additional renaming."""
        gdf = gpd.read_file(os.path.join("tests", "data", "triplegs.geojson"))
        gdf["additional_col"] = [11, 22]
        gdf.rename(columns={"User": "user_id"}, inplace=True)
        mapper = {"additional_col": "additional_col_renamed"}
        tpls = read_triplegs_gpd(gdf, mapper=mapper, tz="utc")
        gdf.rename(columns=mapper, inplace=True)

        assert_index_equal(tpls.columns, gdf.columns)


class TestRead_Staypoints_Gpd:
    """Test `read_staypoints_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        # read from file and transform to trackintel format
        gdf = gpd.read_file(os.path.join("tests", "data", "staypoints.geojson"))
        gdf.set_index("id", inplace=True)
        sp_from_gpd = read_staypoints_gpd(gdf, "start_time", "end_time", geom_col="geometry", crs="EPSG:4326", tz="utc")

        # read from csv file
        sp_file = os.path.join("tests", "data", "staypoints.csv")
        sp_from_csv = ti.read_staypoints_csv(sp_file, sep=";", tz="utc", index_col="id")
        sp_from_csv = sp_from_csv.rename(columns={"geom": "geometry"})

        assert_frame_equal(sp_from_gpd, sp_from_csv, check_exact=False)

    def test_mapper(self):
        """Test if mapper argument allows for additional renaming."""
        gdf = gpd.read_file(os.path.join("tests", "data", "staypoints.geojson"))
        gdf["additional_col"] = [11, 22]
        gdf.rename(columns={"start_time": "started_at", "end_time": "finished_at"}, inplace=True)
        mapper = {"additional_col": "additional_col_renamed"}
        sp = read_staypoints_gpd(gdf, mapper=mapper, tz="utc")
        gdf.rename(columns=mapper, inplace=True)

        assert_index_equal(gdf.columns, sp.columns)


@pytest.fixture()
def example_locations():
    """Model conform locations to test with."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)

    list_dict = [
        {"user_id": 0, "center": p1},
        {"user_id": 0, "center": p2},
        {"user_id": 1, "center": p2},
    ]
    locs = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    locs.index.name = "id"

    coords = [[8.45, 47.6], [8.45, 47.4], [8.55, 47.4], [8.55, 47.6], [8.45, 47.6]]
    extent = Polygon(coords)
    locs["extent"] = extent  # broadcasting
    locs["extent"] = gpd.GeoSeries(locs["extent"])  # dtype
    assert locs.as_locations
    return locs


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

    def test_extent_col(self, example_locations):
        """Test function with optional geom-column "extent"."""
        locs = example_locations.copy()
        del locs["extent"]
        coords = [[8.45, 47.6], [8.45, 47.4], [8.55, 47.4], [8.55, 47.6], [8.45, 47.6]]
        locs["extent_wrongname"] = Polygon(coords)
        locs = read_locations_gpd(locs, extent="extent_wrongname")
        assert_geodataframe_equal(locs, example_locations)

    def test_mapper(self, example_locations):
        """Test if mapper argument allows for additional renaming."""
        example_locations["additional_col"] = [11, 22, 33]
        mapper = {"additional_col": "additional_col_renamed"}
        locs = read_locations_gpd(example_locations, mapper=mapper)
        example_locations.rename(columns=mapper, inplace=True)
        assert_geodataframe_equal(locs, example_locations)


@pytest.fixture
def example_trips():
    """Model conform trips to test with."""
    start = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    h = pd.Timedelta("1h")

    mp1 = MultiPoint([(0.0, 0.0), (1.0, 1.0)])
    mp2 = MultiPoint([(2.0, 2.0), (3.0, 3.0)])

    list_dict = [
        {"user_id": 0, "origin_staypoint_id": 0, "destination_staypoint_id": 1, "geom": mp1},
        {"user_id": 0, "origin_staypoint_id": 1, "destination_staypoint_id": 2, "geom": mp2},
        {"user_id": 1, "origin_staypoint_id": 0, "destination_staypoint_id": 1, "geom": mp2},
    ]
    for n, d in enumerate(list_dict):
        d["started_at"] = start + 4 * n * h
        d["finished_at"] = d["started_at"] + h
    trips = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:2056")
    trips.index.name = "id"
    assert trips.as_trips
    return trips


class TestRead_Trips_Gpd:
    """Test `read_trips_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        df = pd.read_csv(os.path.join("tests", "data", "trips.csv"), sep=";")
        df.set_index("id", inplace=True)
        trips_from_gpd = read_trips_gpd(df, tz="utc")

        trips_file = os.path.join("tests", "data", "trips.csv")
        trips_from_csv = ti.read_trips_csv(trips_file, sep=";", tz="utc", index_col="id")

        assert_frame_equal(trips_from_gpd, trips_from_csv, check_exact=False)

    def test_with_geometry(self, example_trips):
        """Test if optional geometry gets read."""
        trips = example_trips.copy()
        del trips["geom"]
        mp1 = MultiPoint([(0.0, 0.0), (1.0, 1.0)])
        mp2 = MultiPoint([(2.0, 2.0), (3.0, 3.0)])
        trips["geom"] = [mp1, mp2, mp2]
        trips = read_trips_gpd(trips, geom_col="geom", crs="EPSG:2056", tz="utc")
        example_trips = example_trips[trips.columns]  # copy changed column order
        assert_geodataframe_equal(trips, example_trips)

    def test_without_geometry(self, example_trips):
        """Test if DataFrame without geometry stays the same."""
        columns_without_geom = example_trips.columns.difference(["geom"])
        trips = pd.DataFrame(example_trips[columns_without_geom], copy=True)
        trips_after_function = read_trips_gpd(trips)
        assert_frame_equal(trips, trips_after_function)

    def test_mapper(self, example_trips):
        """Test if mapper argument allows for additional renaming."""
        example_trips["additional_col"] = [11, 22, 33]
        mapper = {"additional_col": "additional_col_renamed"}
        trips = read_trips_gpd(example_trips, mapper=mapper)
        example_trips.rename(columns=mapper, inplace=True)
        assert_geodataframe_equal(trips, example_trips)

    def test_multiple_timezones_in_col(self, example_trips):
        """Test if datetimes in column don't have the same timezone get casted to UTC."""
        example_trips["started_at"] = [
            pd.Timestamp("2021-08-01 16:00:00", tz="Europe/Amsterdam"),
            pd.Timestamp("2021-08-01 16:00:00", tz="Asia/Muscat"),
            pd.Timestamp("2021-08-01 16:00:00", tz="Pacific/Niue"),
        ]
        trips = read_trips_gpd(example_trips)
        example_trips["started_at"] = pd.to_datetime(example_trips["started_at"], utc=True)
        assert_geodataframe_equal(example_trips, trips)


@pytest.fixture
def example_tours():
    """Model conform tours to test with."""
    start = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    h = pd.Timedelta("1h")

    list_dict = [
        {"user_id": 0, "started_at": start + 0 * h, "finished_at": start + 1 * h},
        {"user_id": 0, "started_at": start + 2 * h, "finished_at": start + 3 * h},
        {"user_id": 1, "started_at": start + 4 * h, "finished_at": start + 5 * h},
    ]
    tours = pd.DataFrame(data=list_dict)
    tours.index.name = "id"
    tours.as_tours
    return tours


class TestRead_Tours_Gpd:
    """Test `read_trips_gpd()` function."""

    def test_csv(self):
        """Test if the results of reading from gpd and csv agrees."""
        tours_file = os.path.join("tests", "data", "tours.csv")
        df = pd.read_csv(tours_file, sep=";")
        df.set_index("id", inplace=True)
        tours_from_gpd = read_tours_gpd(df, tz="utc")

        tours_from_csv = ti.read_tours_csv(tours_file, sep=";", tz="utc", index_col="id")

        assert_frame_equal(tours_from_gpd, tours_from_csv, check_exact=False)

    def test_mapper(self, example_tours):
        example_tours["additional_col"] = [11, 22, 33]
        mapper = {"additional_col": "additional_col_renamed"}
        tours = read_tours_gpd(example_tours, mapper=mapper)
        example_tours.rename(columns=mapper, inplace=True)
        assert_frame_equal(tours, example_tours)
