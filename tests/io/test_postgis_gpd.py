"""A copy of the geopandas postgis test to verify the continuous integration"""
import datetime
import os

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point
import trackintel as ti


@pytest.fixture()
def conn_postgis():
    """
    Initiates a connection to a postGIS database that must already exist.

    Yields
    -------
    conn_string, con
    """
    psycopg2 = pytest.importorskip("psycopg2")

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        con = psycopg2.connect(conn_string)
    except psycopg2.OperationalError:
        try:
            # psycopg2.connect may gives operational error due to
            # unsupported frontend protocol in conda environment.
            conn_string = conn_string + "?sslmode=disable"
            con = psycopg2.connect(conn_string)
        except psycopg2.OperationalError:
            pytest.skip("Cannot connect with postgresql database")

    yield conn_string, con
    con.close()


@pytest.fixture
def example_positionfixes():
    """Positionfixes to load into the database."""
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
    assert pfs.as_positionfixes
    return pfs


@pytest.fixture
def example_staypoints():
    """Staypoints to load into the database."""
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    one_hour = datetime.timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t2, "geometry": p1},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "geometry": p2},
        {"user_id": 1, "started_at": t3, "finished_at": t3 + one_hour, "geometry": p3},
    ]
    spts = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    spts.index.name = "id"
    assert spts.as_staypoints
    return spts


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
        {"id": 0, "user_id": 0, "started_at": t1, "finished_at": t2, "geometry": g1},
        {"id": 1, "user_id": 0, "started_at": t2, "finished_at": t3, "geometry": g2},
        {"id": 2, "user_id": 1, "started_at": t3, "finished_at": t3 + one_hour, "geometry": g3},
    ]

    tpls = gpd.GeoDataFrame(data=list_dict, geometry="geometry", crs="EPSG:4326")
    tpls.set_index("id", inplace=True)

    assert tpls.as_triplegs
    return tpls


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
    spts = gpd.GeoDataFrame(data=list_dict, geometry="center", crs="EPSG:4326")
    spts.index.name = "id"
    assert spts.as_locations
    return spts


@pytest.fixture
def example_trips():
    """Trips to load into the database."""
    t1 = pd.Timestamp("1971-01-01 00:00:00", tz="utc")
    t2 = pd.Timestamp("1971-01-01 05:00:00", tz="utc")
    t3 = pd.Timestamp("1971-01-02 07:00:00", tz="utc")
    h = datetime.timedelta(hours=1)

    list_dict = [
        {"user_id": 0, "started_at": t1, "finished_at": t2, "origin_staypoint_id": 0, "destination_staypoint_id": 1},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "origin_staypoint_id": 1, "destination_staypoint_id": 2},
        {
            "user_id": 1,
            "started_at": t3,
            "finished_at": t3 + h,
            "origin_staypoint_id": 0,
            "destination_staypoint_id": 1,
        },
    ]
    trips = gpd.GeoDataFrame(data=list_dict)
    trips.index.name = "id"
    assert trips.as_trips
    return trips


def del_table(con, table):
    """Delete table in con."""
    try:
        cursor = con.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        cursor.close()
        con.commit()


class TestPositionfixes:
    def test_io_positionfixes(self, example_positionfixes, conn_postgis):
        """Test if positionfixes written to and read back from database are the same."""
        pfs = example_positionfixes.copy()
        conn_string, conn = conn_postgis
        table = "positionfixes"
        geom_col = pfs.geometry.name

        try:
            pfs.as_positionfixes.to_postgis(conn_string, table)
            pfs_db = ti.io.read_positionfixes_postgis(conn_string, table, geom_col)
            pfs_db = pfs_db.set_index("id")
            assert_geodataframe_equal(pfs, pfs_db)
        finally:
            del_table(conn, table)
        pass

    def test_no_crs_setting(self, example_positionfixes, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        pfs = example_positionfixes.copy()
        conn_string, conn = conn_postgis
        table = "positionfixes"
        geom_col = pfs.geometry.name
        pfs.crs = None
        try:
            pfs.as_positionfixes.to_postgis(conn_string, table)
            pfs_db = ti.io.read_positionfixes_postgis(conn_string, table, geom_col)
            pfs_db = pfs_db.set_index("id")
            assert_geodataframe_equal(pfs, pfs_db)
        finally:
            del_table(conn, table)
        pass


class TestStaypoints:
    def test_io_staypoints(self, example_staypoints, conn_postgis):
        """Test if staypoints written to and read back from database are the same."""
        spts = example_staypoints.copy()
        conn_string, conn = conn_postgis
        table = "staypoints"
        geom_col = spts.geometry.name

        try:
            spts.as_staypoints.to_postgis(conn_string, table)
            spts_db = ti.io.read_staypoints_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(spts, spts_db)
        finally:
            del_table(conn, table)

    def test_no_crs_setting(self, example_staypoints, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        spts = example_staypoints.copy()
        conn_string, conn = conn_postgis
        table = "staypoints"
        geom_col = example_staypoints.geometry.name
        spts.crs = None
        try:
            spts.as_staypoints.to_postgis(conn_string, table)
            spts_db = ti.io.read_staypoints_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(spts, spts_db)
        finally:
            del_table(conn, table)


class TestTriplegs:
    def test_io_triplegs(self, example_triplegs, conn_postgis):
        """Test if triplegs written to and read back from database are the same."""
        tpls = example_triplegs.copy()
        conn_string, conn = conn_postgis
        table = "triplegs"
        geom_col = tpls.geometry.name

        try:
            tpls.as_triplegs.to_postgis(conn_string, table)
            tpls_db = ti.io.read_triplegs_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(tpls, tpls_db)
        finally:
            del_table(conn, table)

    def test_no_crs_setting(self, example_triplegs, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        tpls = example_triplegs.copy()
        conn_string, conn = conn_postgis
        table = "triplegs"
        geom_col = tpls.geometry.name
        tpls.crs = None

        try:
            tpls.as_triplegs.to_postgis(conn_string, table)
            tpls_db = ti.io.read_triplegs_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(tpls, tpls_db)
        finally:
            del_table(conn, table)


class TestLocations:
    def test_io_locations(self, example_locations, conn_postgis):
        """Test if locations written to and read back from database are the same."""
        locs = example_locations.copy()
        conn_string, conn = conn_postgis
        table = "locations"
        geom_col = locs.geometry.name

        try:
            locs.as_locations.to_postgis(conn_string, table)
            locs_db = ti.io.read_locations_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(locs, locs_db)
        finally:
            del_table(conn, table)

    def test_no_crs_setting(self, example_locations, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        locs = example_locations.copy()
        conn_string, conn = conn_postgis
        table = "locations"
        geom_col = locs.geometry.name
        locs.crs = None
        try:
            locs.as_locations.to_postgis(conn_string, table)
            locs_db = ti.io.read_locations_postgis(conn_string, table, geom_col)
            assert_geodataframe_equal(locs, locs_db)
        finally:
            del_table(conn, table)


class TestTrips:
    def test_io_trips(self, example_trips, conn_postgis):
        """Test if trips written to and read back from database are the same."""
        trips = example_trips.copy()
        conn_string, conn = conn_postgis
        table = "trips"

        try:
            trips.as_trips.to_postgis(conn_string, table)
            trips_db = ti.io.read_trips_postgis(conn_string, table)
            assert_frame_equal(trips, trips_db)
        finally:
            del_table(conn, table)
