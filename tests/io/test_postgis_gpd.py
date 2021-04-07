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
def engine_postgis():
    """
    Initiates a connection engine to a postGIS database that must already exist.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from sqlalchemy.engine.url import URL

    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    dbname = "test_geopandas"

    try:
        con = sqlalchemy.create_engine(
            URL(
                drivername="postgresql+psycopg2",
                username=user,
                database=dbname,
                password=password,
                host=host,
                port=port,
            )
        )
        con.begin()
    except Exception:
        pytest.skip("Cannot connect with postgresql database")

    yield con
    con.dispose()


@pytest.fixture()
def conn_string_postgis():
    """
    Returns a database connection in the format ``postgresql://username:password@host:socket/database``
    """

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")

    conn_string = "postgresql://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
    return conn_string


@pytest.fixture()
def connection_postgis():
    """
    Initiates a connection to a postGIS database that must already exist.
    See create_postgis for more information.
    """
    psycopg2 = pytest.importorskip("psycopg2")

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    try:
        con = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
    except psycopg2.OperationalError:
        try:
            # psycopg2.connect gives operational error due to unsupported frontend protocol in conda environment
            con = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port, sslmode='disable')
        except psycopg2.OperationalError:
            pytest.skip("Cannot connect with postgresql database")

    yield con
    con.close()


@pytest.fixture
def example_positionfixes():
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp('1971-01-01 00:00:00', tz='utc')
    t2 = pd.Timestamp('1971-01-01 05:00:00', tz='utc')
    t3 = pd.Timestamp('1971-01-02 07:00:00', tz='utc')

    list_dict = [{'user_id': 0, 'tracked_at': t1, 'geometry': p1},
                 {'user_id': 0, 'tracked_at': t2, 'geometry': p2},
                 {'user_id': 1, 'tracked_at': t3, 'geometry': p3}]
    pfs = gpd.GeoDataFrame(data=list_dict, geometry='geometry', crs='EPSG:4326')
    pfs.index.name = 'id'
    assert pfs.as_positionfixes
    return pfs


@pytest.fixture
def example_staypoints():
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    t1 = pd.Timestamp('1971-01-01 00:00:00', tz='utc')
    t2 = pd.Timestamp('1971-01-01 05:00:00', tz='utc')
    t3 = pd.Timestamp('1971-01-02 07:00:00', tz='utc')
    one_hour = datetime.timedelta(hours=1)

    list_dict = [{'user_id': 0, 'started_at': t1, 'finished_at': t2, 'geometry': p1},
                 {'user_id': 0, 'started_at': t2, 'finished_at': t3, 'geometry': p2},
                 {'user_id': 1, 'started_at': t3, 'finished_at': t3 + one_hour, 'geometry': p3}]
    spts = gpd.GeoDataFrame(data=list_dict, geometry='geometry', crs='EPSG:4326')
    spts.index.name = 'id'
    assert spts.as_staypoints
    return spts


@pytest.fixture
def example_triplegs():
    # three linestring geometries that are only slightly different (last coordinate)
    g1 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.4)])
    g2 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.5)])
    g3 = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.6)])

    t1 = pd.Timestamp('1971-01-01 00:00:00', tz='utc')
    t2 = pd.Timestamp('1971-01-01 05:00:00', tz='utc')
    t3 = pd.Timestamp('1971-01-02 07:00:00', tz='utc')
    one_hour = datetime.timedelta(hours=1)

    list_dict = [{'id': 0, 'user_id': 0, 'started_at': t1, 'finished_at': t2, 'geometry': g1},
                 {'id': 1, 'user_id': 0, 'started_at': t2, 'finished_at': t3, 'geometry': g2},
                 {'id': 2, 'user_id': 1, 'started_at': t3, 'finished_at': t3 + one_hour, 'geometry': g3}]

    tpls = gpd.GeoDataFrame(data=list_dict, geometry='geometry', crs='EPSG:4326')
    tpls.set_index('id', inplace=True)

    assert tpls.as_triplegs
    return tpls


@pytest.fixture
def example_locations():
    p1 = Point(8.5067847, 47.4)
    p2 = Point(8.5067847, 47.5)
    p3 = Point(8.5067847, 47.6)

    list_dict = [{'user_id': 0, 'center': p1},
                 {'user_id': 0, 'center': p2},
                 {'user_id': 1, 'center': p3}]
    spts = gpd.GeoDataFrame(data=list_dict, geometry='center', crs='EPSG:4326')
    spts.index.name = 'id'
    assert spts.as_locations
    return spts


@pytest.fixture
def example_trips():
    t1 = pd.Timestamp('1971-01-01 00:00:00', tz='utc')
    t2 = pd.Timestamp('1971-01-01 05:00:00', tz='utc')
    t3 = pd.Timestamp('1971-01-02 07:00:00', tz='utc')
    h = datetime.timedelta(hours=1)

    list_dict = [
        {'user_id': 0, 'started_at': t1, 'finished_at': t2, 'origin_staypoint_id': 0, 'destination_staypoint_id': 1},
        {'user_id': 0, 'started_at': t2, 'finished_at': t3, 'origin_staypoint_id': 1, 'destination_staypoint_id': 2},
        {'user_id': 1, 'started_at': t3, 'finished_at': t3+h, 'origin_staypoint_id': 0, 'destination_staypoint_id': 1}
    ]
    trips = gpd.GeoDataFrame(data=list_dict)
    trips.index.name = 'id'
    assert trips.as_trips
    return trips


def del_table(con, table):
    try:
        cursor = con.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        cursor.close()
        con.commit()


class TestPositionfixes:
    def test_read_write_positionfixes(self, example_positionfixes, conn_string_postgis, connection_postgis):
        pfs = example_positionfixes
        cs = conn_string_postgis + "?sslmode=disable"  # just for me tho
        table = 'positionfixes'
        geom_col = pfs.geometry.name

        try:
            pfs.as_positionfixes.to_postgis(cs, table)
            pfs_db = ti.io.read_positionfixes_postgis(cs, table, geom_col)
            pfs_db = pfs_db.set_index('id')
            assert_geodataframe_equal(pfs, pfs_db)
        finally:
            del_table(connection_postgis, table)
        pass


class TestStaypoints:
    def test_read_write_staypoints(self, example_staypoints, conn_string_postgis, connection_postgis):
        spts = example_staypoints
        cs = conn_string_postgis + "?sslmode=disable"
        table = "staypoints"
        geom_col = spts.geometry.name

        try:
            spts.as_staypoints.to_postgis(cs, table)
            spts_db = ti.io.read_staypoints_postgis(cs, table, geom_col)
            assert_geodataframe_equal(spts, spts_db)
        finally:
            del_table(connection_postgis, table)


class TestTriplegs:
    def test_read_write_triplegs(self, example_triplegs, conn_string_postgis, connection_postgis):
        tpls = example_triplegs
        cs = conn_string_postgis + "?sslmode=disable"
        table = "triplegs"
        geom_col = tpls.geometry.name

        try:
            tpls.as_triplegs.to_postgis(cs, table)
            tpls_db = ti.io.read_triplegs_postgis(cs, table, geom_col)
            assert_geodataframe_equal(tpls, tpls_db)
        finally:
            del_table(connection_postgis, table)


class TestLocations:
    def test_read_write_locations(self, example_locations, conn_string_postgis, connection_postgis):
        locs = example_locations
        cs = conn_string_postgis + "?sslmode=disable"
        table = "locations"
        geom_col = locs.geometry.name

        try:
            locs.as_locations.to_postgis(cs, table)
            locs_db = ti.io.read_locations_postgis(cs, table, geom_col)
            assert_geodataframe_equal(locs, locs_db)
        finally:
            del_table(connection_postgis, table)


class TestTrips:
    def test_read_write_trips(self, example_trips, conn_string_postgis, connection_postgis):
        trips = example_trips
        cs = conn_string_postgis + "?sslmode=disable"
        table = "trips"

        try:
            trips.as_trips.to_postgis(cs, table)
            trips_db = ti.io.read_trips_postgis(cs, table)
            assert_frame_equal(trips, trips_db)
        finally:
            del_table(connection_postgis, table)
