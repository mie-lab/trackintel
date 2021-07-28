"""A copy of the geopandas postgis test to verify the continuous integration"""
import datetime
import os

import geopandas as gpd
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import LineString, Point, Polygon
import sqlalchemy
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
            # https://stackoverflow.com/questions/61081102/psycopg2-connect-gives-operational-error-unsupported-frontend-protocol
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
        {"user_id": 0, "tracked_at": t1, "geom": p1},
        {"user_id": 0, "tracked_at": t2, "geom": p2},
        {"user_id": 1, "tracked_at": t3, "geom": p3},
    ]
    pfs = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
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
        {"user_id": 0, "started_at": t1, "finished_at": t2, "geom": p1},
        {"user_id": 0, "started_at": t2, "finished_at": t3, "geom": p2},
        {"user_id": 1, "started_at": t3, "finished_at": t3 + one_hour, "geom": p3},
    ]
    stps = gpd.GeoDataFrame(data=list_dict, geometry="geom", crs="EPSG:4326")
    stps.index.name = "id"
    assert stps.as_staypoints
    return stps


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
    assert locs.as_locations
    return locs


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
    trips = pd.DataFrame(data=list_dict)
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


def get_table_schema(con, table):
    """Get Schema of an SQL table (column names, datatypes)"""
    # https://stackoverflow.com/questions/20194806/how-to-get-a-list-column-names-and-datatypes-of-a-table-in-postgresql
    query = f"""
    SELECT
        a.attname as "Column",
        pg_catalog.format_type(a.atttypid, a.atttypmod) as "Datatype"
    FROM
        pg_catalog.pg_attribute a
    WHERE
        a.attnum > 0
        AND NOT a.attisdropped
        AND a.attrelid = (
            SELECT c.oid
            FROM pg_catalog.pg_class c
                LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname ~ '^{table}$'
                AND pg_catalog.pg_table_is_visible(c.oid)
        );"""
    cur = con.cursor()
    cur.execute(query)
    schema = cur.fetchall()
    column, datatype = map(list, zip(*schema))
    return column, datatype


def get_tuple_count(con, table):
    """Return the count of entries in sql table."""
    query = f"""
    SELECT COUNT(*)
    FROM {table}
    """
    cur = con.cursor()
    cur.execute(query)
    count = cur.fetchall()
    return count[0][0]


def _get_srid(gdf):
    """Extract srid from gdf and default to -1 if there isn't one.

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    int
    """
    if gdf.crs is not None:
        return gdf.crs.to_epsg()
    return -1


class TestPositionfixes:
    def test_write(self, example_positionfixes, conn_postgis):
        """Test if write of positionfixes create correct schema in database."""
        pfs = example_positionfixes.copy()
        conn_string, conn = conn_postgis
        table = "positionfixes"
        try:
            pfs.as_positionfixes.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = pfs.columns.tolist() + [pfs.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
            srid = _get_srid(pfs)
            geom_schema = f"geometry(Point,{srid})"
            assert geom_schema in dtypes
        finally:
            del_table(conn, table)

    def test_read(self, example_positionfixes, conn_postgis):
        """Test if positionfixes written to and read back from database are the same."""
        pfs = example_positionfixes.copy()
        conn_string, conn = conn_postgis
        table = "positionfixes"
        sql = f"SELECT * FROM {table}"
        geom_col = pfs.geometry.name

        try:
            pfs.as_positionfixes.to_postgis(table, conn_string)
            pfs_db = ti.io.read_positionfixes_postgis(sql, conn_string, geom_col)
            pfs_db = pfs_db.set_index("id")
            print(pfs_db)
            assert_geodataframe_equal(pfs, pfs_db)
        finally:
            del_table(conn, table)

    def test_no_crs(self, example_positionfixes, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        pfs = example_positionfixes.copy()
        conn_string, conn = conn_postgis
        table = "positionfixes"
        sql = f"SELECT * FROM {table}"
        geom_col = pfs.geometry.name
        pfs.crs = None
        try:
            pfs.as_positionfixes.to_postgis(table, conn_string)
            pfs_db = ti.io.read_positionfixes_postgis(sql, conn_string, geom_col)
            pfs_db = pfs_db.set_index("id")
            assert_geodataframe_equal(pfs, pfs_db)
        finally:
            del_table(conn, table)


class TestTriplegs:
    def test_write(self, example_triplegs, conn_postgis):
        """Test if write of triplegs create correct schema in database."""
        tpls = example_triplegs.copy()
        conn_string, conn = conn_postgis
        table = "triplegs"
        try:
            tpls.as_triplegs.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = tpls.columns.tolist() + [tpls.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
            srid = _get_srid(tpls)
            geom_schema = f"geometry(LineString,{srid})"
            assert geom_schema in dtypes
        finally:
            del_table(conn, table)

    def test_read(self, example_triplegs, conn_postgis):
        """Test if triplegs written to and read back from database are the same."""
        tpls = example_triplegs.copy()
        conn_string, conn = conn_postgis
        table = "triplegs"
        sql = f"SELECT * FROM {table}"
        geom_col = tpls.geometry.name

        try:
            tpls.as_triplegs.to_postgis(table, conn_string)
            tpls_db = ti.io.read_triplegs_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(tpls, tpls_db)
        finally:
            del_table(conn, table)

    def test_no_crs(self, example_triplegs, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        tpls = example_triplegs.copy()
        conn_string, conn = conn_postgis
        table = "triplegs"
        sql = f"SELECT * FROM {table}"
        geom_col = tpls.geometry.name
        tpls.crs = None

        try:
            tpls.as_triplegs.to_postgis(table, conn_string)
            tpls_db = ti.io.read_triplegs_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(tpls, tpls_db)
        finally:
            del_table(conn, table)


class TestStaypoints:
    def test_write(self, example_staypoints, conn_postgis):
        """Test if write of staypoints create correct schema in database."""
        spts = example_staypoints.copy()
        conn_string, conn = conn_postgis
        table = "staypoints"
        try:
            spts.as_staypoints.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = spts.columns.tolist() + [spts.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
            srid = _get_srid(spts)
            geom_schema = f"geometry(Point,{srid})"
            assert geom_schema in dtypes
        finally:
            del_table(conn, table)

    def test_read(self, example_staypoints, conn_postgis):
        """Test if staypoints written to and read back from database are the same."""
        spts = example_staypoints.copy()
        conn_string, conn = conn_postgis
        table = "staypoints"
        sql = f"SELECT * FROM {table}"
        geom_col = spts.geometry.name

        try:
            spts.as_staypoints.to_postgis(table, conn_string)
            spts_db = ti.io.read_staypoints_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(spts, spts_db)
        finally:
            del_table(conn, table)

    def test_no_crs(self, example_staypoints, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        spts = example_staypoints.copy()
        conn_string, conn = conn_postgis
        table = "staypoints"
        sql = f"SELECT * FROM {table}"
        geom_col = example_staypoints.geometry.name
        spts.crs = None
        try:
            spts.as_staypoints.to_postgis(table, conn_string)
            spts_db = ti.io.read_staypoints_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(spts, spts_db)
        finally:
            del_table(conn, table)


class TestLocations:
    def test_write(self, example_locations, conn_postgis):
        """Test if write of locations create correct schema in database."""
        locs = example_locations.copy()
        conn_string, conn = conn_postgis
        table = "locations"
        try:
            locs.as_locations.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = locs.columns.tolist() + [locs.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
            srid = _get_srid(locs)
            geom_schema = f"geometry(Point,{srid})"
            assert geom_schema in dtypes
        finally:
            del_table(conn, table)

    def test_read(self, example_locations, conn_postgis):
        """Test if locations written to and read back from database are the same."""
        locs = example_locations.copy()
        conn_string, conn = conn_postgis
        table = "locations"
        sql = f"SELECT * FROM {table}"
        geom_col = locs.geometry.name

        try:
            locs.as_locations.to_postgis(table, conn_string)
            locs_db = ti.io.read_locations_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(locs, locs_db)
        finally:
            del_table(conn, table)

    def test_no_crs(self, example_locations, conn_postgis):
        """Test if writing reading to postgis also works correctly without CRS."""
        locs = example_locations.copy()
        conn_string, conn = conn_postgis
        table = "locations"
        sql = f"SELECT * FROM {table}"
        geom_col = locs.geometry.name
        locs.crs = None
        try:
            locs.as_locations.to_postgis(table, conn_string)
            locs_db = ti.io.read_locations_postgis(sql, conn_string, geom_col)
            assert_geodataframe_equal(locs, locs_db)
        finally:
            del_table(conn, table)

    def test_write_extent(self, example_locations, conn_postgis):
        """Test if extent geometry is handled correctly."""
        conn_string, conn = conn_postgis
        table = "locations"
        coords = [[8.45, 47.6], [8.45, 47.4], [8.55, 47.4], [8.55, 47.6], [8.45, 47.6]]
        extent = Polygon(coords)
        example_locations["extent"] = extent  # broadcasting
        example_locations["extent"] = gpd.GeoSeries(example_locations["extent"])  # dtype
        try:
            example_locations.as_locations.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = example_locations.columns.tolist() + [example_locations.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
            srid = _get_srid(example_locations)
            geom_schema_center = f"geometry(Point,{srid})"
            geom_schema_extent = f"geometry(Polygon,{srid})"
            assert geom_schema_center in dtypes
            assert geom_schema_extent in dtypes
        finally:
            del_table(conn, table)


class TestTrips:
    def test_write(self, example_trips, conn_postgis):
        """Test if write of locations create correct schema in database."""
        trips = example_trips.copy()
        conn_string, conn = conn_postgis
        table = "trips"
        try:
            trips.as_trips.to_postgis(table, conn_string)
            columns_db, dtypes = get_table_schema(conn, table)
            columns = trips.columns.tolist() + [trips.index.name]
            assert len(columns_db) == len(columns)
            assert set(columns_db) == set(columns)
        finally:
            del_table(conn, table)

    def test_read(self, example_trips, conn_postgis):
        """Test if trips written to and read back from database are the same."""
        trips = example_trips.copy()
        conn_string, conn = conn_postgis
        table = "trips"
        sql = f"SELECT * FROM {table}"

        try:
            trips.as_trips.to_postgis(table, conn_string)
            trips_db = ti.io.read_trips_postgis(sql, conn_string)
            assert_frame_equal(trips, trips_db)
        finally:
            del_table(conn, table)


class TestGetSrid:
    def test_srid(self, example_positionfixes):
        """Test if `_get_srid` returns the correct srid."""
        gdf = example_positionfixes.copy()
        gdf.crs = None
        assert _get_srid(gdf) == -1
        srid = 3857
        gdf.set_crs(f"epsg:{srid}", inplace=True)
        assert _get_srid(gdf) == srid


class Test_Handle_Con_String:
    def test_conn_string(self, conn_postgis):
        """Test if decorator opens a connection with connection string and closes it."""
        conn_string, _ = conn_postgis

        @ti.io.postgis._handle_con_string
        def wrapped(con):
            assert isinstance(con, sqlalchemy.engine.Connection)
            assert not con.closed
            return con

        assert wrapped(conn_string).closed

    def test_conn(self, conn_postgis):
        """Test handeling of connection input"""
        _, conn = conn_postgis

        @ti.io.postgis._handle_con_string
        def wrapped(con):
            assert con is conn

        wrapped(conn)
