"""A copy of the geopandas postgis test to verify the continuous integration"""
import datetime
import os

import geopandas
import pandas as pd
import pytest
from geopandas import GeoDataFrame, read_file, read_postgis
from geopandas.tests.util import create_postgis, validate_boro_df
from shapely.geometry import LineString
from sqlalchemy import create_engine


@pytest.fixture()
def engine_postgis():
    """
    Initiaties a connection engine to a postGIS database that must already exist.
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


@pytest.fixture
def df_nybb():
    nybb_path = geopandas.datasets.get_path("nybb")
    df = read_file(nybb_path)
    return df


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
    Initiaties a connection to a postGIS database that must already exist.
    See create_postgis for more information.
    """
    psycopg2 = pytest.importorskip("psycopg2")
    from psycopg2 import OperationalError

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    try:
        con = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
    except OperationalError:
        pytest.skip("Cannot connect with postgresql database")

    yield con
    con.close()


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

    tpls = GeoDataFrame(data=list_dict, geometry='geometry', crs='EPSG:4326').set_index('id')

    assert tpls.as_triplegs
    return tpls


class TestIO:
    def test_read_postgis_default(self, connection_postgis, df_nybb):
        con = connection_postgis
        create_postgis(con, df_nybb)

        sql = "SELECT * FROM nybb;"
        df = read_postgis(sql, con)

        validate_boro_df(df)
        # no crs defined on the created geodatabase, and none specified
        # by user; should not be set to 0, as from get_srid failure
        assert df.crs is None

    # def test_postgis_test_that_fails():
    #     pytest.skip("This skip should cause a fail for the postgis test run")


def test_read_write_tripleg_engine(example_triplegs, conn_string_postgis):
    tpls = example_triplegs
    # con_string
    conn_string = conn_string_postgis
    tpls.as_triplegs.to_postgis(conn_string, 'triplegs')

    # engine
    engine = create_engine(conn_string)
    tpls.as_triplegs.to_postgis(engine, 'triplegs')


def test_read_write_tripleg(example_triplegs, conn_string_postgis):
    tpls = example_triplegs
    conn_string = conn_string_postgis
    tpls.as_triplegs.to_postgis(conn_string, 'triplegs')

    # read triplegs from database
    # tpls2 = ti.io.read_triplegs_postgis()

    # compare both

    assert True
