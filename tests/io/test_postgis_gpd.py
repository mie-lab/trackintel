"""A copy of the geopandas postgis test to verify the continuous integration"""
import os

import pandas as pd

import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis

from geopandas.io.sql import _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest

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