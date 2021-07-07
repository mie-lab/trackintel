from functools import wraps
from inspect import signature

import geopandas as gpd
import pandas as pd
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import create_engine

import trackintel as ti


def _handle_con_string(func):
    """Decorator function to create a `Connection` out of a connection string."""

    @wraps(func)  # copy all metadata
    def wrapper(*args, **kwargs):
        # bind to name for easy access of both kwargs and args
        bound_values = signature(func).bind(*args, **kwargs)
        con = bound_values.arguments["con"]
        # only do something if connection string
        if not isinstance(con, str):
            return func(*args, **kwargs)

        engine = create_engine(con)
        con = engine.connect()

        # overwrite con argument with open connection
        bound_values.arguments["con"] = con
        args = bound_values.args
        kwargs = bound_values.kwargs
        try:
            result = func(*args, **kwargs)
        finally:
            con.close()
        return result

    return wrapper


@_handle_con_string
def read_positionfixes_postgis(sql, con, geom_col="geom", *args, **kwargs):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM positionfixes"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further keyword arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.

    Examples
    --------
    >>> pfs = ti.io.postgis.read_postifionfixes("SELECT * FROM postionfixes", con, geom_col="geom")
    """
    pfs = gpd.GeoDataFrame.from_postgis(sql, con, geom_col, *args, **kwargs)
    return ti.io.read_positionfixes_gpd(pfs, geom=geom_col)


@_handle_con_string
def write_positionfixes_postgis(positionfixes, con, table_name, schema=None, sql_chunksize=None, if_exists="fail"):
    """Stores positionfixes to PostGIS. Usually, this is directly called on a positionfixes
    DataFrame (see example below).

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes to store to the database.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    Examples
    --------
    >>> pfs.as_positionfixes.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_positionfixes_postgis(pfs, conn_string, table_name)
    """
    positionfixes.to_postgis(table_name, con, if_exists=if_exists, schema=schema, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_triplegs_postgis(sql, con, geom_col="geom", *args, **kwargs):
    """Reads triplegs from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM triplegs"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further keyword arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
    """
    tpls = gpd.GeoDataFrame.from_postgis(sql, con, geom_col=geom_col, index_col="id", *args, **kwargs)
    return ti.io.read_triplegs_gpd(tpls, geom=geom_col)


@_handle_con_string
def write_triplegs_postgis(
    triplegs, con, table_name, schema=None, sql_chunksize=None, if_exists="fail", *args, **kwargs
):
    """Stores triplegs to PostGIS. Usually, this is directly called on a triplegs
    DataFrame (see example below).

    Parameters
    ----------
    triplegs : GeoDataFrame
        The triplegs to store to the database.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    Examples
    --------
    >>> tpls.as_triplegs.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_triplegs_postgis(tpls, conn_string, table_name)
    """
    triplegs.to_postgis(table_name, con, if_exists=if_exists, schema=schema, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_staypoints_postgis(sql, con, geom_col="geom", *args, **kwargs):
    """Read staypoints from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM staypoints"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further keyword arguments as available in GeoPanda's GeoDataFrame.from_postgis().


    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
    """
    spts = gpd.GeoDataFrame.from_postgis(sql, con, geom_col=geom_col, index_col="id", *args, **kwargs)

    return ti.io.read_staypoints_gpd(spts, geom=geom_col)


@_handle_con_string
def write_staypoints_postgis(staypoints, con, table_name, schema=None, sql_chunksize=None, if_exists="fail"):
    """Stores staypoints to PostGIS. Usually, this is directly called on a staypoints
    DataFrame (see example below).

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to store to the database.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    Examples
    --------
    >>> spts.as_staypoints.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_staypoints_postgis(spts, conn_string, table_name)
    """

    # todo: Think about a concept for the indices. At the moment, an index
    # column is required when downloading. This means, that the ID column is
    # taken as pandas index. When uploading the default is "no index" and
    # thereby the index column is lost

    # make a copy in order to avoid changing the geometry of the original array
    staypoints.to_postgis(table_name, con, if_exists=if_exists, schema=schema, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_locations_postgis(sql, con, geom_col="geom", *args, **kwargs):
    """Reads locations from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM locations"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table. For the center of the location.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further keyword arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.
    """
    locs = gpd.GeoDataFrame.from_postgis(sql, con, geom_col=geom_col, index_col="id", *args, **kwargs)

    return ti.io.read_locations_gpd(locs, center=geom_col)


@_handle_con_string
def write_locations_postgis(locations, con, table_name, schema=None, sql_chunksize=None, if_exists="fail"):
    """Store locations to PostGIS. Usually, this is directly called on a locations GeoDataFrame (see example below).

    Parameters
    ----------
    locations : GeoDataFrame
        The locations to store to the database.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    Examples
    --------
    >>> locs.as_locations.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_locations_postgis(locs, conn_string, table_name)
    """
    # Assums that "extent" is not geometry column but center is.
    # May build additional check for that.
    if "extent" in locations.columns:
        # geopandas.to_postgis can only handle one geometry column -> do it manually
        if locations.crs is not None:
            srid = locations.crs.to_epsg()
        else:
            srid = -1
        extent_schema = Geometry("POLYGON", srid)
        dtype = {"extent": extent_schema}
        locations = locations.copy()
        locations["extent"] = locations["extent"].apply(lambda x: WKTElement(x.wkt, srid=srid))
    else:
        dtype = None

    locations.to_postgis(
        table_name, con, if_exists=if_exists, schema=schema, index=True, chunksize=sql_chunksize, dtype=dtype
    )


@_handle_con_string
def read_trips_postgis(sql, con, *args, **kwargs):
    """Read trips from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM trips"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further keyword arguments as available in GeoPanda's GeoDataFrame.from_postgis().


    Returns
    -------
    DataFrame
        A DataFrame containing the trips.
    """
    trips = pd.read_sql(sql, con, index_col="id", *args, **kwargs)

    return ti.io.read_trips_gpd(trips)


@_handle_con_string
def write_trips_postgis(trips, con, table_name, schema=None, sql_chunksize=None, if_exists="fail"):
    """Stores trips to PostGIS. Usually, this is directly called on a trips
    DataFrame (see example below).

    Parameters
    ----------
    trips : DataFrame
        The trips to store to the database.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    Examples
    --------
    >>> trips.as_trips.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_trips_postgis(trips, conn_string, table_name)
    """
    trips.to_sql(table_name, con, if_exists=if_exists, schema=schema, index=True, chunksize=sql_chunksize)
