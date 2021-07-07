from functools import wraps
from inspect import signature

import geopandas as gpd
import pandas as pd
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import create_engine


def _handle_con_string(func):
    """Handle connection string input."""

    @wraps(func)  # copy all metadata
    def wrapper(*args, **kwargs):
        # bind to name for easy access of both kwargs and args
        bound_values = signature(func).bind(*args, **kwargs)
        con = bound_values.arguments["con"]
        # only do something if string
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
def read_positionfixes_postgis(con, table_name, geom_col="geom", *args, **kwargs):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The table from PostGIS database to read the positionfixes from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    **kwargs
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, con, geom_col=geom_col, *args, **kwargs)
    # assert validity of positionfixes
    pfs.as_positionfixes
    return pfs


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
    """
    positionfixes.to_postgis(table_name, con, if_exists=if_exists, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_triplegs_postgis(con, table_name, geom_col="geom", *args, **kwargs):
    """Reads triplegs from a PostGIS database.

    Parameters
    ----------
    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The table to read the triplegs from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
    """
    pfs = gpd.GeoDataFrame.from_postgis(
        "SELECT * FROM %s" % table_name, con, geom_col=geom_col, index_col="id", *args, **kwargs
    )
    # assert validity of triplegs
    pfs.as_triplegs
    return pfs


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
    """
    triplegs.to_postgis(table_name, con, if_exists=if_exists, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_staypoints_postgis(con, table_name, geom_col="geom", *args, **kwargs):
    """Read staypoints from a PostGIS database.

    Parameters
    ----------
    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The table to read the staypoints from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
    """
    stps = gpd.GeoDataFrame.from_postgis(
        "SELECT * FROM %s" % table_name, con, geom_col=geom_col, index_col="id", *args, **kwargs
    )

    # assert validity of staypoints
    stps.as_staypoints
    return stps


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
    """

    # todo: Think about a concept for the indices. At the moment, an index
    # column is required when downloading. This means, that the ID column is
    # taken as pandas index. When uploading the default is "no index" and
    # thereby the index column is lost

    # make a copy in order to avoid changing the geometry of the original array
    staypoints.to_postgis(table_name, con, if_exists=if_exists, index=True, chunksize=sql_chunksize)


@_handle_con_string
def read_locations_postgis(con, table_name, geom_col="geom", *args, **kwargs):
    """Reads locations from a PostGIS database.

    Parameters
    ----------
    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The table to read the locations from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.
    """
    locs = gpd.GeoDataFrame.from_postgis(
        "SELECT * FROM %s" % table_name, con, geom_col=geom_col, index_col="id", *args, **kwargs
    )

    # assert validity of locations
    locs.as_locations
    return locs


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
    """

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

    locations.to_postgis(table_name, con, if_exists=if_exists, index=True, chunksize=sql_chunksize, dtype=dtype)


@_handle_con_string
def read_trips_postgis(con, table_name, *args, **kwargs):
    """Read trips from a PostGIS database.

    Parameters
    ----------
    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    table_name : str
        The table to read the trips from.

    Returns
    -------
    DataFrame
        A DataFrame containing the trips.
    """
    trips = pd.read_sql("SELECT * FROM %s" % table_name, con, index_col="id", *args, **kwargs)

    # assert validity of trips
    trips.as_trips
    return trips


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
    """

    # make a copy in order to avoid changing the geometry of the original array
    trips.to_sql(table_name, con, if_exists=if_exists, index=True, chunksize=sql_chunksize)
