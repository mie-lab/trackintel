from functools import wraps
from inspect import signature

import geopandas as gpd
from geopandas.io.sql import _get_srid_from_crs
from shapely import wkb
import pandas as pd
from geoalchemy2 import Geometry
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
def read_positionfixes_postgis(
    sql,
    con,
    geom_col="geom",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM positionfixes"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    crs : optional
        Coordinate reference system to use for the returned GeoDataFrame

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_positionfixes_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.

    Examples
    --------
    >>> pfs = ti.io.read_positionfixes_postgis("SELECT * FROM positionfixes", con, geom_col="geom")
    >>> pfs = ti.io.read_positionfixes_postgis("SELECT * FROM positionfixes", con, geom_col="geom",
    ...                                        index_col="id", user_id="USER", tracked_at="time")
    """
    pfs = gpd.GeoDataFrame.from_postgis(
        sql,
        con,
        geom_col=geom_col,
        crs=crs,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        params=params,
        chunksize=chunksize,
    )
    return ti.io.read_positionfixes_gpd(pfs, **kwargs)


@_handle_con_string
def write_positionfixes_postgis(
    positionfixes, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    positionfixes.to_postgis(
        name,
        con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
    )


@_handle_con_string
def read_triplegs_postgis(
    sql,
    con,
    geom_col="geom",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Reads triplegs from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM triplegs"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    crs : optional
        Coordinate reference system to use for the returned GeoDataFrame

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_triplegs_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.

    Examples
    --------
    >>> tpls = ti.io.read_triplegs_postgis("SELECT * FROM triplegs", con, geom_col="geom")
    >>> tpls = ti.io.read_triplegs_postgis("SELECT * FROM triplegs", con, geom_col="geom", index_col="id",
    ...                                    started_at="start_time", finished_at="end_time", user_id="USER")
    """
    tpls = gpd.GeoDataFrame.from_postgis(
        sql,
        con,
        geom_col=geom_col,
        crs=crs,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        params=params,
        chunksize=chunksize,
    )
    return ti.io.read_triplegs_gpd(tpls, **kwargs)


@_handle_con_string
def write_triplegs_postgis(
    triplegs, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    triplegs.to_postgis(
        name,
        con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
    )


@_handle_con_string
def read_staypoints_postgis(
    sql,
    con,
    geom_col="geom",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Read staypoints from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM staypoints"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, default 'geom'
        The geometry column of the table.

    crs : optional
        Coordinate reference system to use for the returned GeoDataFrame

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_staypoints_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.


    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.

    Examples
    --------
    >>> sp = ti.io.read_staypoints_postgis("SELECT * FROM staypoints", con, geom_col="geom")
    >>> sp = ti.io.read_staypoints_postgis("SELECT * FROM staypoints", con, geom_col="geom", index_col="id",
    ...                                      started_at="start_time", finished_at="end_time", user_id="USER")
    """
    sp = gpd.GeoDataFrame.from_postgis(
        sql,
        con,
        geom_col=geom_col,
        crs=crs,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        params=params,
        chunksize=chunksize,
    )

    return ti.io.read_staypoints_gpd(sp, **kwargs)


@_handle_con_string
def write_staypoints_postgis(
    staypoints, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    staypoints.to_postgis(
        name,
        con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
    )


@_handle_con_string
def read_locations_postgis(
    sql,
    con,
    center="center",
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Reads locations from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM locations"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    center : str, default 'center'
        The geometry column of the table. For the center of the location.

    crs : optional
        Coordinate reference system to use for the returned GeoDataFrame

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_locations_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.

    Examples
    --------
    >>> locs = ti.io.read_locations_postgis("SELECT * FROM locations", con, center="center")
    >>> locs = ti.io.read_locations_postgis("SELECT * FROM locations", con, center="geom", index_col="id",
    ...                                     user_id="USER", extent="extent")
    )
    """
    locs = gpd.GeoDataFrame.from_postgis(
        sql,
        con,
        geom_col=center,
        crs=crs,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        params=params,
        chunksize=chunksize,
    )

    return ti.io.read_locations_gpd(locs, center=center, **kwargs)


@_handle_con_string
def write_locations_postgis(
    locations, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    # Assums that "extent" is not geometry column but center is.
    # May build additional check for that.
    if "extent" in locations.columns:
        # geopandas.to_postgis can only handle one geometry column -> do it manually
        srid = _get_srid_from_crs(locations)
        extent_schema = Geometry("POLYGON", srid)

        if dtype is None:
            dtype = {"extent": extent_schema}
        else:
            dtype["extent"] = extent_schema
        locations = locations.copy()
        locations["extent"] = locations["extent"].apply(lambda x: wkb.dumps(x, srid=srid, hex=True))

    locations.to_postgis(
        name,
        con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
    )


@_handle_con_string
def read_trips_postgis(
    sql,
    con,
    geom_col=None,
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Read trips from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM trips"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, optional
        The geometry column of the table (if exists). Start and endpoint of the trip.

    crs : optional
        Coordinate reference system if table has geometry.

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_trips_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.


    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the trips.

    Examples
    --------
    >>> trips = ti.io.read_trips_postgis("SELECT * FROM trips", con)
    >>> trips = ti.io.read_trips_postgis("SELECT * FROM trips", con, geom_col="geom", index_col="id",
    ...                                  started_at="start_time", finished_at="end_time", user_id="USER",
    ...                                  origin_staypoint_id="ORIGIN", destination_staypoint_id="DEST")

    """
    if geom_col is None:
        trips = pd.read_sql(
            sql,
            con,
            index_col=index_col,
            coerce_float=coerce_float,
            params=params,
            parse_dates=parse_dates,
            chunksize=chunksize,
        )
    else:
        trips = gpd.GeoDataFrame.from_postgis(
            sql,
            con,
            geom_col=geom_col,
            crs=crs,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )

    return ti.io.read_trips_gpd(trips, **kwargs)


@_handle_con_string
def write_trips_postgis(
    trips, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    if isinstance(trips, gpd.GeoDataFrame):
        trips.to_postgis(
            name,
            con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
        )
    else:  # is DataFrame
        trips.to_sql(
            name,
            con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
        )


@_handle_con_string
def read_tours_postgis(
    sql,
    con,
    geom_col=None,
    crs=None,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
    params=None,
    chunksize=None,
    **kwargs
):
    """Read tours from a PostGIS database.

    Parameters
    ----------
    sql : str
        SQL query e.g. "SELECT * FROM tours"

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    geom_col : str, optional
        The geometry column of the table (if exists).

    crs : optional
        Coordinate reference system if table has geometry.

    index_col : string or list of strings, optional
        Column(s) to set as index(MultiIndex)

    coerce_float : boolean, default True
        Attempt to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets

    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
            strftime compatible in case of parsing string times, or is one of
            (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict
            corresponds to the keyword arguments of
            :func:`pandas.to_datetime`. Especially useful with databases
            without native Datetime support, such as SQLite.

    params : list, tuple or dict, optional, default None
        List of parameters to pass to execute method.

    chunksize : int, default None
        If specified, return an iterator where chunksize is the number
        of rows to include in each chunk.

    **kwargs
        Further keyword arguments as available in trackintels trackintel.io.read_tours_gpd().
        Especially useful to rename column names from the SQL table to trackintel conform column names.
        See second example how to use it in code.

    Returns
    -------
    GeoDataFrame (as trackintel tours)
        A GeoDataFrame containing the tours.

    Examples
    --------
    >>> tours = ti.io.read_tours_postgis("SELECT * FROM tours", con)
    >>> tours = ti.io.read_tours_postgis("SELECT * FROM tours", con, index_col="id", started_at="start_time",
    ...                                  finished_at="end_time", user_id="USER")
    """
    if geom_col is None:
        tours = pd.read_sql(
            sql,
            con,
            index_col=index_col,
            coerce_float=coerce_float,
            params=params,
            parse_dates=parse_dates,
            chunksize=chunksize,
        )
    else:
        tours = gpd.GeoDataFrame.from_postgis(
            sql,
            con,
            geom_col=geom_col,
            crs=crs,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )

    return ti.io.read_tours_gpd(tours, **kwargs)


@_handle_con_string
def write_tours_postgis(
    tours, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
):
    write_trips_postgis(
        tours,
        name=name,
        con=con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
    )


# helper docstring to change __doc__ of all write functions conveniently in one place
__doc = """Stores {long} to PostGIS. Usually, this is directly called on a {long}
    DataFrame (see example below).

    Parameters
    ----------
    {long} : GeoDataFrame (as trackintel {long})
        The {long} to store to the database.

    name : str
        The name of the table to write to.

    con : str, sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Connection string or active connection to PostGIS database.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    if_exists : str, {{'fail', 'replace', 'append'}}, default 'fail'
        How to behave if the table already exists.

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.

    index : bool, default True
        Write DataFrame index as a column. Uses index_label as the column name in the table.

    index_label : str or sequence, default None
        Column label for index column(s). If None is given (default) and index is True, then the index names are used.

    chunksize : int, optional
        How many entries should be written at the same time.

    dtype: dict of column name to SQL type, default None
        Specifying the datatype for columns.
        The keys should be the column names and the values should be the SQLAlchemy types.

    Examples
    --------
    >>> {short}.as_{long}.to_postgis(conn_string, table_name)
    >>> ti.io.postgis.write_{long}_postgis({short}, conn_string, table_name)
"""

write_positionfixes_postgis.__doc__ = __doc.format(long="positionfixes", short="pfs")
write_triplegs_postgis.__doc__ = __doc.format(long="triplegs", short="tpls")
write_staypoints_postgis.__doc__ = __doc.format(long="staypoints", short="sp")
write_locations_postgis.__doc__ = __doc.format(long="locations", short="locs")
write_trips_postgis.__doc__ = __doc.format(long="trips", short="trips")
write_tours_postgis.__doc__ = __doc.format(long="tours", short="tours")
