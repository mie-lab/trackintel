import geopandas as gpd
import pandas as pd
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import create_engine
import warnings


def read_positionfixes_postgis(conn_string, table_name, geom_col="geom", *args, **kwargs):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The table to read the positionfixes from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    *args
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    **kwargs
        Further arguments as available in GeoPanda's GeoDataFrame.from_postgis().

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, geom_col=geom_col, *args, **kwargs)
    finally:
        conn.close()
    assert pfs.as_positionfixes
    return pfs


def write_positionfixes_postgis(
    positionfixes, conn_string, table_name, schema=None, sql_chunksize=None, if_exists="replace"
):
    """Stores positionfixes to PostGIS. Usually, this is directly called on a positionfixes
    DataFrame (see example below).

    **Attention!** This replaces the table if it already exists!

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'replace'
        What should happen if the table already exists.

    Examples
    --------
    >>> df.as_positionfixes.to_postgis(conn_string, table_name)
    """
    # make a copy in order to avoid changing the geometry of the original array
    positionfixes_postgis = positionfixes.copy()
    if positionfixes_postgis.crs is not None:
        srid = positionfixes_postgis.crs.to_epsg()
    else:
        srid = -1
    geom_schema = Geometry("Point", srid)

    geom_col = positionfixes_postgis.geometry.name
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry column does not contain geometry.", UserWarning)
        positionfixes_postgis[geom_col] = positionfixes[geom_col].apply(lambda x: WKTElement(x.wkt, srid=srid))

    if "id" not in positionfixes_postgis.columns:
        positionfixes_postgis["id"] = positionfixes_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        positionfixes_postgis.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            dtype={geom_col: geom_schema},
            chunksize=sql_chunksize,
        )
    finally:
        conn.close()


def read_triplegs_postgis(conn_string, table_name, geom_col="geom", *args, **kwargs):
    """Reads triplegs from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The table to read the triplegs from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis(
            "SELECT * FROM %s" % table_name, conn, geom_col=geom_col, index_col="id", *args, **kwargs
        )
    finally:
        conn.close()
    assert pfs.as_triplegs
    return pfs


def write_triplegs_postgis(
    triplegs, conn_string, table_name, schema=None, sql_chunksize=None, if_exists="replace", *args, **kwargs
):
    """Stores triplegs to PostGIS. Usually, this is directly called on a triplegs
    DataFrame (see example below).

    **Attention!** This replaces the table if it already exists!

    Parameters
    ----------
    triplegs : GeoDataFrame
        The triplegs to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'replace'
        What should happen if the table already exists.

    Examples
    --------
    >>> df.as_triplegs.to_postgis(conn_string, table_name)
    """
    triplegs_postgis = triplegs.copy()
    if triplegs_postgis.crs is not None:
        srid = triplegs_postgis.crs.to_epsg()
    else:
        srid = -1
    geom_schema = Geometry("LINESTRING", srid)
    geom_col = triplegs_postgis.geometry.name
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry column does not contain geometry.", UserWarning)
        triplegs_postgis[geom_col] = triplegs[geom_col].apply(lambda x: WKTElement(x.wkt, srid=srid))

    if "id" not in triplegs_postgis.columns:
        triplegs_postgis["id"] = triplegs_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        triplegs_postgis.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            dtype={geom_col: geom_schema},
            chunksize=sql_chunksize,
        )
    finally:
        conn.close()


def read_staypoints_postgis(conn_string, table_name, geom_col="geom", *args, **kwargs):
    """Reads staypoints from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The table to read the staypoints from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis(
            "SELECT * FROM %s" % table_name, conn, geom_col=geom_col, index_col="id", *args, **kwargs
        )
    finally:
        conn.close()
    assert pfs.as_staypoints
    return pfs


def write_staypoints_postgis(staypoints, conn_string, table_name, schema=None, sql_chunksize=None, if_exists="replace"):
    """Stores staypoints to PostGIS. Usually, this is directly called on a staypoints
    DataFrame (see example below).

    **Attention!** This replaces the table if it already exists!

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'replace'
        What should happen if the table already exists.

    Examples
    --------
    >>> df.as_staypoints.to_postgis(conn_string, table_name)
    """

    # todo: Think about a concept for the indices. At the moment, an index
    # column is required when downloading. This means, that the ID column is
    # taken as pandas index. When uploading the default is "no index" and
    # thereby the index column is lost

    # make a copy in order to avoid changing the geometry of the original array
    staypoints_postgis = staypoints.copy()
    if staypoints_postgis.crs is not None:
        srid = staypoints_postgis.crs.to_epsg()
    else:
        srid = -1
    geom_schema = Geometry("POINT", srid)

    geom_col = staypoints_postgis.geometry.name
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry column does not contain geometry.", UserWarning)
        staypoints_postgis[geom_col] = staypoints[geom_col].apply(lambda x: WKTElement(x.wkt, srid=srid))

    if "id" not in staypoints_postgis.columns:
        staypoints_postgis["id"] = staypoints_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        staypoints_postgis.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            dtype={geom_col: geom_schema},
            chunksize=sql_chunksize,
        )
    finally:
        conn.close()


def read_locations_postgis(conn_string, table_name, geom_col="geom", *args, **kwargs):
    """Reads locations from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The table to read the locations from.

    geom_col : str, default 'geom'
        The geometry column of the table.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        locs = gpd.GeoDataFrame.from_postgis(
            "SELECT * FROM %s" % table_name, conn, geom_col=geom_col, index_col="id", *args, **kwargs
        )
    finally:
        conn.close()
    assert locs.as_locations
    return locs


def write_locations_postgis(locations, conn_string, table_name, schema=None, sql_chunksize=None, if_exists="replace"):
    """Stores locations to PostGIS. Usually, this is directly called on a locations
    GeoDataFrame (see example below).

    **Attention!** This replaces the table if it already exists!

    Parameters
    ----------
    locations : GeoDataFrame
        The locations to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'replace'
        What should happen if the table already exists.

    Examples
    --------
    >>> df.as_locations.to_postgis(conn_string, table_name)
    """

    # make a copy in order to avoid changing the geometry of the original array
    locations_postgis = locations.copy()

    if locations_postgis.crs is not None:
        srid = locations_postgis.crs.to_epsg()
    else:
        srid = -1
    center_schema = Geometry("POINT", srid)
    extent_schema = Geometry("GEOMETRY", srid)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry column does not contain geometry.", UserWarning)
        locations_postgis["center"] = locations["center"].apply(lambda x: WKTElement(x.wkt, srid=srid))
        if "extent" in locations_postgis.columns:
            locations_postgis["extent"] = locations["extent"].apply(lambda x: WKTElement(x.wkt, srid=srid))

    if "id" not in locations_postgis.columns:
        locations_postgis["id"] = locations_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    dtype = {"center": center_schema}
    if "extent" in locations_postgis.columns:
        dtype["extent"] = extent_schema
    try:
        locations_postgis.to_sql(
            table_name, engine, schema=schema, if_exists=if_exists, index=False, dtype=dtype, chunksize=sql_chunksize
        )
    finally:
        conn.close()


def read_trips_postgis(conn_string, table_name, *args, **kwargs):
    """Reads trips from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The table to read the trips from.

    Returns
    -------
    DataFrame
        A DataFrame containing the trips.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        trps = pd.read_sql("SELECT * FROM %s" % table_name, conn, index_col="id", *args, **kwargs)
    finally:
        conn.close()
    assert trps.as_trips
    return trps


def write_trips_postgis(trips, conn_string, table_name, schema=None, sql_chunksize=None, if_exists="replace"):
    """Stores trips to PostGIS. Usually, this is directly called on a trips
    DataFrame (see example below).

    **Attention!** This replaces the table if it already exists!

    Parameters
    ----------
    trips : DataFrame
        The trips to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g.,
        ``postgresql://username:password@host:socket/database``.

    table_name : str
        The name of the table to write to.

    schema : str, optional
        The schema (if the database supports this) where the table resides.

    sql_chunksize : int, optional
        How many entries should be written at the same time.

    if_exists : str, {'fail', 'replace', 'append'}, default 'replace'
        What should happen if the table already exists.

    Examples
    --------
    >>> df.as_trips.to_postgis(conn_string, table_name)
    """

    # make a copy in order to avoid changing the geometry of the original array
    trips_postgis = trips.copy()
    if "id" not in trips_postgis.columns:
        trips_postgis["id"] = trips_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        trips_postgis.to_sql(
            table_name, engine, schema=schema, if_exists=if_exists, index=False, chunksize=sql_chunksize
        )
    finally:
        conn.close()
