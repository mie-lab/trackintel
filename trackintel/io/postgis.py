import geopandas as gpd
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import create_engine


def read_positionfixes_postgis(conn_string, table_name, geom_col='geom', *args, **kwargs):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g., 
        ``postgresql://username:password@host:socket/database``.
    
    table_name : str
        The table to read the positionfixes from.

    geom_col : str
        The geometry column of the table. Default ist 'geom'

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, 
                                            geom_col=geom_col, index_col='id',
                                            *args, **kwargs)
    finally:
        conn.close()
    assert pfs.as_positionfixes
    return pfs


def write_positionfixes_postgis(positionfixes, conn_string, table_name):
    """Stores positionfixes to PostGIS. Usually, this is directly called on a positionfixes 
    dataframe (see example below).

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

    Examples
    --------
    >>> df.as_positionfixes.to_postgis(conn_string, table_name)
    """
    positionfixes['geom'] = positionfixes['geom'].apply(lambda x: WKTElement(x.wkt, srid=4326))
    if 'id' not in positionfixes.columns:
        positionfixes['id'] = positionfixes.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        positionfixes.to_sql(table_name, engine, if_exists='replace', index=False, 
                             dtype={'geom': Geometry('POINT', srid=4326)})
    finally:
        conn.close()


def read_triplegs_postgis(conn_string, table_name, geom_col='geom', *args, **kwargs):
    """Reads triplegs from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g., 
        ``postgresql://username:password@host:socket/database``.
    
    table_name : str
        The table to read the triplegs from.

    geom_col : str
        The geometry column of the table. Default ist 'geom'

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, 
                                            geom_col=geom_col, index_col='id',
                                            *args, **kwargs)
    finally:
        conn.close()
    assert pfs.as_triplegs
    return pfs


def write_triplegs_postgis(triplegs, conn_string, table_name):
    """Stores triplegs to PostGIS. Usually, this is directly called on a triplegs 
    dataframe (see example below).

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

    Examples
    --------
    >>> df.as_triplegs.to_postgis(conn_string, table_name)
    """
    triplegs['geom'] = triplegs['geom'].apply(lambda x: WKTElement(x.wkt, srid=4326))
    if 'id' not in triplegs.columns:
        triplegs['id'] = triplegs.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        triplegs.to_sql(table_name, engine, if_exists='replace', index=False, 
                        dtype={'geom': Geometry('LINESTRING', srid=4326)})
    finally:
        conn.close()


def read_staypoints_postgis(conn_string, table_name, geom_col='geom', *args, **kwargs):
    """Reads staypoints from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g., 
        ``postgresql://username:password@host:socket/database``.
    
    table_name : str
        The table to read the staypoints from.

    geom_col : str
        The geometry column of the table. Default ist 'geom'

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, 
                                            geom_col=geom_col, index_col='id',
                                            *args, **kwargs)
    finally:
        conn.close()
    assert pfs.as_staypoints
    return pfs


def write_staypoints_postgis(staypoints, conn_string, table_name):
    """Stores staypoints to PostGIS. Usually, this is directly called on a staypoints 
    dataframe (see example below).

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

    Examples
    --------
    >>> df.as_staypoints.to_postgis(conn_string, table_name)
    """
    staypoints['geom'] = staypoints['geom'].apply(lambda x: WKTElement(x.wkt, srid=4326))
    if 'id' not in staypoints.columns:
        staypoints['id'] = staypoints.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        staypoints.to_sql(table_name, engine, if_exists='replace', index=False, 
                        dtype={'geom': Geometry('POINT', srid=4326)})
    finally:
        conn.close()

        