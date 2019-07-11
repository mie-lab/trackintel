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
                                            geom_col=geom_col,
                                            *args, **kwargs)
    finally:
        conn.close()
    assert pfs.as_positionfixes
    return pfs


def write_positionfixes_postgis(positionfixes, conn_string, table_name, schema=None,
                                sql_chunksize=None,  if_exists='replace'):
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
    
    # make a copy in order to avoid changing the geometry of the original array
    positionfixes_postgis = positionfixes.copy()

    # If this GeoDataFrame already has an SRID, we use it, otherwise we default to WGS84.
    if (positionfixes_postgis.crs is not None):
        srid = int(positionfixes_postgis.crs['init'].split(':')[1])
    else:
        srid = 4326
    positionfixes_postgis['geom'] = \
    positionfixes_postgis['geom'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    if 'id' not in positionfixes_postgis.columns:
        positionfixes_postgis['id'] = positionfixes_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        positionfixes_postgis.to_sql(table_name, engine, schema=schema,
                                     if_exists=if_exists, index=False,  
                                     dtype={'geom': Geometry('POINT', srid=srid)},
                                     chunksize=sql_chunksize)
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


def write_triplegs_postgis(triplegs, conn_string, table_name, schema=None,
                           sql_chunksize=None):
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
    
    # make a copy in order to avoid changing the geometry of the original array
    triplegs_postgis = triplegs.copy()
    
    srid = int(triplegs_postgis.crs['init'].split(':')[1])
    triplegs_postgis['geom'] = \
        triplegs_postgis['geom'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    if 'id' not in triplegs_postgis.columns:
        triplegs_postgis['id'] = triplegs_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        triplegs_postgis.to_sql(table_name, engine, schema=schema,
                                if_exists='replace', index=False, 
                        dtype={'geom': Geometry('LINESTRING', srid=srid)},
                        chunksize=sql_chunksize)
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


def write_staypoints_postgis(staypoints, conn_string, table_name, schema=None,
                             sql_chunksize=None, if_exists='replace'):
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
    
    # todo: Think about a conecpt for the indices. At the moment, an index 
    # column is required when downloading. This means, that the ID column is 
    # taken as pandas index. When uploading the default is "no index" and
    # thereby the index column is lost
    
    # make a copy in order to avoid changing the geometry of the original array
    staypoints_postgis = staypoints.copy()
    
    srid = int(staypoints_postgis.crs['init'].split(':')[1])
    staypoints_postgis['geom'] = \
        staypoints_postgis['geom'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    if 'id' not in staypoints_postgis.columns:
        staypoints_postgis['id'] = staypoints_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        staypoints_postgis.to_sql(table_name, engine, schema=schema,
                                  if_exists=if_exists, index=False, 
                        dtype={'geom': Geometry('POINT', srid=srid)},
                        chunksize=sql_chunksize)
    finally:
        conn.close()
        
        
def read_places_postgis(conn_string, table_name, geom_col='geom', *args, **kwargs):
    """Reads places from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g., 
        ``postgresql://username:password@host:socket/database``.
    
    table_name : str
        The table to read the places from.

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
        plcs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, 
                                            geom_col=geom_col, index_col='id',
                                            *args, **kwargs)
    finally:
        conn.close()
    assert plcs.as_places
    return plcs


def write_places_postgis(places, conn_string, table_name, schema=None,
                         sql_chunksize=None, if_exists='replace'):
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
    
    # make a copy in order to avoid changing the geometry of the original array
    places_postgis = places.copy()
    
    srid = int(places_postgis.crs['init'].split(':')[1])
    places_postgis['center'] = \
        places_postgis['center'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    places_postgis['extent'] = \
        places_postgis['extent'].apply(lambda x: WKTElement(x.wkt, srid=srid))
    if 'id' not in places_postgis.columns:
        places_postgis['id'] = places_postgis.index

    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        places_postgis.to_sql(table_name, engine, schema=schema,
                              if_exists=if_exists, index=False, 
                        dtype={'center': Geometry('POINT', srid=srid),
                               'extent': Geometry('GEOMETRY', srid=srid)},
                               chunksize=sql_chunksize)
    finally:
        conn.close()

        