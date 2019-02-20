import geopandas as gpd
from geoalchemy2 import Geometry, WKTElement
from sqlalchemy import *


def read_positionfixes_postgis(conn_string, table_name, *args, **kwargs):
    """Reads positionfixes from a PostGIS database.

    Parameters
    ----------
    conn_string : str
        A connection string to connect to a database, e.g., 
        `'postgresql://username:password@host:socket/database'`.
    
    table_name : str
        The table to read the positionfixes from.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    engine = create_engine(conn_string)
    conn = engine.connect()
    try:
        pfs = gpd.GeoDataFrame.from_postgis("SELECT * FROM %s" % table_name, conn, 
                                            geom_col='geom', index_col='id')
    finally:
        conn.close()
    assert pfs.as_positionfixes
    return pfs


def write_positionfixes_postgis(positionfixes, conn_string, table_name):
    """Stores positionfixes to PostGIS.

    Attention! This replaces the table if it already exists!

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes to store to the database.

    conn_string : str
        A connection string to connect to a database, e.g., 
        `'postgresql://username:password@host:socket/database'`.
    
    table_name : str
        The name of the table to write to.
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