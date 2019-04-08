import dateutil.parser

import pandas as pd
import geopandas as gpd
import shapely

from shapely.geometry import Point
from shapely import wkt


def read_positionfixes_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts longitude and latitude and 
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of positionfixes. 
    See :doc:`/modules/model`.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geom'] = list(zip(df.longitude, df.latitude))
    df['geom'] = df['geom'].apply(Point)
    df['tracked_at'] = df['tracked_at'].apply(dateutil.parser.parse)
    df = df.drop(['longitude', 'latitude'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geom')
    assert gdf.as_positionfixes
    return gdf


def write_positionfixes_csv(positionfixes, filename, *args, **kwargs):
    """Wraps the pandas to_csv function, but strips the geometry column ('geom') and 
    stores the longitude and latitude in respective columns.

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    gdf = positionfixes.copy()
    gdf['longitude'] = positionfixes['geom'].apply(lambda p: p.coords[0][0])
    gdf['latitude'] = positionfixes['geom'].apply(lambda p: p.coords[0][1])
    gdf = gdf.drop('geom', axis=1)
    gdf.to_csv(filename, index=False, *args, **kwargs)


def read_triplegs_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the leg geometry and
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of triplegs. 
    See :doc:`/modules/model`.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geom'] = df['geom'].apply(wkt.loads)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)
    gdf = gpd.GeoDataFrame(df, geometry='geom')
    assert gdf.as_triplegs
    return gdf


def write_triplegs_csv(triplegs, filename, *args, **kwargs):
    """Wraps the pandas to_csv function, but transforms the geom into WKT 
    before writing.

    Parameters
    ----------
    triplegs : GeoDataFrame
        The triplegs to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    gdf = triplegs.copy()
    gdf['geom'] = triplegs['geom'].apply(wkt.dumps)
    gdf.to_csv(filename, index=False, *args, **kwargs)


def read_staypoints_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the staypoint 
    geometry and builds a geopandas GeoDataFrame. This also validates that 
    the ingested data conforms to the trackintel understanding of staypoints. 
    See :doc:`/modules/model`.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
    """
    df = pd.read_csv(*args, **kwargs)
    df['geom'] = df['geom'].apply(wkt.loads)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)
    gdf = gpd.GeoDataFrame(df, geometry='geom')
    assert gdf.as_staypoints
    return gdf


def write_staypoints_csv(staypoints, filename, *args, **kwargs):
    """Wraps the pandas to_csv function, but transforms the geom into WKT 
    before writing.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    gdf = staypoints.copy()
    gdf['geom'] = staypoints['geom'].apply(wkt.dumps)
    gdf.to_csv(filename, index=False, *args, **kwargs)
