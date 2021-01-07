import dateutil.parser
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point


def read_positionfixes_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts longitude and latitude and 
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of positionfixes (see 
    :doc:`/modules/model`). 

    Note that this function is primarily useful if data is available in a 
    longitude/latitude format. If your data already contains a WKT column, it
    might be easier to just use the GeoPandas import functions.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.

    Examples
    --------
    >>> trackintel.read_positionfixes_csv('data.csv')
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
    gdf['longitude'] = positionfixes.geometry.apply(lambda p: p.coords[0][0])
    gdf['latitude'] = positionfixes.geometry.apply(lambda p: p.coords[0][1])
    gdf = gdf.drop(gdf.geometry.name, axis=1)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_triplegs_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the leg geometry and
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of triplegs (see :doc:`/modules/model`).

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
    gdf[gdf.geometry.name] = triplegs.geometry.apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_staypoints_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the staypoint 
    geometry and builds a geopandas GeoDataFrame. This also validates that 
    the ingested data conforms to the trackintel understanding of staypoints 
    (see :doc:`/modules/model`).

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
    gdf[gdf.geometry.name] = gdf.geometry.apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_locations_csv(*args, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the location 
    center (and extent) and builds a geopandas GeoDataFrame. This also 
    validates that the ingested data conforms to the trackintel understanding 
    of locations (see :doc:`/modules/model`).

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.
    """
    # Todo: How to implement flexible geometry names in locations (gdf with potentially 2 geometry columns)
    df = pd.read_csv(*args, **kwargs)
    df['center'] = df['center'].apply(wkt.loads)
    if 'extent' in df.columns:
        df['extent'] = df['extent'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='center')
    assert gdf.as_locations
    return gdf


def write_locations_csv(locations, filename, *args, **kwargs):
    """Wraps the pandas to_csv function, but transforms the center (and 
    extent) into WKT before writing.

    Parameters
    ----------
    locations : GeoDataFrame
        The locations to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    gdf = locations.copy()
    gdf['center'] = locations['center'].apply(wkt.dumps)
    if 'extent' in gdf.columns:
        gdf['extent'] = locations['extent'].apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_trips_csv(*args, **kwargs):
    """Wraps the pandas read_csv function and extraces proper datetimes. This also 
    validates that the ingested data conforms to the trackintel understanding 
    of trips (see :doc:`/modules/model`).

    Returns
    -------
    DataFrame
        A DataFrame containing the trips.
    """
    df = pd.read_csv(*args, **kwargs)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)
    assert df.as_trips
    return df


def write_trips_csv(trips, filename, *args, **kwargs):
    """Wraps the pandas to_csv function.

    Parameters
    ----------
    trips : DataFrame
        The trips to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    df = trips.copy()
    df.to_csv(filename, index=True, *args, **kwargs)
