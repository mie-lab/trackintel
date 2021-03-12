import warnings

import numpy as np
import dateutil
import dateutil.parser
import geopandas as gpd
import pandas as pd
import pytz
import warnings
from shapely import wkt
from shapely.geometry import Point


def localize_timestamp(dt_series, pytz_tzinfo, col_name):
    """
    Helper function that adds timezone info to timestamp

    Parameters
    ----------
    dt_series: pandas.Series
        a pandas datetime series
    pytz_tzinfo: str
        pytz compatible timezone string. If none UTC will be assumed
    col_name: str
        Column name for informative warning message

    Returns
    -------
    pd.Series
        a timezone aware pandas datetime series

    """
    if pytz_tzinfo is None:
        warnings.warn("Assuming UTC timezone for column {}".format(col_name))
        pytz_tzinfo = 'utc'

    timezone = pytz.timezone(pytz_tzinfo)
    return dt_series.apply(pd.Timestamp, tz=timezone)


def read_positionfixes_csv(*args, columns=None, tz=None, index_col=object(), crs=None, **kwargs):
    """Wraps the pandas read_csv function, extracts longitude and latitude and
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of positionfixes (see
    :doc:`/modules/model`).

    Parameters
    ----------
    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.
    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.
    crs: pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the positionfixes.

    Notes
    -----
    Note that this function is primarily useful if data is available in a
    longitude/latitude format. If your data already contains a WKT column, it
    might be easier to just use the GeoPandas import functions.

    Examples
    --------
    >>> trackintel.read_positionfixes_csv('data.csv')
    >>> trackintel.read_positionfixes_csv('data.csv', columns={'time':'tracked_at', 'User':'user_id'})
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn("Assuming default index as unique identifier. Pass 'index_col=None' as explicit" +
                      "argument to avoid a warning when reading csv files.")
    elif index_col is not None:
        kwargs['index_col'] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)
    df['geom'] = list(zip(df.longitude, df.latitude))
    df['geom'] = df['geom'].apply(Point)
    df['tracked_at'] = df['tracked_at'].apply(dateutil.parser.parse)

    # check and/or set timezone
    for col in ['tracked_at']:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)
        else:
            # dateutil parser timezones are sometimes not compatible with pandas (e.g., in asserts)
            tz = df[col].iloc[0].tzinfo.tzname(df[col].iloc[0])
            df[col] = df[col].dt.tz_convert(tz)

    df = df.drop(['longitude', 'latitude'], axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geom')
    if crs:
        gdf.set_crs(crs, inplace=True)
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


def read_triplegs_csv(*args, columns=None, tz=None, index_col=object(), crs=None, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the leg geometry and
    builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of triplegs (see :doc:`/modules/model`).


    Parameters
    ----------
    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.
    index_col : str, optional
        column name to be used as index. If None the default index is assumed 
        as unique identifier.
    crs: pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the triplegs.
        
    Examples
    --------
    >>> trackintel.read_triplegs_csv('data.csv')
    >>> trackintel.read_triplegs_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn("Assuming default index as unique identifier. Pass 'index_col=None' as explicit" +
                      "argument to avoid a warning when reading csv files.")
    elif index_col is not None:
        kwargs['index_col'] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)
    df['geom'] = df['geom'].apply(wkt.loads)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)
        else:
            # dateutil parser timezones are sometimes not compatible with pandas (e.g., in asserts)
            tz = df[col].iloc[0].tzinfo.tzname(df[col].iloc[0])
            df[col] = df[col].dt.tz_convert(tz)

    gdf = gpd.GeoDataFrame(df, geometry='geom')
    if crs:
        gdf.set_crs(crs, inplace=True)
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
    geo_col_name = triplegs.geometry.name
    gdf = pd.DataFrame(triplegs, copy=True)
    gdf[geo_col_name] = triplegs.geometry.apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_staypoints_csv(*args, columns=None, tz=None, index_col=object(), crs=None, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the staypoint 
    geometry and builds a geopandas GeoDataFrame. This also validates that 
    the ingested data conforms to the trackintel understanding of staypoints 
    (see :doc:`/modules/model`).
    
    Parameters
    ----------
    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.
    index_col : str, optional
        column name to be used as index. If None the default index is assumed 
        as unique identifier.
    crs: pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the staypoints.
            
    Examples
    --------
    >>> trackintel.read_staypoints_csv('data.csv')
    >>> trackintel.read_staypoints_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
    """
    columns = {} if columns is None else columns
    
    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn("Assuming default index as unique identifier. Pass 'index_col=None' as explicit" +
                      "argument to avoid a warning when reading csv files.")
    elif index_col is not None:
        kwargs['index_col'] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)
    df['geom'] = df['geom'].apply(wkt.loads)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)
        else:
            # dateutil parser timezones are sometimes not compatible with pandas (e.g., in asserts)
            tz = df[col].iloc[0].tzinfo.tzname(df[col].iloc[0])
            df[col] = df[col].dt.tz_convert(tz)
        
    gdf = gpd.GeoDataFrame(df, geometry='geom')
    if crs:
        gdf.set_crs(crs, inplace=True)
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
    geo_col_name = staypoints.geometry.name
    gdf = pd.DataFrame(staypoints, copy=True)
    gdf[geo_col_name] = staypoints.geometry.apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_locations_csv(*args, columns=None, index_col=object(), crs=None, **kwargs):
    """Wraps the pandas read_csv function, extracts a WKT for the location 
    center (and extent) and builds a geopandas GeoDataFrame. This also 
    validates that the ingested data conforms to the trackintel understanding 
    of locations (see :doc:`/modules/model`).

    Parameters
    ----------
    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.
    crs: pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the locations.

    Examples
    --------
    >>> trackintel.read_locations_csv('data.csv')
    >>> trackintel.read_locations_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
    """
    columns = {} if columns is None else columns
    
    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn("Assuming default index as unique identifier. Pass 'index_col=None' as explicit" +
                      "argument to avoid a warning when reading csv files.")
    elif index_col is not None:
        kwargs['index_col'] = index_col
    
    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)
    df['center'] = df['center'].apply(wkt.loads)
    if 'extent' in df.columns:
        df['extent'] = df['extent'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='center')
    if crs:
        gdf.set_crs(crs, inplace=True)
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
    gdf = pd.DataFrame(locations, copy=True)
    gdf['center'] = locations['center'].apply(wkt.dumps)
    if 'extent' in gdf.columns:
        gdf['extent'] = locations['extent'].apply(wkt.dumps)
    gdf.to_csv(filename, index=True, *args, **kwargs)


def read_trips_csv(*args, columns=None, tz=None, index_col=object(), **kwargs):
    """Wraps the pandas read_csv function and extracts proper datetimes. This also
    validates that the ingested data conforms to the trackintel understanding 
    of trips (see :doc:`/modules/model`).
    
    Parameters
    ----------
    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.
    index_col : str, optional
        column name to be used as index. If None the default index is assumed 
        as unique identifier.
        
    Returns
    -------
    DataFrame
        A DataFrame containing the trips.
        
            
    Examples
    --------
    >>> trackintel.read_trips_csv('data.csv')
    >>> trackintel.read_trips_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
    """
    columns = {} if columns is None else columns

    if type(index_col) == object:
        warnings.warn("Assuming default index as unique identifier. Pass 'index_col=None' as explicit" +
                      "argument to avoid a warning when reading csv files.")
    elif index_col is not None:
        kwargs['index_col'] = index_col
    
    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)
    df['started_at'] = df['started_at'].apply(dateutil.parser.parse)
    df['finished_at'] = df['finished_at'].apply(dateutil.parser.parse)

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)
        else:
            # dateutil parser timezones are sometimes not compatible with pandas (e.g., in asserts)
            tz = df[col].iloc[0].tzinfo.tzname(df[col].iloc[0])
            df[col] = df[col].dt.tz_convert(tz)

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


def read_tours_csv(*args, columns=None, tz=None, **kwargs):
    """Wraps the pandas read_csv function and extracts proper datetimes. This also
    validates that the ingested data conforms to the trackintel understanding 
    of tours (see :doc:`/modules/model`).

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    Returns
    -------
    DataFrame
        A DataFrame containing the tours.
    """
    # TODO: implement the reading function for tours

    # # check and/or set timezone
    # for col in ['started_at', 'finished_at']:
    #     if not pd.api.types.is_datetime64tz_dtype(df[col]):
    #         df[col] = localize_timestamp(dt_series=df[col], pytz_tzinfo=kwargs.pop('tz', None), col_name=col)
    #         else:
    #             # dateutil parser timezones are sometimes not compatible with pandas (e.g., in asserts)
    #             tz = df[col].iloc[0].tzinfo.tzname(df[col].iloc[0])
    #             df[col] = df[col].dt.tz_convert(tz)
    pass


def write_tours_csv(tours, filename, *args, **kwargs):
    """Wraps the pandas to_csv function.

    Parameters
    ----------
    tours : DataFrame
        The tours to store to the CSV file.
    
    filename : str
        The file to write to.
    """
    # TODO: implement the writing function for tours
    pass
