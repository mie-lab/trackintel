import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
from geopandas.geodataframe import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point


def read_positionfixes_csv(*args, columns=None, tz=None, index_col=object(), crs=None, **kwargs):
    """
    Read positionfixes from csv file.

    Wraps the pandas read_csv function, extracts longitude and latitude and
    builds a geopandas GeoDataFrame (POINT). This also validates that the ingested data
    conforms to the trackintel understanding of positionfixes (see
    :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
        The required columns for this function include: "user_id", "tracked_at", "latitude"
        and "longitude".

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg 'EPSG:4326') or a WKT string.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    pfs : GeoDataFrame (as trackintel positionfixes)
        A GeoDataFrame containing the positionfixes.

    Notes
    -----
    Note that this function is primarily useful if data is available in a
    longitude/latitude format. If your data already contains a WKT column,
    might be easier to just use the GeoPandas import functions
    :func:`trackintel.io.from_geopandas.read_positionfixes_gpd`.

    Examples
    --------
    >>> trackintel.read_positionfixes_csv('data.csv')
    >>> trackintel.read_positionfixes_csv('data.csv', columns={'time':'tracked_at', 'User':'user_id'})
                         tracked_at  user_id                        geom
    id
    0     2008-10-23 02:53:04+00:00        0  POINT (116.31842 39.98470)
    1     2008-10-23 02:53:10+00:00        0  POINT (116.31845 39.98468)
    2     2008-10-23 02:53:15+00:00        0  POINT (116.31842 39.98469)
    3     2008-10-23 02:53:20+00:00        0  POINT (116.31839 39.98469)
    4     2008-10-23 02:53:25+00:00        0  POINT (116.31826 39.98465)
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)

    # construct geom column from lon and lat
    df["geom"] = list(zip(df["longitude"], df["latitude"]))
    df["geom"] = df["geom"].apply(Point)

    # transform to datatime
    df["tracked_at"] = pd.to_datetime(df["tracked_at"])

    # set timezone if none is recognized
    for col in ["tracked_at"]:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = _localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)

    df = df.drop(["longitude", "latitude"], axis=1)
    pfs = gpd.GeoDataFrame(df, geometry="geom")
    if crs:
        pfs.set_crs(crs, inplace=True)

    # assert validity of positionfixes
    pfs.as_positionfixes
    return pfs


def write_positionfixes_csv(positionfixes, filename, *args, **kwargs):
    """
    Write positionfixes to csv file.

    Wraps the pandas to_csv function, but strips the geometry column and
    stores the longitude and latitude in respective columns.

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Notes
    -----
    "longitude" and "latitude" is extracted from the geometry column and the orignal
    geometry column is dropped.

    Examples
    ---------
    >>> ps.as_positionfixes.to_csv("export_pfs.csv")
    """
    gdf = positionfixes.copy()
    gdf["longitude"] = positionfixes.geometry.apply(lambda p: p.coords[0][0])
    gdf["latitude"] = positionfixes.geometry.apply(lambda p: p.coords[0][1])
    df = gdf.drop(gdf.geometry.name, axis=1)

    df.to_csv(filename, index=True, *args, **kwargs)


def read_triplegs_csv(*args, columns=None, tz=None, index_col=object(), geom_col="geom", crs=None, **kwargs):
    """
    Read triplegs from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the tripleg geometry (LINESTRING)
    and builds a geopandas GeoDataFrame. This also validates that the ingested data
    conforms to the trackintel understanding of triplegs (see :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
        The required columns for this function include: "user_id", "started_at", "finished_at"
        and "geom".

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    index_col : str, optional
        Column name to be used as index. If None the default index is assumed
        as unique identifier.

    geom_col : str, default "geom"
        Name of the column containing the geometry as WKT.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    tpls : GeoDataFrame (as trackintel triplegs)
        A GeoDataFrame containing the triplegs.

    Examples
    --------
    >>> trackintel.read_triplegs_csv('data.csv')
    >>> trackintel.read_triplegs_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
        user_id                started_at               finished_at                                               geom
    id
    0         1 2015-11-27 08:00:00+00:00 2015-11-27 10:00:00+00:00  LINESTRING (8.54878 47.37652, 8.52770 47.39935...
    1         1 2015-11-27 12:00:00+00:00 2015-11-27 14:00:00+00:00  LINESTRING (8.56340 47.95600, 8.64560 47.23345...
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)

    # construct geom column
    df[geom_col] = df[geom_col].apply(wkt.loads)

    # transform to datatime
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])

    # set timezone if none is recognized
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = _localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)

    tpls = gpd.GeoDataFrame(df, geometry="geom")
    if crs:
        tpls.set_crs(crs, inplace=True)

    # assert validity of triplegs
    tpls.as_triplegs
    return tpls


def write_triplegs_csv(triplegs, filename, *args, **kwargs):
    """
    Write triplegs to csv file.

    Wraps the pandas to_csv function, but transforms the geometry into WKT
    before writing.

    Parameters
    ----------
    triplegs : GeoDataFrame (as trackintel triplegs)
        The triplegs to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Examples
    --------
    >>> tpls.as_triplegs.to_csv("export_tpls.csv")
    """
    geo_col_name = triplegs.geometry.name
    df = pd.DataFrame(triplegs, copy=True)
    df[geo_col_name] = triplegs.geometry.apply(wkt.dumps)
    df.to_csv(filename, index=True, *args, **kwargs)


def read_staypoints_csv(*args, columns=None, tz=None, index_col=object(), geom_col="geom", crs=None, **kwargs):
    """
    Read staypoints from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the staypoint
    geometry (POINT) and builds a geopandas GeoDataFrame. This also validates that
    the ingested data conforms to the trackintel understanding of staypoints
    (see :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
        The required columns for this function include: "user_id", "started_at", "finished_at"
        and "geom".

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.

    geom_col : str, default "geom"
        Name of the column containing the geometry as WKT.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    sp : GeoDataFrame (as trackintel staypoints)
        A GeoDataFrame containing the staypoints.

    Examples
    --------
    >>> trackintel.read_staypoints_csv('data.csv')
    >>> trackintel.read_staypoints_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
        user_id                started_at               finished_at                      geom
    id
    0         1 2015-11-27 08:00:00+00:00 2015-11-27 10:00:00+00:00  POINT (8.52822 47.39519)
    1         1 2015-11-27 12:00:00+00:00 2015-11-27 14:00:00+00:00  POINT (8.54340 47.95600)
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)

    # construct geom column
    df[geom_col] = df[geom_col].apply(wkt.loads)

    # transform to datatime
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])

    # set timezone if none is recognized
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = _localize_timestamp(dt_series=df[col], pytz_tzinfo=tz, col_name=col)

    sp = gpd.GeoDataFrame(df, geometry="geom")
    if crs:
        sp.set_crs(crs, inplace=True)

    # assert validity of staypoints
    sp.as_staypoints
    return sp


def write_staypoints_csv(staypoints, filename, *args, **kwargs):
    """
    Write staypoints to csv file.

    Wraps the pandas to_csv function, but transforms the geometry into WKT
    before writing.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        The staypoints to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Examples
    --------
    >>> tpls.as_triplegs.to_csv("export_tpls.csv")
    """
    geo_col_name = staypoints.geometry.name
    df = pd.DataFrame(staypoints, copy=True)
    df[geo_col_name] = staypoints.geometry.apply(wkt.dumps)
    df.to_csv(filename, index=True, *args, **kwargs)


def read_locations_csv(*args, columns=None, index_col=object(), geom_col="geom", crs=None, **kwargs):
    """
    Read locations from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the location
    center (POINT) (and extent (POLYGON)) and builds a geopandas GeoDataFrame. This also
    validates that the ingested data conforms to the trackintel understanding
    of locations (see :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
        The required columns for this function include: "user_id" and "center".

    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    locs : GeoDataFrame (as trackintel locations)
        A GeoDataFrame containing the locations.

    Examples
    --------
    >>> trackintel.read_locations_csv('data.csv')
    >>> trackintel.read_locations_csv('data.csv', columns={'User':'user_id'})
        user_id                    center                                             extent
    id
    0         1  POINT (8.54878 47.37652)  POLYGON ((8.548779487999999 47.37651505, 8.527...
    1         1  POINT (8.56340 47.95600)  POLYGON ((8.5634 47.956, 8.6456 47.23345, 8.45...
    """
    columns = {} if columns is None else columns

    # Warning if no 'index_col' parameter is provided
    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    df = pd.read_csv(*args, **kwargs)
    df = df.rename(columns=columns)

    # construct center and extent columns
    df["center"] = df["center"].apply(wkt.loads)
    if "extent" in df.columns:
        df["extent"] = df["extent"].apply(wkt.loads)

    locs = gpd.GeoDataFrame(df, geometry="center")
    if crs:
        locs.set_crs(crs, inplace=True)

    # assert validity of locations
    locs.as_locations
    return locs


def write_locations_csv(locations, filename, *args, **kwargs):
    """
    Write locations to csv file.

    Wraps the pandas to_csv function, but transforms the center (and
    extent) into WKT before writing.

    Parameters
    ----------
    locations : GeoDataFrame (as trackintel locations)
        The locations to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Examples
    --------
    >>> locs.as_locations.to_csv("export_locs.csv")
    """
    df = pd.DataFrame(locations, copy=True)
    df["center"] = locations["center"].apply(wkt.dumps)
    if "extent" in df.columns:
        df["extent"] = locations["extent"].apply(wkt.dumps)
    df.to_csv(filename, index=True, *args, **kwargs)


def read_trips_csv(*args, columns=None, tz=None, index_col=object(), geom_col=None, crs=None, **kwargs):
    """
    Read trips from csv file.

    Wraps the pandas read_csv function and extracts proper datetimes. This also
    validates that the ingested data conforms to the trackintel understanding
    of trips (see :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.
        The required columns for this function include: "user_id", "started_at",
        "finished_at", "origin_staypoint_id" and "destination_staypoint_id".
        An optional column is "geom" of type MultiPoint, containing start and destination points of the trip

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    index_col : str, optional
        column name to be used as index. If None the default index is assumed
        as unique identifier.

    geom_col : str, default None
        Name of the column containing the geometry as WKT.
        If None no geometry gets added.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg “EPSG:4326”) or a WKT string. Ignored if geom_col is None.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    trips : (Geo)DataFrame (as trackintel trips)
        A DataFrame containing the trips. GeoDataFrame if geometry column exists.

    Notes
    -----
    Geometry is not mandatory for trackintel trips.

    Examples
    --------
    >>> trackintel.read_trips_csv('data.csv')
    >>> trackintel.read_trips_csv('data.csv', columns={'start_time':'started_at', 'User':'user_id'})
        user_id                started_at               finished_at  origin_staypoint_id  destination_staypoint_id\
    id
    0         1 2015-11-27 08:00:00+00:00 2015-11-27 08:15:00+00:00                    2                         5
    1         1 2015-11-27 08:20:22+00:00 2015-11-27 08:35:22+00:00                    5                         3
                                geom  
    id                                                     
    0   MULTIPOINT (116.31842 39.98470, 116.29873 39.999729)
    1   MULTIPOINT (116.29873 39.98402, 116.32480 40.009269)
    """
    columns = {} if columns is None else columns

    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    trips = pd.read_csv(*args, **kwargs)
    trips = trips.rename(columns=columns)

    # transform to datatime
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips["finished_at"] = pd.to_datetime(trips["finished_at"])

    # check and/or set timezone
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(trips[col]):
            trips[col] = _localize_timestamp(dt_series=trips[col], pytz_tzinfo=tz, col_name=col)

    # convert to geodataframe
    if geom_col is not None:
        trips[geom_col] = trips[geom_col].apply(wkt.loads)
        trips = gpd.GeoDataFrame(trips, geometry=geom_col, crs=crs)

    # assert validity of trips
    trips.as_trips
    return trips


def write_trips_csv(trips, filename, *args, **kwargs):
    """
    Write trips to csv file.

    Wraps the pandas to_csv function.
    Geometry get transformed to WKT before writing.

    Parameters
    ----------
    trips : (Geo)DataFrame (as trackintel trips)
        The trips to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Examples
    --------
    >>> trips.as_trips.to_csv("export_trips.csv")
    """
    df = trips.copy()
    if isinstance(df, GeoDataFrame):
        geom_col_name = df.geometry.name
        df[geom_col_name] = df[geom_col_name].apply(wkt.dumps)
    df.to_csv(filename, index=True, *args, **kwargs)


def read_tours_csv(*args, columns=None, index_col=object(), tz=None, **kwargs):
    """
    Read tours from csv file.

    Wraps the pandas read_csv function and extracts proper datetimes. This also
    validates that the ingested data conforms to the trackintel understanding
    of tours (see :doc:`/modules/model`).

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    tours : DataFrame (as trackintel tours)
        A DataFrame containing the tours.

    Examples
    --------
    >>> trackintel.read_tours_csv('data.csv', columns={'uuid':'user_id'})
    """
    columns = {} if columns is None else columns

    if type(index_col) == object:
        warnings.warn(
            "Assuming default index as unique identifier. Pass 'index_col=None' as explicit"
            + "argument to avoid a warning when reading csv files."
        )
    elif index_col is not None:
        kwargs["index_col"] = index_col

    tours = pd.read_csv(*args, **kwargs)
    tours = tours.rename(columns=columns)

    # transform to datatime
    tours["started_at"] = pd.to_datetime(tours["started_at"])
    tours["finished_at"] = pd.to_datetime(tours["finished_at"])

    # check and/or set timezone
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(tours[col]):
            tours[col] = _localize_timestamp(dt_series=tours[col], pytz_tzinfo=tz, col_name=col)

    # assert validity of tours
    tours.as_tours
    return tours


def write_tours_csv(tours, filename, *args, **kwargs):
    """
    Write tours to csv file.

    Wraps the pandas to_csv function.

    Parameters
    ----------
    tours : DataFrame (as trackintel tours)
        The tours to store to the CSV file.

    filename : str
        The file to write to.

    args
        Additional arguments passed to pd.DataFrame.to_csv().

    kwargs
        Additional keyword arguments passed to pd.DataFrame.to_csv().

    Examples
    --------
    >>> tours.as_tours.to_csv("export_tours.csv")
    """
    tours.to_csv(filename, index=True, *args, **kwargs)


def _localize_timestamp(dt_series, pytz_tzinfo, col_name):
    """
    Add timezone info to timestamp.

    Parameters
    ----------
    dt_series : pandas.Series
        a pandas datetime series

    pytz_tzinfo : str
        pytz compatible timezone string. If none UTC will be assumed

    col_name : str
        Column name for informative warning message

    Returns
    -------
    pd.Series
        a timezone aware pandas datetime series
    """
    if pytz_tzinfo is None:
        warnings.warn("Assuming UTC timezone for column {}".format(col_name))
        pytz_tzinfo = "utc"

    timezone = pytz.timezone(pytz_tzinfo)
    return dt_series.apply(pd.Timestamp, tz=timezone)
