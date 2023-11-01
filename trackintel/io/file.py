import ast

import geopandas as gpd
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from trackintel.io.from_geopandas import (
    read_locations_gpd,
    read_positionfixes_gpd,
    read_staypoints_gpd,
    read_tours_gpd,
    read_triplegs_gpd,
    read_trips_gpd,
)
from trackintel.io.util import _index_warning_default_none
from trackintel.model.util import doc, _shared_docs


@_index_warning_default_none
def read_positionfixes_csv(*args, columns=None, tz=None, index_col=None, geom_col="geom", crs=None, **kwargs):
    """
    Read positionfixes from csv file.

    Wraps the pandas read_csv function, extracts longitude and latitude and
    builds a POINT GeoSeries, extracts datetime from column `tracked_at`.

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

    geom_col : str, default "geom"
        Name of the column containing the geometry.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg 'EPSG:4326') or a WKT string.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    pfs : Positionfixes

    Notes
    -----
    Note that this function is primarily useful if data is available in a
    longitude/latitude format. If your data already contains a WKT column,
    might be easier to just use the GeoPandas import functions
    :func:`trackintel.io.read_positionfixes_gpd`.

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

    df = pd.read_csv(*args, index_col=index_col, **kwargs)
    df.rename(columns=columns, inplace=True)

    df["tracked_at"] = pd.to_datetime(df["tracked_at"])
    df[geom_col] = gpd.points_from_xy(df["longitude"], df["latitude"])
    df.drop(columns=["longitude", "latitude"], inplace=True)
    return read_positionfixes_gpd(df, geom_col=geom_col, crs=crs, tz=tz)


def write_positionfixes_csv(positionfixes, filename, *args, **kwargs):
    """
    Write positionfixes to csv file.

    Wraps the pandas to_csv function, but strips the geometry column and
    stores the longitude and latitude in respective columns.

    Parameters
    ----------
    positionfixes : Positionfixes

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
    >>> pfs.to_csv("export_pfs.csv")
    >>> ti.io.write_positionfixes_csv(pfs, "export_pfs.csv")
    """
    gdf = positionfixes.copy()
    gdf["longitude"] = positionfixes.geometry.x
    gdf["latitude"] = positionfixes.geometry.y
    df = gdf.drop(columns=[gdf.geometry.name])

    df.to_csv(filename, index=True, *args, **kwargs)


@_index_warning_default_none
def read_triplegs_csv(*args, columns=None, tz=None, index_col=None, geom_col="geom", crs=None, **kwargs):
    """
    Read triplegs from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the tripleg geometry (LINESTRING)
    and builds a Triplegs instance, extracts datetime from column `started_at` & `finished_at`.

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
    tpls : Triplegs

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
    df = pd.read_csv(*args, index_col=index_col, **kwargs)
    df.rename(columns=columns, inplace=True)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])
    df[geom_col] = gpd.GeoSeries.from_wkt(df[geom_col])
    return read_triplegs_gpd(df, geom_col=geom_col, crs=crs, tz=tz, mapper=columns)


@doc(
    _shared_docs["write_csv"],
    first_arg="\ntriplegs : Triplegs\n",
    long="triplegs",
    short="tpls",
)
def write_triplegs_csv(triplegs, filename, *args, **kwargs):
    pd.DataFrame.to_csv(triplegs.to_wkt(rounding_precision=-1, trim=False), filename, index=True, *args, **kwargs)


@_index_warning_default_none
def read_staypoints_csv(*args, columns=None, tz=None, index_col=None, geom_col="geom", crs=None, **kwargs):
    """
    Read staypoints from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the staypoint geometry (Point)
    and builds a Staypoints instance, extracts datetime from column `started_at` & `finished_at`.

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
    sp : Staypoints

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
    df = pd.read_csv(*args, index_col=index_col, **kwargs)
    df.rename(columns=columns, inplace=True)
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])
    df[geom_col] = gpd.GeoSeries.from_wkt(df[geom_col])
    return read_staypoints_gpd(df, geom_col=geom_col, crs=crs, tz=tz)


@doc(
    _shared_docs["write_csv"],
    first_arg="\nstaypoints : Staypoints\n",
    long="staypoints",
    short="sp",
)
def write_staypoints_csv(staypoints, filename, *args, **kwargs):
    pd.DataFrame.to_csv(staypoints.to_wkt(rounding_precision=-1, trim=False), filename, index=True, *args, **kwargs)


@_index_warning_default_none
def read_locations_csv(*args, columns=None, index_col=None, crs=None, **kwargs):
    """
    Read locations from csv file.

    Wraps the pandas read_csv function, extracts a WKT for the location center geometry (POINT)
    (and optional extent (POLYGON)) and builds a Locations instance, extracts datetime from
    column `started_at` & `finished_at`.

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
    locs : Locations

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
    df = pd.read_csv(*args, index_col=index_col, **kwargs)
    df.rename(columns=columns, inplace=True)

    df["center"] = gpd.GeoSeries.from_wkt(df["center"])
    if "extent" in df.columns:
        df["extent"] = gpd.GeoSeries.from_wkt(df["extent"])
    return read_locations_gpd(df, crs=crs)


@doc(
    _shared_docs["write_csv"],
    first_arg="\nlocations : Locations\n",
    long="locations",
    short="locs",
)
def write_locations_csv(locations, filename, *args, **kwargs):
    pd.DataFrame.to_csv(locations.to_wkt(rounding_precision=-1, trim=False), filename, index=True, *args, **kwargs)


@_index_warning_default_none
def read_trips_csv(*args, columns=None, tz=None, index_col=None, geom_col=None, crs=None, **kwargs):
    """
    Read trips from csv file.

    Wraps the pandas read_csv function and builds a Trips instance.
    Extracts datetime from column `started_at` & `finished_at`.

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
    trips : Trips
        A TripsDataFrame containing the trips. TripsGeoDataFrame if geometry column exists.

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
    trips = pd.read_csv(*args, index_col=index_col, **kwargs)
    trips.rename(columns=columns, inplace=True)

    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips["finished_at"] = pd.to_datetime(trips["finished_at"])

    if geom_col is not None:
        trips[geom_col] = gpd.GeoSeries.from_wkt(trips[geom_col])

    return read_trips_gpd(trips, geom_col=geom_col, crs=crs, tz=tz)


@doc(_shared_docs["write_csv"], first_arg="\ntrips : Trips\n", long="trips", short="trips")
def write_trips_csv(trips, filename, *args, **kwargs):
    if isinstance(trips, GeoDataFrame):
        trips = trips.to_wkt(rounding_precision=-1, trim=False)
    # static call necessary as TripsDataFrame has a to_csv method as well.
    pd.DataFrame.to_csv(trips, filename, index=True, *args, **kwargs)


@_index_warning_default_none
def read_tours_csv(*args, columns=None, index_col=None, tz=None, **kwargs):
    """
    Read tours from csv file.

    Extracts datetime from column `started_at` & `finished_at`.

    Parameters
    ----------
    args
        Arguments as passed to pd.read_csv().

    columns : dict, optional
        The column names to rename in the format {'old_name':'trackintel_standard_name'}.

    index_col : str, optional
        column name to be used as index. If None the default index is assumed as unique identifier.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    kwargs
        Additional keyword arguments passed to pd.read_csv().

    Returns
    -------
    tours : Tours

    Examples
    --------
    >>> trackintel.read_tours_csv('data.csv', columns={'uuid':'user_id'})
    """
    columns = {} if columns is None else columns
    kwargs.setdefault("converters", {}).setdefault("trips", ast.literal_eval)
    tours = pd.read_csv(*args, index_col=index_col, **kwargs)
    tours.rename(columns=columns, inplace=True)

    tours["started_at"] = pd.to_datetime(tours["started_at"])
    tours["finished_at"] = pd.to_datetime(tours["finished_at"])

    return read_tours_gpd(tours, tz=tz)


@doc(_shared_docs["write_csv"], first_arg="\ntours : Tours\n", long="tours", short="tours")
def write_tours_csv(tours, filename, *args, **kwargs):
    pd.DataFrame.to_csv(tours, filename, index=True, *args, **kwargs)
