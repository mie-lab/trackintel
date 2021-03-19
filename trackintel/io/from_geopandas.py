import pandas as pd

from trackintel.io.file import localize_timestamp


def read_positionfixes_gpd(gdf, tracked_at='tracked_at', user_id='user_id', geom='geom', tz=None, mapper={}):
    """
    warps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf :
        GeoDataFrame with valid point geometry, containing the positionfixes to import
    tracked_at : str
        name of the column storing the timestamps. The default is 'tracked_at'.
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    geom : str
        name of the column storing the geometry. The default is 'geom'.
    tz: str
        pytz compatible timezone string. If None UTC will be assumed
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the positionfixes

    """

    columns = {tracked_at: 'tracked_at',
               user_id: 'user_id',
               geom: 'geom'}
    columns.update(mapper)

    pfs = gdf.rename(columns=columns)
    pfs = pfs.set_geometry('geom')

    # check and/or set timezone
    for col in ['tracked_at']:
        if not pd.api.types.is_datetime64tz_dtype(pfs[col]):
            pfs[col] = localize_timestamp(dt_series=pfs[col], pytz_tzinfo=tz, col_name=col)
        
    assert pfs.as_positionfixes
    return pfs


def staypoints_from_gpd(gdf, started_at='started_at', finished_at='finished_at', user_id='user_id', geom='geom',
                        tz=None,
                        mapper={}):
    """
    warps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the staypoints to import
    started_at : str
        name of the column storing the starttime of the staypoints. The default is 'started_at'.
    finished_at : str
        name of the column storing the endtime of the staypoints. The default is 'finished_at'.
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    geom : str
        name of the column storing the geometry. The default is 'geom'.
    tz: str
        pytz compatible timezone string. If None UTC is assumed.
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the staypoints

    """

    columns = {started_at: 'started_at',
               finished_at: 'finished_at',
               user_id: 'user_id',
               geom: 'geom'}
    columns.update(mapper)

    stp = gdf.rename(columns=columns)
    stp = stp.set_geometry('geom')

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(stp[col]):
            stp[col] = localize_timestamp(dt_series=stp[col], pytz_tzinfo=tz, col_name=col)
        
    assert stp.as_staypoints
    return stp


def triplegs_from_gpd(gdf, started_at='started_at', finished_at='finished_at', user_id='user_id', geom='geometry',
                      tz=None, mapper={}):
    """
    warps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid line geometry, containing the triplegs to import
    started_at : str
        name of the column storing the starttime of the triplegs. The default is 'started_at'.
    finished_at : str
        name of the column storing the endtime of the triplegs. The default is 'finished_at'.
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    geom : str
        name of the column storing the geometry. The default is 'geom'.
    tz : str
        pytz compatible timezone string. If None UTC is assumed.
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the triplegs

    """

    columns = {started_at: 'started_at',
               finished_at: 'finished_at',
               user_id: 'user_id',
               geom: 'geom'}
    columns.update(mapper)

    tpl = gdf.rename(columns=columns)
    tpl = tpl.set_geometry('geom')

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(tpl[col]):
            tpl[col] = localize_timestamp(dt_series=tpl[col], pytz_tzinfo=tz, col_name=col)

    assert tpl.as_triplegs

    return tpl


def trips_from_gpd(gdf, started_at='started_at', finished_at='finished_at', user_id='user_id',
                   origin_staypoint_id='origin_staypoint_id', destination_staypoint_id='destination_staypoint_id',
                   tz=None, mapper={}):
    """
    warps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid geometry, containing the trips to import
    started_at : str
        name of the column storing the starttime of the staypoints. The default is 'started_at'.
    finished_at : str
        name of the column storing the endtime of the staypoints. The default is 'finished_at'.
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    origin_staypoint_id : str
        name of the column storing the staypoint_id of the start of the tripleg
    destination_staypoint_id : str
        name of the column storing the staypoint_id of the end of the tripleg
    tz : str
        pytz compatible timezone string. If None UTC is assumed.
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the trips

    """

    columns = {started_at: 'started_at',
               finished_at: 'finished_at',
               user_id: 'user_id',
               origin_staypoint_id: 'origin_staypoint_id',
               destination_staypoint_id: 'destination_staypoint_id'
               }
    columns.update(mapper)

    tps = gdf.rename(columns=columns)

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(tps[col]):
            tps[col] = localize_timestamp(dt_series=tps[col], pytz_tzinfo=tz, col_name=col)

    assert tps.as_trips
    return tps


def locations_from_gpd(gdf, user_id='user_id', center='center', mapper={}):
    """
    warps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the locations to import
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    center : str
        name of the column storing the geometry (Center of the location). The default is 'center'.
    tz : str
        pytz compatible timezone string. If None UTC is assumed.
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the locations

    """

    columns = {user_id: 'user_id',
               center: 'center'}
    columns.update(mapper)

    lcs = gdf.rename(columns=columns)
    lcs = lcs.set_geometry('center')

    assert lcs.as_locations

    return lcs


def tours_from_gpd(gdf, user_id='user_id', started_at='started_at', finished_at='finished_at',
                   origin_destination_location_id='origin_destination_location_id', journey='journey',
                   tz=None, mapper={}):
    """
    wraps the pd.rename function to simplify the import of GeoDataFrames

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the locations to import
    user_id : str
        name of the column storing the user_id. The default is 'user_id'.
    started_at : str
        name of the column storing the starttime of the staypoints. The default is 'started_at'.
    finished_at : str
        name of the column storing the endtime of the staypoints. The default is 'finished_at'.
    origin_destination_location_id :
        the name of the column storing the id of the location where the tour starts and ends. The default is 'origin_destination_location_id'.
    journey : str
        name of the column storing the information (bool) if the tour is a journey. The default is 'journey'.
    mapper : dict
        further columns that should be renamed.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the tours

    """

    columns = {user_id: 'user_id',
               started_at: 'tracked_at',
               finished_at: 'finished_at',
               origin_destination_location_id: 'origin_destination_location_id',
               journey: 'journey'}
    columns.update(mapper)

    trs = gdf.rename(columns=columns)

    # check and/or set timezone
    for col in ['started_at', 'finished_at']:
        if not pd.api.types.is_datetime64tz_dtype(trs[col]):
            trs[col] = localize_timestamp(dt_series=trs[col], pytz_tzinfo=tz, col_name=col)

    assert trs.as_tours
    return trs
