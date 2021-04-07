import pandas as pd

from trackintel.io.file import _localize_timestamp


def read_positionfixes_gpd(gdf, tracked_at="tracked_at", user_id="user_id", geom="geom", tz=None, mapper={}):
    """
    Read positionfixes from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the positionfixes to import

    tracked_at : str, default 'tracked_at'
        name of the column storing the timestamps.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    geom : str, default 'geom'
        name of the column storing the geometry.

    tz : str, optional
        pytz compatible timezone string. If None UTC will be assumed

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    pfs : GeoDataFrame (as trackintel positionfixes)
        A GeoDataFrame containing the positionfixes.

    Examples
    --------
    >>> trackintel.read_positionfixes_gpd(gdf, user_id='User', geom='geometry', tz='utc')
    """
    columns = {tracked_at: "tracked_at", user_id: "user_id", geom: "geom"}
    columns.update(mapper)

    pfs = gdf.rename(columns=columns)
    pfs = pfs.set_geometry("geom")

    # check and/or set timezone
    for col in ["tracked_at"]:
        if not pd.api.types.is_datetime64tz_dtype(pfs[col]):
            pfs[col] = _localize_timestamp(dt_series=pfs[col], pytz_tzinfo=tz, col_name=col)

    assert pfs.as_positionfixes
    return pfs


def read_staypoints_gpd(
    gdf, started_at="started_at", finished_at="finished_at", user_id="user_id", geom="geom", tz=None, mapper={}
):
    """
    Read staypoints from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the staypoints to import

    started_at : str, default 'started_at'
        name of the column storing the starttime of the staypoints.

    finished_at : str, default 'finished_at'
        name of the column storing the endtime of the staypoints.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    geom : str, default 'geom'
        name of the column storing the geometry.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    stps : GeoDataFrame (as trackintel staypoints)
        A GeoDataFrame containing the staypoints

    Examples
    --------
    >>> trackintel.read_staypoints_gpd(gdf, started_at='start_time', finished_at='end_time', tz='utc')
    """
    columns = {started_at: "started_at", finished_at: "finished_at", user_id: "user_id", geom: "geom"}
    columns.update(mapper)

    stps = gdf.rename(columns=columns)
    stps = stps.set_geometry("geom")

    # check and/or set timezone
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(stps[col]):
            stps[col] = _localize_timestamp(dt_series=stps[col], pytz_tzinfo=tz, col_name=col)

    assert stps.as_staypoints
    return stps


def read_triplegs_gpd(
    gdf, started_at="started_at", finished_at="finished_at", user_id="user_id", geom="geometry", tz=None, mapper={}
):
    """
    Read triplegs from GeoDataFrames.

    warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid line geometry, containing the triplegs to import.

    started_at : str, default 'started_at'
        name of the column storing the starttime of the triplegs.

    finished_at : str, default 'finished_at'
        name of the column storing the endtime of the triplegs.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    geom : str, default 'geom'
        name of the column storing the geometry.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    tpls : GeoDataFrame (as trackintel triplegs)
        A GeoDataFrame containing the triplegs

    Examples
    --------
    >>> trackintel.read_triplegs_gpd(gdf, user_id='User', geom='geometry', tz='utc')
    """
    columns = {started_at: "started_at", finished_at: "finished_at", user_id: "user_id", geom: "geom"}
    columns.update(mapper)

    tpls = gdf.rename(columns=columns)
    tpls = tpls.set_geometry("geom")

    # check and/or set timezone
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(tpls[col]):
            tpls[col] = _localize_timestamp(dt_series=tpls[col], pytz_tzinfo=tz, col_name=col)

    assert tpls.as_triplegs
    return tpls


def read_trips_gpd(
    gdf,
    started_at="started_at",
    finished_at="finished_at",
    user_id="user_id",
    origin_staypoint_id="origin_staypoint_id",
    destination_staypoint_id="destination_staypoint_id",
    tz=None,
    mapper={},
):
    """
    Read trips from GeoDataFrames/DataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames (DataFrames).

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        GeoDataFrame/DataFrame containing the trips to import.

    started_at : str, default 'started_at'
        name of the column storing the starttime of the staypoints.

    finished_at : str, default 'finished_at'
        name of the column storing the endtime of the staypoints.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    origin_staypoint_id : str, default 'origin_staypoint_id'
        name of the column storing the staypoint_id of the start of the tripleg

    destination_staypoint_id : str, default 'destination_staypoint_id'
        name of the column storing the staypoint_id of the end of the tripleg

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    trips : GeoDataFrame/DataFrame (as trackintel trips)
        A GeoDataFrame/DataFrame containing the trips.

    Examples
    --------
    >>> trackintel.read_trips_gpd(df, tz='utc')
    """
    columns = {
        started_at: "started_at",
        finished_at: "finished_at",
        user_id: "user_id",
        origin_staypoint_id: "origin_staypoint_id",
        destination_staypoint_id: "destination_staypoint_id",
    }
    columns.update(mapper)

    trips = gdf.rename(columns=columns)

    # check and/or set timezone
    for col in ["started_at", "finished_at"]:
        if not pd.api.types.is_datetime64tz_dtype(trips[col]):
            trips[col] = _localize_timestamp(dt_series=trips[col], pytz_tzinfo=tz, col_name=col)

    assert trips.as_trips
    return trips


def read_locations_gpd(gdf, user_id="user_id", center="center", mapper={}):
    """
    Read locations from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the locations to import.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    center : str, default 'center'
        name of the column storing the geometry (Center of the location).

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    locs : GeoDataFrame (as trackintel locations)
        A GeoDataFrame containing the locations.

    Examples
    --------
    >>> trackintel.read_locations_gpd(df, user_id='User', center='geometry')
    """
    columns = {user_id: "user_id", center: "center"}
    columns.update(mapper)

    locs = gdf.rename(columns=columns)
    locs = locs.set_geometry("center")

    assert locs.as_locations
    return locs


def read_tours_gpd(
    gdf,
    user_id="user_id",
    started_at="started_at",
    finished_at="finished_at",
    origin_destination_location_id="origin_destination_location_id",
    journey="journey",
    tz=None,
    mapper={},
):
    """
    Read tours from GeoDataFrames.

    Wraps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the locations to import.

    user_id : str, default 'user_id'
        name of the column storing the user_id.

    started_at : str, default 'started_at'
        name of the column storing the starttime of the staypoints.

    finished_at : str, default 'finished_at'
        name of the column storing the endtime of the staypoints.

    origin_destination_location_id : str, default 'origin_destination_location_id'
        the name of the column storing the id of the location where the tour starts and ends.

    journey : str, default 'journey'
        name of the column storing the information (bool) if the tour is a journey.

    mapper : dict, optional
        further columns that should be renamed.

    Returns
    -------
    trs : GeoDataFrame (as trackintel tours)
        A GeoDataFrame containing the tours
    """
    # columns = {user_id: 'user_id',
    #            started_at: 'tracked_at',
    #            finished_at: 'finished_at',
    #            origin_destination_location_id: 'origin_destination_location_id',
    #            journey: 'journey'}
    # columns.update(mapper)

    # trs = gdf.rename(columns=columns)

    # # check and/or set timezone
    # for col in ['started_at', 'finished_at']:
    #     if not pd.api.types.is_datetime64tz_dtype(trs[col]):
    #         trs[col] = localize_timestamp(dt_series=trs[col], pytz_tzinfo=tz, col_name=col)

    # assert trs.as_tours
    # return trs
    pass
