import warnings
import pandas as pd
import geopandas as gpd
import pytz

from trackintel import Positionfixes, Staypoints, Triplegs, Locations, Trips, Tours


def read_positionfixes_gpd(
    gdf, tracked_at="tracked_at", user_id="user_id", geom_col=None, crs=None, tz=None, mapper=None
):
    """
    Read positionfixes from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the positionfixes to import

    tracked_at : str, default 'tracked_at'
        Name of the column storing the timestamps.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    geom_col : str, optional
        Name of the column storing the geometry. If None assumes geometry is already set.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string.

    tz : str, optional
        pytz compatible timezone string. If None UTC will be assumed

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    pfs : Positionfixes
        A GeoDataFrame containing the positionfixes.

    Examples
    --------
    >>> trackintel.read_positionfixes_gpd(gdf, user_id='User', geom_col='geom', tz='utc')
    """
    columns = {tracked_at: "tracked_at", user_id: "user_id"}
    if mapper is not None:
        columns.update(mapper)

    pfs = _trackintel_model(gdf, columns, geom_col, crs, ["tracked_at"], tz)
    return Positionfixes(pfs)


def read_staypoints_gpd(
    gdf,
    started_at="started_at",
    finished_at="finished_at",
    user_id="user_id",
    geom_col=None,
    crs=None,
    tz=None,
    mapper=None,
):
    """
    Read staypoints from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the staypoints to import

    started_at : str, default 'started_at'
        Name of the column storing the starttime of the staypoints.

    finished_at : str, default 'finished_at'
        Name of the column storing the endtime of the staypoints.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    geom_col : str
        Name of the column storing the geometry. If None assumes geometry is already set.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    sp : Staypoints
        A GeoDataFrame containing the staypoints

    Examples
    --------
    >>> trackintel.read_staypoints_gpd(gdf, started_at='start_time', finished_at='end_time', tz='utc')
    """
    columns = {started_at: "started_at", finished_at: "finished_at", user_id: "user_id"}
    if mapper is not None:
        columns.update(mapper)

    sp = _trackintel_model(gdf, columns, geom_col, crs, ["started_at", "finished_at"], tz)

    return Staypoints(sp)


def read_triplegs_gpd(
    gdf,
    started_at="started_at",
    finished_at="finished_at",
    user_id="user_id",
    geom_col=None,
    crs=None,
    tz=None,
    mapper=None,
):
    """
    Read triplegs from GeoDataFrames.

    warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid line geometry, containing the triplegs to import.

    started_at : str, default 'started_at'
        Name of the column storing the starttime of the triplegs.

    finished_at : str, default 'finished_at'
        Name of the column storing the endtime of the triplegs.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    geom_col : str, optional
        Name of the column storing the geometry. If None assumes geometry is already set.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    tpls : Triplegs
        A GeoDataFrame containing the triplegs

    Examples
    --------
    >>> trackintel.read_triplegs_gpd(gdf, user_id='User', geom_col='geom', tz='utc')
    """
    columns = {started_at: "started_at", finished_at: "finished_at", user_id: "user_id"}
    if mapper is not None:
        columns.update(mapper)

    tpls = _trackintel_model(gdf, columns, geom_col, crs, ["started_at", "finished_at"], tz)
    return Triplegs(tpls)


def read_trips_gpd(
    gdf,
    started_at="started_at",
    finished_at="finished_at",
    user_id="user_id",
    origin_staypoint_id="origin_staypoint_id",
    destination_staypoint_id="destination_staypoint_id",
    geom_col=None,
    crs=None,
    tz=None,
    mapper=None,
):
    """
    Read trips from GeoDataFrames/DataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames (DataFrames).

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        (Geo)DataFrame containing the trips to import.

    started_at : str, default 'started_at'
        Name of the column storing the starttime of the staypoints.

    finished_at : str, default 'finished_at'
        Name of the column storing the endtime of the staypoints.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    origin_staypoint_id : str, default 'origin_staypoint_id'
        Name of the column storing the staypoint_id of the start of the tripleg.

    destination_staypoint_id : str, default 'destination_staypoint_id'
        Name of the column storing the staypoint_id of the end of the tripleg

    geom_col : str, optional
        Name of the column storing the geometry. If None assumes has no geometry!

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string. Ignored if "geom_col" is None.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    trips : Trips

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
    if mapper is not None:
        columns.update(mapper)

    trips = _trackintel_model(gdf, columns, geom_col, crs, ["started_at", "finished_at"], tz)

    return Trips(trips)


def read_locations_gpd(gdf, user_id="user_id", center="center", extent=None, crs=None, mapper=None):
    """
    Read locations from GeoDataFrames.

    Warps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with valid point geometry, containing the locations to import.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    center : str, default 'center'
        Name of the column storing the geometry (center of the location).

    extent : str, optional
        Name of the column storing the additionaly geometry (extent of location).

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string.

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    locs : Locations

    Examples
    --------
    >>> trackintel.read_locations_gpd(df, user_id='User', center='geometry')
    """
    columns = {user_id: "user_id", center: "center"}
    if extent is not None:
        columns[extent] = "extent"
    if mapper is not None:
        columns.update(mapper)

    locs = _trackintel_model(gdf, columns, "center", crs)

    if extent is not None:
        locs["extent"] = gpd.GeoSeries(locs["extent"])

    return Locations(locs)


def read_tours_gpd(
    gdf,
    user_id="user_id",
    started_at="started_at",
    finished_at="finished_at",
    tz=None,
    mapper=None,
):
    """
    Read tours from GeoDataFrames.

    Wraps the pd.rename function to simplify the import of GeoDataFrames.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing the tours to import.

    user_id : str, default 'user_id'
        Name of the column storing the user_id.

    started_at : str, default 'started_at'
        Name of the column storing the start time of the tours.

    finished_at : str, default 'finished_at'
        Name of the column storing the end time of the tours.

    tz : str, optional
        pytz compatible timezone string. If None UTC is assumed.

    mapper : dict, optional
        Further columns that should be renamed.

    Returns
    -------
    tours : Tours
    """
    columns = {
        user_id: "user_id",
        started_at: "started_at",
        finished_at: "finished_at",
    }
    if mapper is not None:
        columns.update(mapper)

    tours = _trackintel_model(gdf, set_names=columns, tz_cols=["started_at", "finished_at"], tz=tz)
    return Tours(tours)


def _trackintel_model(gdf, set_names=None, geom_col=None, crs=None, tz_cols=None, tz=None):
    """Help function to assure the trackintel model on a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame

    set_names : dict, optional
        Renaming dictionary for the columns of the GeoDataFrame.

    set_geometry : str, optional
        Set geometry of GeoDataFrame.

    crs : pyproj.crs or str, optional
        Set coordinate reference system. The value can be anything accepted
        by pyproj.CRS.from_user_input(), such as an authority string
        (eg "EPSG:4326") or a WKT string.

    tz_cols : list, optional
        List of timezone aware datetime columns.

    tz : str, optional
        pytz compatible timezone string. If None UTC will be assumed

    Returns
    -------
    gdf : GeoDataFrame
        The input GeoDataFrame transformed to match the trackintel format.
    """
    if set_names is not None:
        gdf = gdf.rename(columns=set_names)

    if tz_cols is not None:
        for col in tz_cols:
            if not isinstance(gdf[col].dtype, pd.DatetimeTZDtype):
                gdf[col] = _localize_timestamp(dt_series=gdf[col], pytz_tzinfo=tz, col_name=col)

    # If is not GeoDataFrame and no geom_col is set end early.
    # That allows us to handle DataFrames and GeoDataFrames in one function.
    if not isinstance(gdf, gpd.GeoDataFrame) and geom_col is None:
        return gdf

    if geom_col is not None:
        gdf = gdf.set_geometry(geom_col)
    else:
        try:
            gdf.geometry
        except AttributeError:
            raise AttributeError("GeoDataFrame has no geometry, set it with keyword argument.")

    if crs is not None:
        gdf = gdf.set_crs(crs)

    return gdf


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
        warnings.warn(f"Assuming UTC timezone for column {col_name}")
        pytz_tzinfo = "utc"

    def localize(ts, tz):
        """Localize ts if tz is not set else leave it be"""
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            return ts
        return pd.Timestamp.tz_localize(ts, tz)

    # localize all datetimes without a timezone
    dt_series = dt_series.apply(localize, tz=pytz_tzinfo)
    # create a Timeseries (utc=False will create warning)
    dt_series = pd.to_datetime(dt_series, utc=True)
    # convert it back to right tz
    return dt_series.dt.tz_convert(pytz_tzinfo)
