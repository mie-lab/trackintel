import datetime

import pandas as pd


def temporal_tracking_quality(source, granularity="all"):
    """Calculate per-user temporal tracking quality.

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)
        The source dataframe to perform the spatial filtering

    granularity : str, {'all', 'day', 'hour'}, default 'day'
        - 'all' : overall tracking quality
        - 'day' : tracking quality by days
        - 'hour': tracking quality by hours

    Returns
    -------
    DataFrame
        A per-user temporal tracking quality dataframe.
    """
    df = source.copy()
    df.reset_index(inplace=True)
    if granularity == "all":
        quality = df.groupby("user_id", as_index=False).apply(_get_tracking_quality_user, granularity)
    elif granularity == "day":
        # split records that span several days
        df = _split_overlaps(df, granularity=granularity)
        # get the tracked day relative to the first day
        start_date = df["started_at"].min().date()
        df["day"] = df["started_at"].apply(lambda x: (x.date() - start_date).days)
        # calculate per-user per-day tracking quality
        quality = df.groupby(["user_id", "day"], as_index=False).apply(_get_tracking_quality_user, granularity)
    elif granularity == "hour":
        # split records that span several hours
        df = _split_overlaps(df, granularity="hour")
        # get the tracked day relative to the first day
        start_date = df["started_at"].min().date()
        df["day"] = df["started_at"].apply(lambda x: (x.date() - start_date).days)
        # get the hour
        df["hour"] = df["started_at"].dt.hour
        # calculate per-user per-hour tracking quality
        quality = df.groupby(["user_id", "hour"], as_index=False).apply(_get_tracking_quality_user, granularity)
    else:
        raise AttributeError(f"granularity unknown. We only support ['all', 'day', 'hour']. You passed {granularity}")
    return quality


def _get_tracking_quality_user(df, granularity="all"):
    """Tracking quality per-user per-granularity.

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)
        The source dataframe

    granularity : str, {'all', 'day', 'hour'}, default 'all'
        Determines the extent of the tracking
        - 'all' : the entire tracking period
        - 'day' : a whole day
        - 'hour': a whole hour

    Returns
    -------
    pandas.Series
        A pandas.Series object containing the tracking quality
    """
    tracked_duration = (df["finished_at"] - df["started_at"]).dt.total_seconds().sum()
    if granularity == "all":
        # the whole tracking period
        extent = (df["finished_at"].max() - df["started_at"].min()).total_seconds()
    elif granularity == "day":
        # total seconds of a day
        extent = 60 * 60 * 24 - 1
    elif granularity == "hour":
        # total seconds of a day times number of different days
        extent = (60 * 60 - 1) * len(df["day"].unique())
        print(df[["started_at", "finished_at"]])
        print(extent)
    return pd.Series([tracked_duration / extent], index=["quality"])


def _split_overlaps(source, granularity="day"):
    """Split input df that have a duration of several days or hours.

    Parameters
    ----------
    source : GeoDataFrame (as trackintel datamodels)
        The source to perform the split

    granularity : str, {'day', 'hour'}, default 'day'
        - 'day' : split records that have duration of several days
        - 'hour': split records that have duration of several hours

    Returns
    -------
    GeoDataFrame (as original trackintel datamodels)
        The GeoDataFrame object after the spliting
    """
    df = source.copy()
    if granularity == "day":
        change_flag = df["started_at"].dt.date != df["finished_at"].dt.date
    elif granularity == "hour":
        change_flag = df["started_at"].dt.hour != df["finished_at"].dt.hour
    i = 0

    while change_flag.sum() > 0:

        # calculate new finished_at timestamp (1sec before midnight)
        finished_at_temp = df.loc[change_flag, "finished_at"].copy()
        if granularity == "day":
            df.loc[change_flag, "finished_at"] = df.loc[change_flag, "started_at"].apply(
                lambda x: x.replace(hour=23, minute=59, second=59)
            )
        elif granularity == "hour":
            df.loc[change_flag, "finished_at"] = df.loc[change_flag, "started_at"].apply(
                lambda x: x.replace(minute=59, second=59)
            )

        # create new entries with remaining timestamp
        new_df = df.loc[change_flag].copy()
        new_df.loc[change_flag, "started_at"] = df.loc[change_flag, "finished_at"] + datetime.timedelta(seconds=1)
        new_df.loc[change_flag, "finished_at"] = finished_at_temp

        df = df.append(new_df, ignore_index=True, sort=True)
        if granularity == "day":
            change_flag = df["started_at"].dt.date != df["finished_at"].dt.date
        elif granularity == "hour":
            change_flag = df["started_at"].dt.hour != df["finished_at"].dt.hour
        i = i + 1

    return df