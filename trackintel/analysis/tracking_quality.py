import datetime

import pandas as pd
import numpy as np


def temporal_tracking_quality(source, granularity="all"):
    """
    Calculate per-user temporal tracking quality (temporal coverage).

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)
        The source dataframe to calculate temporal tracking quality.

    granularity : {"all", "day", "week", "weekday", "hour"}
        The level of which the tracking quality is calculated. The default "all" returns
        the overall tracking quality; "day" the tracking quality by days; "week" the quality
        by weeks; "weekday" the quality by day of the week (e.g, Mondays, Tuesdays, etc.) and
        "hour" the quality by hours.

    Returns
    -------
    quality: DataFrame
        A per-user per-granularity temporal tracking quality dataframe.

    Notes
    -----
    Requires at least the following columns:
    ``['user_id', 'started_at', 'finished_at']``
    which means the function supports trackintel ``staypoints``, ``triplegs``, ``trips`` and ``tours``
    datamodels and their combinations (e.g., staypoints and triplegs sequence).

    The temporal tracking quality is the ratio of tracking time and the total time extent. It is
    calculated and returned per-user in the defined ``granularity``. The possible time extents of
    the different granularities are different:

    - ``all`` considers the time between the latest "finished_at" and the earliest "started_at";
    - ``week`` considers the whole week (604800 sec)
    - ``day`` and ``weekday`` consider the whole day (86400 sec)
    - ``hour`` considers the whole hour (3600 sec).

    The tracking quality of each user is calculated based on his or her own tracking extent.
    For granularity = ``day`` or ``week``, the quality["day"] or quality["week"] column displays the
    time relative to the first record in the entire dataset.

    Examples
    --------
    >>> # calculate overall tracking quality of stps
    >>> temporal_tracking_quality(stps, granularity="all")
    >>> # calculate per-day tracking quality of stps and tpls sequence
    >>> temporal_tracking_quality(stps_tpls, granularity="day")
    """
    required_columns = ["user_id", "started_at", "finished_at"]
    if any([c not in source.columns for c in required_columns]):
        raise KeyError(
            "To successfully calculate the user-level tracking quality, "
            + "the source dataframe must have the columns [%s], but it has [%s]."
            % (", ".join(required_columns), ", ".join(source.columns))
        )

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

        # calculate per-user per-day raw tracking quality
        raw_quality = df.groupby(["user_id", "day"], as_index=False).apply(_get_tracking_quality_user, granularity)
        # add quality = 0 records
        quality = _get_all_quality(df, raw_quality, granularity)

    elif granularity == "week":
        # split records that span several days
        df = _split_overlaps(df, granularity="day")
        # get the tracked week relative to the first day
        start_date = df["started_at"].min().date()
        df["week"] = df["started_at"].apply(lambda x: (x.date() - start_date).days // 7)

        # calculate per-user per-week raw tracking quality
        raw_quality = df.groupby(["user_id", "week"], as_index=False).apply(_get_tracking_quality_user, granularity)
        # add quality = 0 records
        quality = _get_all_quality(df, raw_quality, granularity)

    elif granularity == "weekday":
        # split records that span several days
        df = _split_overlaps(df, granularity="day")

        # get the tracked week relative to the first day
        start_date = df["started_at"].min().date()
        df["week"] = df["started_at"].apply(lambda x: (x.date() - start_date).days // 7)
        # get the weekday
        df["weekday"] = df["started_at"].dt.weekday

        # calculate per-user per-weekday raw tracking quality
        raw_quality = df.groupby(["user_id", "weekday"], as_index=False).apply(_get_tracking_quality_user, granularity)
        # add quality = 0 records
        quality = _get_all_quality(df, raw_quality, granularity)

    elif granularity == "hour":
        # first do a day split to speed up the hour split
        df = _split_overlaps(df, granularity="day")
        df = _split_overlaps(df, granularity=granularity)

        # get the tracked day relative to the first day
        start_date = df["started_at"].min().date()
        df["day"] = df["started_at"].apply(lambda x: (x.date() - start_date).days)
        # get the hour
        df["hour"] = df["started_at"].dt.hour

        # calculate per-user per-hour raw tracking quality
        raw_quality = df.groupby(["user_id", "hour"], as_index=False).apply(_get_tracking_quality_user, granularity)
        # add quality = 0 records
        quality = _get_all_quality(df, raw_quality, granularity)

    else:
        raise AttributeError(
            f"granularity unknown. We only support ['all', 'day', 'week', 'weekday', 'hour']. You passed {granularity}"
        )

    return quality


def _get_all_quality(df, raw_quality, granularity):
    """
    Add tracking quality values for empty bins.

    raw_quality is calculated using `groupby` and does not report bins (=granularties) with
    quality = 0. This function adds these values.

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)

    raw_quality: DataFrame
        The calculated raw tracking quality directly from the groupby operations.

    granularity : {"all", "day", "weekday", "week", "hour"}
        Used for accessing the column in raw_quality.

    Returns
    -------
    quality: pandas.Series
        A pandas.Series object containing the tracking quality
    """
    all_users = df["user_id"].unique()
    all_granularity = np.arange(df[granularity].max() + 1)
    # construct array containing all user and granularity combinations
    all_combi = np.array(np.meshgrid(all_users, all_granularity)).T.reshape(-1, 2)
    # the records with no corresponding raw_quality is nan, and transformed into 0
    all_combi = pd.DataFrame(all_combi, columns=["user_id", granularity])
    quality = all_combi.merge(raw_quality, how="left", on=["user_id", granularity], validate="one_to_one")
    quality.fillna(0, inplace=True)
    return quality


def _get_tracking_quality_user(df, granularity="all"):
    """
    Tracking quality per-user per-granularity.

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)
        The source dataframe

    granularity : {"all", "day", "weekday", "week", "hour"}, default "all"
        Determines the extent of the tracking. "all" the entire tracking period,
        "day" and "weekday" a whole day, "week" a whole week, and "hour" a whole hour.

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
        # total seconds in a day
        extent = 60 * 60 * 24
    elif granularity == "weekday":
        # total seconds in an day * number of tracked weeks
        # (entries from multiple weeks may be grouped together)
        extent = 60 * 60 * 24 * (df["week"].max() - df["week"].min() + 1)
    elif granularity == "week":
        # total seconds in a week
        extent = 60 * 60 * 24 * 7
    elif granularity == "hour":
        # total seconds in an hour * number of tracked days
        # (entries from multiple days may be grouped together)
        extent = (60 * 60) * (df["day"].max() - df["day"].min() + 1)
    else:
        raise AttributeError(
            f"granularity unknown. We only support ['all', 'day', 'week', 'weekday', 'hour']. You passed {granularity}"
        )
    return pd.Series([tracked_duration / extent], index=["quality"])


def _split_overlaps(source, granularity="day"):
    """
    Split input df that have a duration of several days or hours.

    Parameters
    ----------
    source : GeoDataFrame (as trackintel datamodels)
        The source to perform the split

    granularity : {'day', 'hour'}, default 'day'
        The criteria of splitting. "day" splits records that have duration of several
        days and "hour" splits records that have duration of several hours.

    Returns
    -------
    GeoDataFrame (as trackintel datamodels)
        The GeoDataFrame object after the splitting
    """
    df = source.copy()
    change_flag = __get_split_index(df, granularity=granularity)

    # Iteratively split one day/hour from multi day/hour entries until no entry spans over multiple days/hours
    while change_flag.sum() > 0:

        # calculate new finished_at timestamp (00:00 midnight)
        finished_at_temp = df.loc[change_flag, "finished_at"].copy()
        if granularity == "day":
            df.loc[change_flag, "finished_at"] = df.loc[change_flag, "started_at"].apply(
                lambda x: x.replace(hour=23, minute=59, second=59) + datetime.timedelta(seconds=1)
            )
        elif granularity == "hour":
            df.loc[change_flag, "finished_at"] = df.loc[change_flag, "started_at"].apply(
                lambda x: x.replace(minute=59, second=59) + datetime.timedelta(seconds=1)
            )

        # create new entries with remaining timestamp
        new_df = df.loc[change_flag].copy()
        new_df.loc[change_flag, "started_at"] = df.loc[change_flag, "finished_at"]
        new_df.loc[change_flag, "finished_at"] = finished_at_temp

        df = df.append(new_df, ignore_index=True, sort=True)

        change_flag = __get_split_index(df, granularity=granularity)

    if "duration" in df.columns:
        df["duration"] = df["finished_at"] - df["started_at"]

    return df


def __get_split_index(df, granularity="day"):
    """
    Get the index that needs to be splitted.

    Parameters
    ----------
    df : GeoDataFrame (as trackintel datamodels)
        The source to perform the split.

    granularity : {'day', 'hour'}, default 'day'
        The criteria of spliting. "day" splits records that have duration of several
        days and "hour" splits records that have duration of several hours.

    Returns
    -------
    change_flag: pd.Series
        Boolean index indicating which records needs to be splitted
    """
    change_flag = df["started_at"].dt.date != (df["finished_at"] - pd.to_timedelta("1s")).dt.date
    if granularity == "hour":
        hour_flag = df["started_at"].dt.hour != (df["finished_at"] - pd.to_timedelta("1s")).dt.hour
        # union of day and hour change flag
        change_flag = change_flag | hour_flag

    return change_flag
