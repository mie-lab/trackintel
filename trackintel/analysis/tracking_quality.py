import warnings

import numpy as np
import pandas as pd


def temporal_tracking_quality(source, granularity="all"):
    """
    Calculate per-user temporal tracking quality (temporal coverage).

    Parameters
    ----------
    df : Trackintel class
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
    calculated and returned per-user in the defined ``granularity``. The time extents
    and the columns for the returned ``quality`` df for different ``granularity`` are:

    - ``all``:
        - `time extent`: between the latest "finished_at" and the earliest "started_at" for each user.
        - `columns`: ``['user_id', 'quality']``.
    - ``week``:
        - `time extent`: the whole week (604800 sec) for each user.
        - `columns`: ``['user_id', 'week_monday', 'quality']``.
    - ``day``:
        - `time extent`: the whole day (86400 sec) for each user
        - `columns`: ``['user_id', 'day', 'quality']``
    - ``weekday``
        - `time extent`: the whole day (86400 sec) * number of tracked weeks for each user for each user
        - `columns`: ``['user_id', 'weekday', 'quality']``
    - ``hour``:
        - `time extent`: the whole hour (3600 sec) * number of tracked days for each user
        - `columns`: ``['user_id', 'hour', 'quality']``

    Examples
    --------
    >>> # calculate overall tracking quality of staypoints
    >>> temporal_tracking_quality(sp, granularity="all")
    >>> # calculate per-day tracking quality of sp and tpls sequence
    >>> temporal_tracking_quality(sp_tpls, granularity="day")
    """
    required_columns = ["user_id", "started_at", "finished_at"]
    if any([c not in source.columns for c in required_columns]):
        raise KeyError(
            "To successfully calculate the user-level tracking quality, "
            f"the source dataframe must have the columns {required_columns}, but it has [{', '.join(source.columns)}]."
        )

    df = source.copy()
    df.reset_index(inplace=True)

    # filter out records with duration <= 0
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    df = df.loc[df["duration"] > 0].copy()
    # ensure proper handle of empty dataframes
    if len(df) == 0:
        warnings.warn("The input dataframe does not contain any record with positive duration. Please check.")
        return None

    if granularity == "all":
        quality = df.groupby("user_id", as_index=False).apply(_get_tracking_quality_user, granularity)
        return quality

    # split records that span several days
    df = _split_overlaps(df, granularity="day")
    if granularity == "day":
        grouper = pd.Grouper(key="started_at", freq="D")
        column_name = "day"

    elif granularity == "week":
        grouper = pd.Grouper(key="started_at", freq="W")
        column_name = "week_monday"

    elif granularity == "weekday":
        # get the tracked week relative to the first day
        start_date = df["started_at"].min().floor(freq="D")
        df["week"] = ((df["started_at"] - start_date)).dt.days // 7

        grouper = df["started_at"].dt.weekday
        column_name = "weekday"

    elif granularity == "hour":
        df = _split_overlaps(df, granularity="hour")
        # get the tracked day relative to the first day
        start_date = df["started_at"].min().floor(freq="D")
        df["day"] = (df["started_at"] - start_date).dt.days

        grouper = df["started_at"].dt.hour
        column_name = "hour"

    else:
        raise AttributeError(
            f"granularity unknown. We only support ['all', 'day', 'week', 'weekday', 'hour']. You passed {granularity}"
        )

    # calculate per-user per-grouper tracking quality
    quality = df.groupby(["user_id", grouper]).apply(_get_tracking_quality_user, granularity).reset_index()

    # rename and reorder
    quality.rename(columns={"started_at": column_name}, inplace=True)
    quality = quality[["user_id", column_name, "quality"]]

    return quality


def _get_tracking_quality_user(df, granularity="all"):
    """
    Tracking quality per-user per-granularity.

    Parameters
    ----------
    df : Trackintel class
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
    source : Trackintel class
        The GeoDataFrame to perform the split on.

    granularity : {'day', 'hour'}, default 'day'
        The criteria of splitting. "day" splits records that have duration of several
        days and "hour" splits records that have duration of several hours.

    Returns
    -------
    Trackintel class
        The input object after the splitting
    """
    freq = "H" if granularity == "hour" else "D"
    gdf = source.copy()
    gdf[["started_at", "finished_at"]] = gdf.apply(_get_times, axis="columns", result_type="expand", freq=freq)
    # must call DataFrame.explode directly because GeoDataFrame.explode cannot be used on multiple columns
    gdf = pd.DataFrame.explode(gdf, ["started_at", "finished_at"], ignore_index=True)
    if "duration" in gdf.columns:
        gdf["duration"] = gdf["finished_at"] - gdf["started_at"]
    return gdf


def _get_times(row, freq="D"):
    """
    Returns the times for splitting range (start-finish) at frequency borders.

    Use it with `.apply()` for a single row of a DataFrame. Set result_type="expand".

    Parameters
    ----------
    row : Series
        Row of dataframe with columns ["started_at", "finished_at"].

    freq : str or DateOffset, default 'D'
        Pandas frequency string.

    Returns
    -------
    Tuple of lists
        Tuple of (start, end) times.
    """
    result = []
    if row["started_at"] != row["started_at"].ceil(freq):
        result.append(row["started_at"])  # is not on border -> not included in date_range
    result.extend(pd.date_range(row["started_at"].ceil(freq), row["finished_at"], freq=freq).to_list())
    if (row["finished_at"] != result[-1]) or (len(result) == 1):  # len check for started_at == finished_at
        result.append(row["finished_at"])  # is not on border -> not included in data_range
    return result[:-1], result[1:]
