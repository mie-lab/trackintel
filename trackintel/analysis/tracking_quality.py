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
        quality = df.groupby("user_id", as_index=False).apply(
            _get_tracking_quality_user, granularity, include_groups=False
        )
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
        grouper = df["started_at"].dt.weekday
        column_name = "weekday"

    elif granularity == "hour":
        df = _split_overlaps(df, granularity="hour")
        grouper = df["started_at"].dt.hour
        column_name = "hour"

    else:
        raise ValueError(
            f"granularity unknown. We only support ['all', 'day', 'week', 'weekday', 'hour']. You passed {granularity}"
        )

    start_date = df["started_at"].min().floor(freq="D")
    # calculate per-user per-grouper tracking quality
    quality = (
        df.groupby(["user_id", grouper])[["started_at", "finished_at"]]
        .apply(_get_tracking_quality_user, start_date, granularity)
        .reset_index()
    )

    # rename and reorder
    quality.rename(columns={"started_at": column_name}, inplace=True)
    quality = quality[["user_id", column_name, "quality"]]

    return quality


def _get_tracking_quality_user(df, start_date, granularity="all"):
    """
    Tracking quality per-user per-granularity.

    Parameters
    ----------
    df : Trackintel class
        The source dataframe

    start_date: pd.Timestamp
        When measurement started, used to calculate in which weekday or week the measurement lies.

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
        df["week"] = ((df["started_at"] - start_date)).dt.days // 7
        extent = 60 * 60 * 24 * (df["week"].max() - df["week"].min() + 1)
    elif granularity == "week":
        # total seconds in a week
        extent = 60 * 60 * 24 * 7
    elif granularity == "hour":
        # total seconds in an hour * number of tracked days
        # (entries from multiple days may be grouped together)
        df["day"] = (df["started_at"] - start_date).dt.days
        extent = (60 * 60) * (df["day"].max() - df["day"].min() + 1)
    else:
        raise ValueError(
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
    freq = "h" if granularity == "hour" else "D"
    gdf = source.copy()
    tz = getattr(gdf["started_at"].dtype, "tz", None)

    if tz is not None and str(tz) != "UTC":
        # borders depend on the calendar (e.g. DST days are not 24h long),
        # keep the row-wise splitting, but only for records that contain a
        # frequency border strictly inside (started_at, finished_at)
        next_border = gdf["started_at"].dt.floor(freq) + pd.tseries.frequencies.to_offset(freq)
        crosses = (next_border < gdf["finished_at"]).to_numpy()
        if crosses.any():
            split_times = gdf.loc[crosses].apply(_get_times, axis="columns", result_type="expand", freq=freq)
            starts = gdf["started_at"].to_numpy(dtype=object)
            finishes = gdf["finished_at"].to_numpy(dtype=object)
            starts[crosses] = split_times[0].to_numpy()
            finishes[crosses] = split_times[1].to_numpy()
            gdf["started_at"] = starts
            gdf["finished_at"] = finishes
            # DataFrame.explode directly because GeoDataFrame.explode cannot take multiple columns
            gdf = pd.DataFrame.explode(gdf, ["started_at", "finished_at"], ignore_index=True)
        else:
            gdf = gdf.reset_index(drop=True)
        if "duration" in gdf.columns:
            gdf["duration"] = gdf["finished_at"] - gdf["started_at"]
        return gdf

    # naive or UTC timestamps: borders are uniform, so the segments of every
    # record can be computed at once with integer nanosecond arithmetic
    f = pd.Timedelta(1, unit=freq).value
    orig_dtype = gdf["started_at"].dtype
    if tz is None:
        s = gdf["started_at"].to_numpy().astype("datetime64[ns]").astype("int64")
        e = gdf["finished_at"].to_numpy().astype("datetime64[ns]").astype("int64")
    else:
        s = gdf["started_at"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
        e = gdf["finished_at"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    next_border = (s // f) * f + f
    # number of borders strictly inside (started_at, finished_at)
    borders = np.where(e > next_border, (e - next_border - 1) // f + 1, 0)
    segments = borders + 1
    row_idx = np.repeat(np.arange(len(gdf)), segments)
    # position of each output row within its record
    k = np.arange(len(row_idx)) - np.repeat(np.cumsum(segments) - segments, segments)
    new_s = np.where(k == 0, s[row_idx], next_border[row_idx] + (k - 1) * f)
    new_e = np.where(k == segments[row_idx] - 1, e[row_idx], next_border[row_idx] + k * f)
    gdf = gdf.take(row_idx).reset_index(drop=True)
    new_s = pd.Series(new_s.astype("datetime64[ns]"), index=gdf.index)
    new_e = pd.Series(new_e.astype("datetime64[ns]"), index=gdf.index)
    if tz is not None:
        new_s = new_s.dt.tz_localize(tz)
        new_e = new_e.dt.tz_localize(tz)
    gdf["started_at"] = new_s.astype(orig_dtype)
    gdf["finished_at"] = new_e.astype(orig_dtype)
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
