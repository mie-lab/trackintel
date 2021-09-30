import warnings
import numpy as np
import pandas as pd
import trackintel as ti


def location_identifier(staypoints, method="FREQ", pre_filter=True, **pre_filter_kwargs):
    """Assign "home" and "work" activity label for each user with different methods.

    Parameters
    ----------
    staypoints : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    method : {'FREQ', 'OSNA'}, default "FREQ"
        'FREQ': Generate an activity label per user by assigning the most visited location the label "home"
        and the second most visited location the label "work". The remaining locations get no label.
        'OSNA': Use weekdays data divided in three time frames ["rest", "work", "leisure"]. Finds most popular
        home location for timeframes "rest" and "leisure" and most popular "work" location for "work" timeframe.

    pre_filter : bool, default True
        Prefiltering the staypoints to exclude locations with not enough data.
        The filter function can also be accessed via `pre_filter_locations`.

    pre_filter_kwargs : dict
        Kwargs to hand to `pre_filter_locations` if used. See function for more informations.

    Returns
    -------
    sp: Geodataframe (as trackintel staypoints)
        With additional column `activity label` assigning one of three activity labels {'home', 'work', None}.

    Note
    ----
    The methods are adapted from [1]. The original algorithms count the distinct hours at a
    location as the home location is derived from geo-tagged tweets. We directly sum the time
    spent at a location as our data model includes that.

    References
    ----------
    [1] Chen, Qingqing, and Ate Poorthuis. 2021.
    ‘Identifying Home Locations in Human Mobility Data: An Open-Source R Package for Comparison and Reproducibility’.
    International Journal of Geographical Information Science 0 (0): 1–24.
    https://doi.org/10.1080/13658816.2021.1887489.

    Examples
    --------
    >>> from ti.analysis.location_identification import location_identifier
    >>> location_identifier(staypoints, pre_filter=True, method="FREQ")
    """
    # assert validity of staypoints
    staypoints.as_staypoints

    sp = staypoints.copy()
    if "location_id" not in sp.columns:
        raise KeyError(
            (
                "To derive activity labels the GeoDataFrame (as trackintel staypoints) must have a column "
                f"named 'location_id' but it has [{', '.join(sp.columns)}]"
            )
        )
    if pre_filter:
        f = pre_filter_locations(sp, **pre_filter_kwargs)
    else:
        f = pd.Series(np.full(len(sp.index), True))

    if method == "FREQ":
        method_val = freq_method(sp[f], "home", "work")
    elif method == "OSNA":
        method_val = osna_method(sp[f])
    else:
        raise ValueError(f"Method {method} does not exist.")

    sp.loc[f, "activity_label"] = method_val["activity_label"]
    return sp


def pre_filter_locations(
    staypoints,
    agg_level="user",
    thresh_sp=10,
    thresh_loc=10,
    thresh_sp_at_loc=10,
    thresh_loc_time="1h",
    thresh_loc_period="5h",
):
    """Filter locations and user out that have not enough data to do a proper analysis.

    To disable a specific filter parameter set it to zero.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        Staypoints with the column "location_id".

    agg_level: {"user", "dataset"}, default "user"
        The level of aggregation when filtering locations. 'user' : locations are filtered per-user;
        'dataset' : locations are filtered over the whole dataset.

    thresh_sp : int, default 10
        Minimum staypoints a user must have to be included.

    thresh_loc : int, default 10
        Minimum locations a user must have to be included.

    thresh_sp_at_loc : int, default 10
        Minimum number of staypoints at a location must have to be included.

    thresh_loc_time : str or pd.Timedelta, default "1h"
        Minimum sum of durations that was spent at location to be included.
        If str must be parsable by pd.to_timedelta.

    thresh_loc_period : str or pd.Timedelta, default "5h"
        Minimum timespan of first to last visit at a location to be included.
        If str must be parsable by pd.to_timedelta.

    Returns
    -------
    total_filter: pd.Series
        Boolean series containing the filter as a mask.

    Examples
    --------
    >>> from ti.analysis.location_identification import pre_filter_locations
    >>> mask = pre_filter_locations(staypoints)
    >>> staypoints = staypoints[mask]
    """
    # assert validity of staypoints
    staypoints.as_staypoints

    sp = staypoints.copy()
    if isinstance(thresh_loc_time, str):
        thresh_loc_time = pd.to_timedelta(thresh_loc_time)
    if isinstance(thresh_loc_period, str):
        thresh_loc_period = pd.to_timedelta(thresh_loc_period)

    # filtering users
    user = sp.groupby("user_id").nunique()
    user_sp = user["started_at"] >= thresh_sp  # every staypoint should have a started_at -> count
    user_loc = user["location_id"] >= thresh_loc
    user_filter_agg = user_sp & user_loc
    user_filter_agg.rename("user_filter", inplace=True)  # rename for merging
    user_filter = pd.merge(sp["user_id"], user_filter_agg, left_on="user_id", right_index=True)["user_filter"]

    # filtering locations
    sp["duration"] = sp["finished_at"] - sp["started_at"]
    if agg_level == "user":
        groupby_loc = ["user_id", "location_id"]
    elif agg_level == "dataset":
        groupby_loc = ["location_id"]
    else:
        raise ValueError(f"Unknown agg_level '{agg_level}' use instead {{'user', 'dataset'}}.")
    loc = sp.groupby(groupby_loc).agg({"started_at": [min, "count"], "finished_at": max, "duration": sum})
    loc.columns = loc.columns.droplevel(0)  # remove possible multi-index
    loc.rename(columns={"min": "started_at", "max": "finished_at", "sum": "duration"}, inplace=True)
    # period for maximal time span first visit - last visit.
    # duration for effective time spent at location summed up.
    loc["period"] = loc["finished_at"] - loc["started_at"]
    loc_sp = loc["count"] >= thresh_sp_at_loc
    loc_time = loc["duration"] >= thresh_loc_time
    loc_period = loc["period"] >= thresh_loc_period
    loc_filter_agg = loc_sp & loc_time & loc_period
    loc_filter_agg.rename("loc_filter", inplace=True)  # rename for merging
    loc_filter = pd.merge(sp[groupby_loc], loc_filter_agg, how="left", left_on=groupby_loc, right_index=True)
    loc_filter = loc_filter["loc_filter"]

    total_filter = user_filter & loc_filter

    return total_filter


def freq_method(staypoints, *labels):
    """Generate an activity label per user.

    Assigning the most visited location the label "home" and the second most visited location the label "work".
    The remaining locations get no label.

    Labels can also be given as arguments.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        Staypoints with the column "location_id".

    labels : collection of str, default ("home", "work")
        Labels in decreasing time of activity.

    Returns
    -------
    sp: GeoDataFrame (as trackintel staypoints)
        The input staypoints with additional column "activity_label".

    Examples
    --------
    >>> from ti.analysis.location_identification import freq_method
    >>> staypoints = freq_method(staypoints, "home", "work")
    """
    sp = staypoints.copy()
    if not labels:
        labels = ("home", "work")
    for name, group in sp.groupby("user_id"):
        if "duration" not in group.columns:
            group["duration"] = group["finished_at"] - group["started_at"]
        # pandas keeps inner order of groups
        sp.loc[sp["user_id"] == name, "activity_label"] = _freq_transform(group, *labels)
    return sp


def _freq_transform(group, *labels):
    """Transform function that assigns the longest (duration) visited locations the labels in order.

    Parameters
    ----------
    group : pd.DataFrame
        Should have columns "location_id" and "duration".

    Returns
    -------
    pd.Series
        dtype : object
    """
    group_agg = group.groupby("location_id").agg({"duration": sum})
    group_agg["activity_label"] = _freq_assign(group_agg["duration"], *labels)
    group_merge = pd.merge(
        group["location_id"], group_agg["activity_label"], how="left", left_on="location_id", right_index=True
    )
    return group_merge["activity_label"]


def _freq_assign(duration, *labels):
    """Assign k labels to k longest durations the rest is `None`.

    Parameters
    ----------
    duration : pd.Series

    Returns
    -------
    np.array
        dtype : object
    """
    kth = (-duration).argsort()[: len(labels)]  # if inefficient use partial sort.
    label_array = np.full(len(duration), fill_value=None)
    labels = labels[: len(kth)]  # if provided with more labels than entries.
    label_array[kth] = labels
    return label_array


def osna_method(staypoints):
    """Find "home" location for timeframes "rest" and "leisure" and "work" location for "work" timeframe.

    Use weekdays data divided in three time frames ["rest", "work", "leisure"] to generate location labels.
    "rest" + "leisure" locations are weighted together. The location with the longest duration is assigned "home" label.
    The longest "work" location is assigned "work" label.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        Staypoints with the column "location_id".

    Returns
    -------
    GeoDataFrame (as trackintel staypoints)
        The input staypoints with additional column "activity_label".

    Note
    ----
    The method is adapted from [1].
    When "home" and "work" label overlap, the method selects the "work" location by the 2nd highest score.
    The original algorithm count the distinct hours at a location as the home location is derived from
    geo-tagged tweets. We directly sum the time spent at a location.

    References
    ----------
    [1] Efstathiades, Hariton, Demetris Antoniades, George Pallis, and Marios Dikaiakos. 2015.
    ‘Identification of Key Locations Based on Online Social Network Activity’.
    In https://doi.org/10.1145/2808797.2808877.

    Examples
    --------
    >>> from ti.analysis.location_identification import osna_method
    >>> staypoints = osna_method(staypoints)
    """
    sp_in = staypoints  # no copy --> used to join back later.
    sp = sp_in.copy()
    sp["duration"] = sp["finished_at"] - sp["started_at"]
    sp["mean_time"] = sp["started_at"] + sp["duration"] / 2

    sp["label"] = sp["mean_time"].apply(_osna_label_timeframes)
    sp.loc[sp["label"] == "rest", "duration"] *= 0.739  # weight given in paper
    sp.loc[sp["label"] == "leisure", "duration"] *= 0.358  # weight given in paper

    groups_map = {
        "rest": "home",
        "leisure": "home",
        "work": "work",
    }  # weekends aren't included in analysis!
    # groupby user, location and label.
    groups = ["user_id", "location_id", sp["label"].map(groups_map)]

    sp_agg = sp.groupby(groups)["duration"].sum()
    if sp_agg.empty:
        warnings.warn("Got empty table in the osna method, check if the dates lie in weekends.")
        sp_in["activity_label"] = pd.NA
        return sp_in

    # create a pivot table -> labels "home" and "work" as columns. ("user_id", "location_id" still in index.)
    sp_pivot = sp_agg.unstack()
    # get index of maximum for columns "work" and "home"
    sp_idxmax = sp_pivot.groupby(["user_id"]).idxmax()
    # first assign labels
    for col in sp_idxmax.columns:
        sp_pivot.loc[sp_idxmax[col].dropna(), "activity_label"] = col

    # The "home" label could overlap with the "work" label
    # we set the rows where "home" is maximum to zero (pd.NaT) and recalculate index of work maximum.
    if all(col in sp_idxmax.columns for col in ["work", "home"]):
        redo_work = sp_idxmax[sp_idxmax["home"] == sp_idxmax["work"]]
        sp_pivot.loc[redo_work["work"], "activity_label"] = "home"
        sp_pivot.loc[redo_work["work"], "work"] = pd.NaT
        sp_idxmax_work = sp_pivot.groupby(["user_id"])["work"].idxmax()
        sp_pivot.loc[sp_idxmax_work.dropna(), "activity_label"] = "work"

    # now join it back together
    sel = sp_in.columns != "activity_label"  # no overlap with older "activity_label"
    return pd.merge(
        sp_in.loc[:, sel],
        sp_pivot["activity_label"],
        how="left",
        left_on=["user_id", "location_id"],
        right_index=True,
    )


def _osna_label_timeframes(dt, weekend=[5, 6], start_rest=2, start_work=8, start_leisure=19):
    """Help function to assign "weekend", "rest", "work", "leisure"."""
    if dt.weekday() in weekend:
        return "weekend"
    if start_rest <= dt.hour < start_work:
        return "rest"
    if start_work <= dt.hour < start_leisure:
        return "work"
    return "leisure"
