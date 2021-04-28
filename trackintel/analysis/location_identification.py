import pandas as pd

_RECEIPTS = {
    "FREQ": freq_receipt,
}


def location_identifier(sps, pre_filter=True, receipt="FREQ"):
    """Finde home and work/location for each user with different receipts.

    Parameters
    ----------
    sps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    pre_filter : bool, default True
        Prefiltering the staypoints to exclude locations with not enough data.
        The filter function can also be acessed via `pre_filter`.

    receipt : {'FREQ'}, default "FREQ"
        Choose which receipt to use.
        - FREQ: Select the most visited location
        - PASS: just for decoration.

    Returns
    -------
    Geodataframe (as trackintel staypoints)
        With additional column assigning one of three location identies {'home', 'work', None}.

    Note
    ----
    The receipt are adapted from [1]. The original algorithms count the distinct hours at a
    location as the home location is derived from geo-tagged tweets. We directly sum the time
    spent at a location as our data model includes that.

    References
    ----------
    Chen, Qingqing, and Ate Poorthuis. 2021.
    ‘Identifying Home Locations in Human Mobility Data: An Open-Source R Package for Comparison and Reproducibility’.
    International Journal of Geographical Information Science 0 (0): 1–24.
    https://doi.org/10.1080/13658816.2021.1887489.

    Examples
    --------
    >>> ti.analysis.location_identification.location_idenifier(sps, pre_filter=True, receipt="FREQ")
    """
    # what do we do here?
    # we take the gdf and assert two things 1. is staypoint 2. has location_id column
    assert sps.as_staypoints
    if "location_id" not in sps.columns:
        raise KeyError((
            "To derive location activities the GeoDataFrame (as trackintel staypoints)must have a column "
            f"named 'location_id' but it has [{', '.join(sps.columns)}]"))
    # then hand it to to the filter function if necessary.
    if pre_filter:
        sps = pre_filter_locations()

    m = _RECEIPTS[receipt]()  # das müssen wir mal schöner machen.

    return m


def pre_filter_locations(sps, agg_level, thresh_min_sp=10, thresh_min_loc=10, thresh_sp_at_loc=10,
                         thresh_loc_time=1, thresh_loc_period=pd.Timedelta("5h")):
    """Filter locations and user out that have not enough data to do a proper analysis.

    Parameters
    ----------
    sps : GeoDataFrame (as trackintel staypoints)
        Staypoints with the column "location_id".

    agg_level: {"user", "dataset"}, default "user"
        The level of aggregation when filtering locations:
        - 'user' : locations are filtered per-user.
        - 'dataset' : locations are filtered over the whole dataset.

    thresh_min_sp : int, default 10
        Minimum staypoints a user must have to be included.

    thresh_min_loc : int, default 10
        Minimum locations a user must have to be included.

    thresh_sp_at_loc : int, default 10
        Minimum number of staypoints at a location must have to be included.

    thresh_loc_time : int, default 1
        Minimum timespan in hour that a user must spend at location for the location.

    thresh_loc_period : pd.Timedelta (or parseable from `pd.to_timedelta`), default 5h
        Minimum number of time a user have spent at location.

    Returns
    -------
    GeoDataFrame (as trackintel staypoints)

    Examples
    --------
    >> do something
    """
    assert sps.as_staypoints
    sps = sps.copy()

    # filtering users
    user = sps.groupby("user_id").nunique()
    user_sp = user['id'] >= thresh_min_sp
    user_loc = user['location_id'] >= thresh_min_loc  # how should we design our values inclusive or exclusive
    user_filter_agg = user_sp & user_loc
    user_filter_agg.rename("user_filter", inplace=True)  # rename for merging
    user_filter = pd.merge(sps['user_id'], user_filter_agg, left_on="user_id", right_index=True)["user_filter"]

    # filtering locations
    sps["timedelta"] = sps["finished_at"] - sps["started_at"]
    if agg_level == "User":
        groupby_loc = ["user_id", "location_id"]
    else:
        groupby_loc = ["location_id"]
    loc = sps.groupby(groupby_loc).agg({"started_at": [min, "count"], "finished_at": max, "time_spent": sum})
    loc.columns = loc.columns.droplevel(0)  # remove multi-index
    loc.rename(columns={"min": "started_at", "max": "finished_at", "sum": "duration"}, inplace=True)
    loc["period"] = loc["finished_at"] - loc["started_at"]
    loc_sp = loc["count"] >= thresh_sp_at_loc
    # how should we handle the timedeltas as input to our function
    # are both integers or do we accept other values?
    if isinstance(thresh_loc_time, int):
        thresh_loc_time = pd.to_timedelta(thresh_loc_time, unit="h")
    loc_time = loc["duration"] >= thresh_loc_time
    loc_period = loc["period"] >= thresh_loc_period
    loc_filter_agg = loc_sp & loc_time & loc_period
    loc_filter_agg.rename("loc_fiter", inplace=True)  # rename for merging
    loc_filter = pd.merge(sps[groupby_loc], loc_filter_agg.reset_index(), on=groupby_loc)

    total_filter = user_filter & loc_filter

    return total_filter


def freq_receipt(sps):
    """Docstring here :D
    """

    pass


def staypoints_with_location_assertion(funct):
    pass
