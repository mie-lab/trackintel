import numpy as np
import pandas as pd


def smoothen_triplegs(triplegs, tolerance=1.0, preserve_topology=True):
    """
    Reduce number of points while retaining structure of tripleg.

    A wrapper function using shapely.simplify():
    https://shapely.readthedocs.io/en/stable/manual.html#object.simplify

    Parameters
    ----------
    triplegs: GeoDataFrame (as trackintel triplegs)
        triplegs to be simplified

    tolerance: float, default 1.0
        a higher tolerance removes more points; the units of tolerance are the same as the
        projection of the input geometry

    preserve_topology: bool, default True
        whether to preserve topology. If set to False the Douglas-Peucker algorithm is used.

    Returns
    -------
    ret_tpls: GeoDataFrame (as trackintel triplegs)
        The simplified triplegs GeoDataFrame
    """
    ret_tpls = triplegs.copy()
    origin_geom = ret_tpls.geom
    simplified_geom = origin_geom.simplify(
        tolerance, preserve_topology=preserve_topology
    )
    ret_tpls.geom = simplified_geom

    return ret_tpls


def generate_trips(stps, tpls, gap_threshold=15):
    """Generate trips based on staypoints and triplegs.

    Parameters
    ----------
    stps : GeoDataFrame (as trackintel staypoints)

    tpls : GeoDataFrame (as trackintel triplegs)

    gap_threshold : float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
        `gap_threshold` minutes, then a new trip begins after the gap.

    Returns
    -------
    staypoints: GeoDataFrame (as trackintel staypoints)
        The original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.

    triplegs: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`trip_id`]``.

    trips: GeoDataFrame (as trackintel trips)
        The generated trips.

    Notes
    -----
    Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities.
    The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
    [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
    The following assumptions are implemented

        - If we do not record a person for more than `gap_threshold` minutes,
          we assume that the person performed an activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination
        - There are no trips without a (recored) tripleg

    Examples
    --------
    >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)
    """

    assert ("activity" in stps.columns), "staypoints need the column 'activities' to be able to generate trips"

    # Copy the input because we add a temporary columns
    tpls = tpls.copy()
    stps = stps.copy()
    gap_threshold = pd.to_timedelta(gap_threshold, unit="min")

    # If the triplegs already have a column "trip_id", we drop it
    if "trip_id" in tpls:
        tpls.drop(columns="trip_id", inplace=True)

    # if the staypoints already have any of the columns "trip_id", "prev_trip_id", "next_trip_id", we drop them
    for col in ["trip_id", "prev_trip_id", "next_trip_id"]:
        if col in stps:
            stps.drop(columns=col, inplace=True)

    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    stps_tpls = pd.concat(
        [
            stps[["started_at", "finished_at", "user_id", "type", "activity"]],
            tpls[["started_at", "finished_at", "user_id", "type"]],
        ]
    )
    # transform nan to bool
    stps_tpls["activity"].fillna(False, inplace=True)

    # create ID field from index
    stps_tpls["id"] = stps_tpls.index

    stps_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    stps_tpls.reset_index(inplace=True, drop=True)

    # delete intermediate activities
    # intermediate activities are activities that are surrounded by other activities
    # therefor their are neither at the end, start, nor in a trip
    first_activities, inter_activities, last_activities = _get_activity_masks(stps_tpls)
    stps_tpls["first_activity"] = first_activities
    stps_tpls["last_activity"] = last_activities  # trip may start here
    stps_tpls = stps_tpls[~inter_activities]  # no trip interaction
    stps_tpls["started_at_next"] = stps_tpls["started_at"].shift(-1)

    # conditions for new trip
    # new user flag
    condition_new_user = stps_tpls["user_id"] != stps_tpls["user_id"].shift(1)

    # start new trip if there is a new activity
    condition_new_activity = stps_tpls["last_activity"]

    # gap conditions
    gap = (stps_tpls["started_at_next"] - stps_tpls["finished_at"]) > gap_threshold
    condition_time_gap = gap.shift(1, fill_value=False)  # trip starts on next entry
    # no gaps between activities
    condition_time_gap = condition_time_gap & ~stps_tpls["first_activity"]

    cond_all = condition_new_user | condition_new_activity | condition_time_gap
    stps_tpls["new_trip"] = cond_all

    # assign an incrementing id to all positionfixes that start a tripleg
    stps_tpls.loc[cond_all, "trip_id"] = np.arange(cond_all.sum())
    stps_tpls["trip_id"].fillna(method="ffill", inplace=True)

    # exclude activities to aggregate trips together.
    # activity can be thought of as the same level as trips.
    stps_tpls_no_act = stps_tpls[~stps_tpls["activity"]]
    stps_tpls_only_act = stps_tpls[stps_tpls["activity"]]

    # add gaps as activities, to simplify id assignment later.
    gaps = pd.DataFrame(stps_tpls.loc[gap, "user_id"])
    gaps["started_at"] = stps_tpls.loc[gap, "finished_at"] + gap_threshold/2
    gaps[["type", "activity", "new_trip"]] = ["gap", True, True]

    # same for user changes
    user_change = pd.DataFrame(stps_tpls.loc[condition_new_user, "user_id"])
    user_change["started_at"] = stps_tpls.loc[condition_new_user, "started_at"] - gap_threshold/2
    user_change[["type", "activity", "new_trip"]] = ["user_change", True, True]

    trips_grouper = stps_tpls_no_act.groupby("trip_id")
    trips = trips_grouper.agg(
        {
            "user_id": "mean",
            "started_at": min,
            "finished_at": max,
            "type": np.array,
            "id": np.array,
        }
    )

    def _seperate_ids(row):
        """Split aggregated ids into staypoints and triplegs arrays."""
        t = (row["type"] == "tripleg")
        tpls_ids = row.loc["id"][t]
        stps_ids = row.loc["id"][np.logical_not(t)]
        return [stps_ids, tpls_ids]

    trips[["stps", "tpls"]] = trips.apply(_seperate_ids, axis=1, result_type="expand")

    # save trip_id as column
    trips.reset_index(inplace=True)
    trips["trip_id"] = trips.index

    # merge with activities
    trips.drop(columns=["type", "id"], inplace=True)  # make space so no overlap with activity "id"
    stps_tpls_only_act.drop(columns=["trip_id"])  # no overlap of activity trip_ids to real trip_ids in trips
    trips = pd.concat((trips, stps_tpls_only_act, gaps, user_change), axis=0, ignore_index=True)
    trips = trips.sort_values(["user_id", "started_at"])

    # add origin/destination ids by shifting
    trips["origin_staypoint_id"] = trips["id"].shift(1)
    trips["destination_staypoint_id"] = trips["id"].shift(-1)

    # add prev_trip_id and next_trip_id for activity staypoints
    trips["prev_trip_id"] = trips["trip_id"].shift(1)
    trips["next_trip_id"] = trips["trip_id"].shift(-1)
    activity_staypoints = trips[trips["type"] == "staypoint"].copy()
    activity_staypoints.index = activity_staypoints["id"]
    stps = stps.join(activity_staypoints[["prev_trip_id", "next_trip_id"]], how="left")

    # transform column to binary
    trips["activity"].fillna(False, inplace=True)
    # delete activities
    trips = trips[~trips["activity"]].copy()

    trips.drop(
        [
            "type",
            "id",
            "activity",
            "started_at_next",
            "first_activity",
            "last_activity",
            "new_trip",
            "prev_trip_id",
            "next_trip_id",
        ],
        inplace=True,
        axis=1,
    )

    trips.rename(columns={"trip_id": "id"}, inplace=True)

    # index management
    trips.set_index("id", inplace=True, drop=False)

    # assign trip_id to tpls
    temp = trips.explode("tpls")
    temp.index = temp["tpls"]
    temp = temp[temp['stps'].notna()]
    temp.rename(columns={"id": "trip_id"}, inplace=True)
    tpls = tpls.join(temp["trip_id"], how="left")

    # assign trip_id to stps, for non-activity stps
    temp = trips.explode("stps")
    temp.index = temp["stps"]
    temp = temp[temp['stps'].notna()]
    temp.rename(columns={"id": "trip_id"}, inplace=True)
    stps = stps.join(temp["trip_id"], how="left")

    # final cleaning
    tpls.drop(columns=["type"], inplace=True)
    stps.drop(columns=["type"], inplace=True)
    trips.drop(columns=["tpls", "stps", "id"], inplace=True)

    # dtype consistency
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype("int64")
    # trip id of stps and tpls can only be in Int64 (missing values)
    stps["trip_id"] = stps["trip_id"].astype("Int64")
    stps["prev_trip_id"] = stps["prev_trip_id"].astype("Int64")
    stps["next_trip_id"] = stps["next_trip_id"].astype("Int64")
    tpls["trip_id"] = tpls["trip_id"].astype("Int64")

    # user_id of trips should be the same as tpls
    trips["user_id"] = trips["user_id"].astype(tpls["user_id"].dtype)

    return stps, tpls, trips


def _get_activity_masks(df):
    """Split activities into three groups depending if other activities.

    Tell if activity is first (trip end), intermediate (can be deleted), or last (trip starts).
    First and last are intended to overlap.

    Parameters
    ----------
    df : DataFrame
        DataFrame with boolean column "activity".

    Returns
    -------
    is_first, is_inter, is_last
        Three boolean Series
    """
    prev_activity = df["activity"].shift(1, fill_value=False)
    next_activity = df["activity"].shift(-1, fill_value=False)
    is_first = df["activity"] & ~prev_activity
    is_last = df["activity"] & ~next_activity
    is_inter = df["activity"] & ~is_first & ~is_last
    return is_first, is_inter, is_last
