import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint


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
    simplified_geom = origin_geom.simplify(tolerance, preserve_topology=preserve_topology)
    ret_tpls.geom = simplified_geom

    return ret_tpls


def generate_trips(spts, tpls, gap_threshold=15, add_geometry=True):
    """Generate trips based on staypoints and triplegs.

    Parameters
    ----------
    spts : GeoDataFrame (as trackintel staypoints)

    tpls : GeoDataFrame (as trackintel triplegs)

    gap_threshold : float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
        `gap_threshold` minutes, then a new trip begins after the gap.

    add_geometry : bool default True
        If True, the start and end coordinates of each trip are added to the output table in a geometry column "geom"
        of type MultiPoint. Set `add_geometry=False` for better runtime performance (if coordinates are not required).

    print_progress : bool, default False
        If print_progress is True, the progress bar is displayed

    Returns
    -------
    staypoints: GeoDataFrame (as trackintel staypoints)
        The original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.

    triplegs: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`trip_id`]``.

    trips: (Geo)DataFrame (as trackintel trips)
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
        - There are no trips without a (recorded) tripleg
        - Trips optionally have their start and end point as geometry of type MultiPoint, if `add_geometry==True`
        - If the origin (or destination) staypoint is unknown, and `add_geometry==True`, the origin (and destination)
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg),
          respectively. Trips with missing values can still be identified via col `origin_staypoint_id`.


    Examples
    --------
    >>> from trackintel.preprocessing.triplegs import generate_trips
    >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)

    trips can also be directly generated using the tripleg accessor
    >>> staypoints, triplegs, trips = triplegs.as_triplegs.generate_trips(staypoints)

    """

    assert "activity" in spts.columns, "staypoints need the column 'activities' to be able to generate trips"

    # Copy the input because we add a temporary columns
    tpls = tpls.copy()
    spts = spts.copy()
    gap_threshold = pd.to_timedelta(gap_threshold, unit="min")

    # If the triplegs already have a column "trip_id", we drop it
    if "trip_id" in tpls:
        tpls.drop(columns="trip_id", inplace=True)
        warnings.warn("Deleted existing column 'trip_id' from tpls.")

    # if the staypoints already have any of the columns "trip_id", "prev_trip_id", "next_trip_id", we drop them
    for col in ["trip_id", "prev_trip_id", "next_trip_id"]:
        if col in spts:
            spts.drop(columns=col, inplace=True)
            warnings.warn(f"Deleted column '{col}' from spts.")

    tpls["type"] = "tripleg"
    spts["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    spts_tpls = pd.concat(
        [
            spts[["started_at", "finished_at", "user_id", "type", "activity"]],
            tpls[["started_at", "finished_at", "user_id", "type"]],
        ]
    )
    if add_geometry:
        spts_tpls["geom"] = pd.concat([spts.geometry, tpls.geometry])

    # transform nan to bool
    spts_tpls["activity"].fillna(False, inplace=True)

    # create ID field from index
    spts_tpls["spts_tpls_id"] = spts_tpls.index

    spts_tpls.sort_values(by=["user_id", "started_at"], inplace=True)

    # conditions for new trip
    # start new trip if the user changes
    condition_new_user = spts_tpls["user_id"] != spts_tpls["user_id"].shift(1)

    # start new trip if there is a new activity (last activity in group)
    _, _, condition_new_activity = _get_activity_masks(spts_tpls)

    # gap conditions
    # start new trip after a gap, difference of started next with finish of current.
    gap = (spts_tpls["started_at"].shift(-1) - spts_tpls["finished_at"]) > gap_threshold
    condition_time_gap = gap.shift(1, fill_value=False)  # trip starts on next entry

    new_trip = condition_new_user | condition_new_activity | condition_time_gap

    # assign an incrementing id to all triplegs that start a trip
    # temporary as empty trips are not filtered out yet.
    spts_tpls.loc[new_trip, "temp_trip_id"] = np.arange(new_trip.sum())
    spts_tpls["temp_trip_id"].fillna(method="ffill", inplace=True)

    # exclude activities to aggregate trips together.
    # activity can be thought of as the same aggregation level as trips.
    spts_tpls_no_act = spts_tpls[~spts_tpls["activity"]]
    spts_tpls_only_act = spts_tpls[spts_tpls["activity"]]

    trips_grouper = spts_tpls_no_act.groupby("temp_trip_id")
    trips = trips_grouper.agg(
        {"user_id": "first", "started_at": min, "finished_at": max, "type": list, "spts_tpls_id": list}
    )

    def _seperate_ids(row):
        """Split aggregated spts_tpls_ids into staypoint ids and tripleg ids columns."""
        row_type = np.array(row["type"])
        row_id = np.array(row["spts_tpls_id"])
        t = row_type == "tripleg"
        tpls_ids = row_id[t]
        spts_ids = row_id[~t]
        # for dropping trips that don't have triplegs
        tpls_ids = tpls_ids if len(tpls_ids) > 0 else None
        return [spts_ids, tpls_ids]

    trips[["spts", "tpls"]] = trips.apply(_seperate_ids, axis=1, result_type="expand")

    # drop all trips that don't contain any triplegs
    trips.dropna(subset=["tpls"], inplace=True)

    # recount trips ignoring empty trips and save trip_id as for id assignment.
    trips.reset_index(inplace=True, drop=True)
    trips["trip_id"] = trips.index

    # add gaps as activities, to simplify id assignment.
    gaps = pd.DataFrame(spts_tpls.loc[gap, "user_id"])
    gaps["started_at"] = spts_tpls.loc[gap, "finished_at"] + gap_threshold / 2
    gaps[["type", "activity"]] = ["gap", True]  # nicer for debugging

    # same for user changes
    user_change = pd.DataFrame(spts_tpls.loc[condition_new_user, "user_id"])
    user_change["started_at"] = spts_tpls.loc[condition_new_user, "started_at"] - gap_threshold / 2
    user_change[["type", "activity"]] = ["user_change", True]  # nicer for debugging

    # merge trips with (filler) activities
    trips.drop(columns=["type", "spts_tpls_id"], inplace=True)  # make space so no overlap with activity "spts_tpls_id"
    # Inserting `gaps` and `user_change` into the dataframe creates buffers that catch shifted
    # "staypoint_id" and "trip_id" from corrupting staypoints/trips.
    trips_with_act = pd.concat((trips, spts_tpls_only_act, gaps, user_change), axis=0, ignore_index=True)
    trips_with_act.sort_values(["user_id", "started_at"], inplace=True)

    # ID assignment #
    # add origin/destination ids by shifting
    trips_with_act["origin_staypoint_id"] = trips_with_act["spts_tpls_id"].shift(1)
    trips_with_act["destination_staypoint_id"] = trips_with_act["spts_tpls_id"].shift(-1)

    # add geometry for start and end points
    if add_geometry:
        trips_with_act["origin_geom"] = trips_with_act["geom"].shift(1)
        trips_with_act["destination_geom"] = trips_with_act["geom"].shift(-1)

    # add prev_trip_id and next_trip_id for activity staypoints
    trips_with_act["prev_trip_id"] = trips_with_act["trip_id"].shift(1)
    trips_with_act["next_trip_id"] = trips_with_act["trip_id"].shift(-1)
    activity_staypoints = trips_with_act[trips_with_act["type"] == "staypoint"].copy()
    activity_staypoints.index = activity_staypoints["spts_tpls_id"]
    spts = spts.join(activity_staypoints[["prev_trip_id", "next_trip_id"]], how="left")

    # transform column to binary
    trips_with_act["activity"].fillna(False, inplace=True)
    # delete activities
    trips = trips_with_act[~trips_with_act["activity"]].copy()

    trips.drop(
        [
            "type",
            "spts_tpls_id",
            "activity",
            "temp_trip_id",
            "prev_trip_id",
            "next_trip_id",
        ],
        inplace=True,
        axis=1,
    )

    # fill missing points and convert to MultiPoint
    # for all trips with missing 'origin_staypoint_id' we now assign the startpoint of the first tripleg of the trip.
    # for all tripls with missing 'destination_staypoint_id' we now assign the endpoint of the last tripleg of the trip.
    if add_geometry:
        # fill geometry for origin staypoints that are NaN
        origin_nan_rows = trips[pd.isna(trips["origin_staypoint_id"])].copy()
        trips.loc[pd.isna(trips["origin_staypoint_id"]), "origin_geom"] = origin_nan_rows.tpls.map(
            # from tpls table, get the first point of the first tripleg for the trip
            lambda x: tpls.loc[x[0], tpls.geometry.name].boundary[0]
        )
        # fill geometry for destionations staypoints that are NaN
        destination_nan_rows = trips[pd.isna(trips["destination_staypoint_id"])].copy()
        trips.loc[pd.isna(trips["destination_staypoint_id"]), "destination_geom"] = destination_nan_rows.tpls.map(
            # from tpls table, get the last point of the last tripleg on the trip
            lambda x: tpls.loc[x[-1], tpls.geometry.name].boundary[1]
        )
        # convert to GeoDataFrame with MultiPoint column
        trips["geom"] = [MultiPoint([x, y]) for x, y in zip(trips.origin_geom, trips.destination_geom)]
        trips = gpd.GeoDataFrame(trips, geometry="geom")
        # cleanup
        trips.drop(["origin_geom", "destination_geom"], inplace=True, axis=1)

    # now handle the data that is aggregated in the trips
    # assign trip_id to tpls
    temp = trips.explode("tpls")
    temp.index = temp["tpls"]
    temp = temp[temp["tpls"].notna()]
    tpls = tpls.join(temp["trip_id"], how="left")

    # assign trip_id to spts, for non-activity spts
    temp = trips.explode("spts")
    temp.index = temp["spts"]
    temp = temp[temp["spts"].notna()]
    spts = spts.join(temp["trip_id"], how="left")

    # final cleaning
    tpls.drop(columns=["type"], inplace=True)
    spts.drop(columns=["type"], inplace=True)
    trips.drop(columns=["tpls", "spts", "trip_id"], inplace=True)

    # dtype consistency
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype("int64")
    trips.index.name = "id"  # TODO: some legacy issue for tests
    # trip id of spts and tpls can only be in Int64 (missing values)
    spts["trip_id"] = spts["trip_id"].astype("Int64")
    spts["prev_trip_id"] = spts["prev_trip_id"].astype("Int64")
    spts["next_trip_id"] = spts["next_trip_id"].astype("Int64")
    tpls["trip_id"] = tpls["trip_id"].astype("Int64")

    # user_id of trips should be the same as tpls
    trips["user_id"] = trips["user_id"].astype(tpls["user_id"].dtype)

    return spts, tpls, trips


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
