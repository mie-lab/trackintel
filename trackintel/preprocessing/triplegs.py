import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point

from trackintel import Staypoints, Triplegs, Trips
from trackintel.preprocessing.util import _explode_agg


def generate_trips(staypoints, triplegs, gap_threshold=15, add_geometry=True):
    """
    Generate trips based on staypoints and triplegs.

    Parameters
    ----------
    staypoints : Staypoints

    triplegs : Triplegs

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
    sp: Staypoints
        The original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.

    tpls: Triplegs
        The original triplegs with a new column ``[`trip_id`]``.

    trips: Trips
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
    >>> from trackintel.preprocessing import generate_trips
    >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)

    trips can also be directly generated using the tripleg class method
    >>> staypoints, triplegs, trips = triplegs.generate_trips(staypoints)
    """
    Triplegs.validate(triplegs)
    Staypoints.validate(staypoints)
    gap_threshold = pd.to_timedelta(gap_threshold, unit="min")
    sp_tpls = _concat_staypoints_triplegs(staypoints, triplegs, add_geometry)

    # conditions for new trip
    # start new trip if the user changes
    condition_new_user = sp_tpls["user_id"] != sp_tpls["user_id"].shift(1)

    # start new trip if there is a new activity (last activity in group)
    _, _, condition_new_activity = _get_activity_masks(sp_tpls)

    # gap conditions
    # start new trip after a gap, difference of started next with finish of current.
    gap = (sp_tpls["started_at"].shift(-1) - sp_tpls["finished_at"]) > gap_threshold
    condition_time_gap = gap.shift(1, fill_value=False)  # trip starts on next entry

    new_trip = condition_new_user | condition_new_activity | condition_time_gap

    # assign an incrementing id to all triplegs that start a trip
    # temporary as empty trips are not filtered out yet.
    sp_tpls.loc[new_trip, "temp_trip_id"] = np.arange(new_trip.sum())
    # fill NA with previous entry
    sp_tpls["temp_trip_id"].ffill(inplace=True)

    # exclude activities to aggregate trips together.
    # activity can be thought of as the same aggregation level as trips.
    sp_tpls_no_act = sp_tpls[~sp_tpls["is_activity"]]
    sp_tpls_only_act = sp_tpls[sp_tpls["is_activity"]]

    trips_grouper = sp_tpls_no_act.groupby("temp_trip_id")
    trips = trips_grouper.agg(
        {"user_id": "first", "started_at": "min", "finished_at": "max", "type": list, "sp_tpls_id": list}
    )

    def _seperate_ids(row):
        """Split aggregated sp_tpls_ids into staypoint ids and tripleg ids columns."""
        row_type = np.array(row["type"])
        row_id = np.array(row["sp_tpls_id"])
        t = row_type == "tripleg"
        tpls_ids = row_id[t]
        sp_ids = row_id[~t]
        # for dropping trips that don't have triplegs
        tpls_ids = tpls_ids if len(tpls_ids) > 0 else None
        return [sp_ids, tpls_ids]

    trips[["sp", "tpls"]] = trips.apply(_seperate_ids, axis=1, result_type="expand")

    # drop all trips that don't contain any triplegs
    trips.dropna(subset=["tpls"], inplace=True)

    # recount trips ignoring empty trips and save trip_id as for id assignment.
    trips.reset_index(inplace=True, drop=True)
    trips["trip_id"] = trips.index

    # add gaps as activities, to simplify id assignment.
    gaps = pd.DataFrame(sp_tpls.loc[gap, "user_id"])
    gaps["started_at"] = sp_tpls.loc[gap, "finished_at"] + gap_threshold / 2
    gaps[["type", "is_activity"]] = ["gap", True]  # nicer for debugging

    # same for user changes
    user_change = pd.DataFrame(sp_tpls.loc[condition_new_user, "user_id"])
    user_change["started_at"] = sp_tpls.loc[condition_new_user, "started_at"] - gap_threshold / 2
    user_change[["type", "is_activity"]] = ["user_change", True]  # nicer for debugging

    # merge trips with (filler) activities
    trips.drop(columns=["type", "sp_tpls_id"], inplace=True)  # make space so no overlap with activity "sp_tpls_id"
    # Inserting `gaps` and `user_change` into the dataframe creates buffers that catch shifted
    # "staypoint_id" and "trip_id" from corrupting staypoints/trips.
    trips_with_act = pd.concat((trips, sp_tpls_only_act, gaps, user_change), axis=0, ignore_index=True)
    trips_with_act.sort_values(["user_id", "started_at"], inplace=True)

    # ID assignment #
    # add origin/destination ids by shifting
    trips_with_act["origin_staypoint_id"] = trips_with_act["sp_tpls_id"].shift(1)
    trips_with_act["destination_staypoint_id"] = trips_with_act["sp_tpls_id"].shift(-1)

    # add geometry for start and end points
    if add_geometry:
        trips_with_act["origin_geom"] = trips_with_act["geom"].shift(1)
        trips_with_act["destination_geom"] = trips_with_act["geom"].shift(-1)

    # add prev_trip_id and next_trip_id for is_activity staypoints
    trips_with_act["prev_trip_id"] = trips_with_act["trip_id"].shift(1)
    trips_with_act["next_trip_id"] = trips_with_act["trip_id"].shift(-1)

    # transform column to binary
    trips_with_act["is_activity"].fillna(False, inplace=True)
    # delete activities
    trips = trips_with_act[~trips_with_act["is_activity"]].copy()

    trips.drop(
        [
            "type",
            "sp_tpls_id",
            "is_activity",
            "temp_trip_id",
            "prev_trip_id",
            "next_trip_id",
        ],
        inplace=True,
        axis=1,
    )

    # now handle the data that is aggregated in the trips
    # assign trip_id to tpls, override "trip_id" -> warning in _create_sp_tpls
    cols = triplegs.columns.difference(["trip_id"])
    tpls = _explode_agg("tpls", "trip_id", triplegs[cols], trips)  # creates copy of triplegs

    # first assign prev_trip_id, next_trip_id for activity staypoints
    activity_staypoints = trips_with_act[trips_with_act["type"] == "staypoint"].copy()
    # containing None changes dtype -> cast to original dtype.
    activity_staypoints.index = activity_staypoints["sp_tpls_id"].astype(staypoints.index.dtype)
    # override ["prev_trip_id", "next_trip_id", "trip_id"] -> warning in _create_sp_tpls
    cols = staypoints.columns.difference(["prev_trip_id", "next_trip_id", "trip_id"])
    sp = staypoints[cols].join(activity_staypoints[["prev_trip_id", "next_trip_id"]], how="left")
    # second assign trip_id to all staypoints
    sp = _explode_agg("sp", "trip_id", sp, trips)

    # fill missing points and convert to MultiPoint
    # for all trips with missing 'origin_staypoint_id' we now assign the startpoint of the first tripleg of the trip.
    # for all tripls with missing 'destination_staypoint_id' we now assign the endpoint of the last tripleg of the trip.
    if add_geometry:
        # fill geometry for origin staypoints that are NaN
        origin_nan_rows = trips[pd.isna(trips["origin_staypoint_id"])].copy()
        trips.loc[pd.isna(trips["origin_staypoint_id"]), "origin_geom"] = origin_nan_rows.tpls.map(
            # from tpls table, get the first point of the first tripleg for the trip
            lambda x: Point(tpls.loc[x[0], tpls.geometry.name].coords[0])
        )
        # fill geometry for destionations staypoints that are NaN
        destination_nan_rows = trips[pd.isna(trips["destination_staypoint_id"])].copy()
        trips.loc[pd.isna(trips["destination_staypoint_id"]), "destination_geom"] = destination_nan_rows.tpls.map(
            # from tpls table, get the last point of the last tripleg on the trip
            lambda x: Point(tpls.loc[x[-1], tpls.geometry.name].coords[-1])
        )
        # convert to GeoDataFrame with MultiPoint column and crs (not-None if possible)
        trips["geom"] = [MultiPoint([x, y]) for x, y in zip(trips.origin_geom, trips.destination_geom)]
        crs_trips = sp.crs if sp.crs else tpls.crs
        trips = gpd.GeoDataFrame(trips, geometry="geom", crs=crs_trips)
        # cleanup
        trips.drop(["origin_geom", "destination_geom"], inplace=True, axis=1)

    # final cleaning
    trips.drop(columns=["tpls", "sp", "trip_id"], inplace=True)

    # dtype consistency
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype("int64")
    trips.index.name = "id"  # TODO: some legacy issue for tests
    # trip id of sp and tpls can only be in Int64 (missing values)
    sp["trip_id"] = sp["trip_id"].astype("Int64")
    sp["prev_trip_id"] = sp["prev_trip_id"].astype("Int64")
    sp["next_trip_id"] = sp["next_trip_id"].astype("Int64")
    tpls["trip_id"] = tpls["trip_id"].astype("Int64")

    # user_id of trips should be the same as tpls
    trips["user_id"] = trips["user_id"].astype(tpls["user_id"].dtype)

    return sp, tpls, Trips(trips)


def _concat_staypoints_triplegs(staypoints, triplegs, add_geometry):
    """Concatenate staypoints and triplegs to sp_tpls with new columns ["type", "is_activity", "sp_tpls_id"].

    Parameters
    ----------
    staypoints : Staypoints
    triplegs : Triplegs
    add_geometry : bool

    Returns
    -------
    sp_tpls : GeoDataFrame
        "type": either "staypoint" or "tripleg"
        "is_activity": True for staypoints that are activity staypoints
        "sp_tpls_id": id of the corresponding staypoint or tripleg

    Raises
    ------
    AttributeError
        Raised if staypoints don't have column "is_activity"
    """
    if "is_activity" not in staypoints:
        raise AttributeError("staypoints need the column 'is_activity' to be able to generate trips")
    # Copy the input because we add temporary column "type"
    tpls = triplegs.copy()
    sp = staypoints.copy()

    # write warnings for columns that we replace
    if "trip_id" in tpls:
        warnings.warn("Override column 'trip_id' in copy of triplegs.")

    intersection = sp.columns.intersection(["trip_id", "prev_trip_id", "next_trip_id"])
    if len(intersection):
        warnings.warn(f"Override column(s) {intersection} in copy of staypoints.")

    tpls["is_activity"] = False  # in case "is_activity" is already a column of tpls
    tpls["type"] = "tripleg"
    sp["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    sp_cols = ["started_at", "finished_at", "user_id", "type", "is_activity"]
    tpls_cols = ["started_at", "finished_at", "user_id", "type"]
    sp_tpls = pd.concat([sp[sp_cols], tpls[tpls_cols]])
    sp_tpls["is_activity"].fillna(False, inplace=True)
    sp_tpls["sp_tpls_id"] = sp_tpls.index  # store id for later reassignment
    if add_geometry:
        # Check if crs is set. Warn if None
        if sp.crs is None:
            warnings.warn("Staypoint crs is not set. Assuming same as for triplegs.")
        if tpls.crs is None:
            warnings.warn("Tripleg crs is not set. Assuming same as for staypoints.")
        assert (
            sp.crs == tpls.crs or sp.crs is None or tpls.crs is None
        ), "CRS of staypoints and triplegs differ. Geometry cannot be joined safely."
        sp_tpls["geom"] = pd.concat([sp.geometry, tpls.geometry])

    sp_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    return sp_tpls


def _get_activity_masks(df):
    """Split activities into three groups depending if other activities.

    Tell if activity is first (trip end), intermediate (can be deleted), or last (trip starts).
    First and last are intended to overlap.

    Parameters
    ----------
    df : DataFrame
        DataFrame with boolean column "is_activity".

    Returns
    -------
    is_first, is_inter, is_last
        Three boolean Series
    """
    prev_activity = df["is_activity"].shift(1, fill_value=False)
    next_activity = df["is_activity"].shift(-1, fill_value=False)
    is_first = df["is_activity"] & ~prev_activity
    is_last = df["is_activity"] & ~next_activity
    is_inter = df["is_activity"] & ~is_first & ~is_last
    return is_first, is_inter, is_last
