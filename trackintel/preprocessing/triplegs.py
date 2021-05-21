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


def generate_trips(stps_input, tpls_input, gap_threshold=15, print_progress=False):
    """
    see _generate_trips_old in test_triplegs.py for the docstring

    Parameters
    ----------
    stps_input
    tpls_input
    gap_threshold
    print_progress

    Returns
    -------

    """

    assert (
        "activity" in stps_input.columns
    ), "staypoints need the column 'activities' \
                                            to be able to generate trips"

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    stps = stps_input.copy()

    # if the triplegs already have a column "trip_id", we drop it
    if "trip_id" in tpls:
        tpls.drop(columns="trip_id", inplace=True)

    # if the staypoints already have any of the columns  "trip_id", "prev_trip_id", "next_trip_id", we drop them
    for col in ["trip_id", "prev_trip_id", "next_trip_id"]:
        if col in stps:
            stps.drop(columns=col, inplace=True)

    tpls["type"] = "tripleg"
    stps["type"] = "staypoint"

    # create table with relevant information from triplegs and staypoints.
    stps_tpls = stps[
        ["started_at", "finished_at", "user_id", "type", "activity"]
    ].append(tpls[["started_at", "finished_at", "user_id", "type"]])

    # create ID field from index
    stps_tpls["id"] = stps_tpls.index
    # transform nan to bool
    stps_tpls["activity"] = stps_tpls["activity"] == True

    stps_tpls.sort_values(by=["user_id", "started_at"], inplace=True)
    stps_tpls.reset_index(inplace=True, drop=True)

    stps_tpls["started_at_next"] = stps_tpls["started_at"].shift(-1)
    stps_tpls["activity_next"] = stps_tpls["activity"].shift(-1)

    # conditions for new trip:

    # new user flag
    condition_new_user = stps_tpls["user_id"] != stps_tpls["user_id"].shift(1)

    # - start a new trip if there is a new activity.
    condition_new_activity = stps_tpls["activity"]

    # if there are two activities in a row, only start a new trip at the last one
    activity_idx = stps_tpls["activity"].to_numpy().nonzero()[0]
    activities_to_del = activity_idx[:-1] - activity_idx[1:]
    idx = (activities_to_del == -1).nonzero()
    idx_orig = activity_idx[idx]  # reproject to original index space

    # todo: exclude consecutive activities (idx stored in variable idx_orig) from creating a new trip

    # todo: add missing gap conditions

    cond_all = condition_new_activity | condition_new_user
    stps_tpls["new_trip"] = cond_all

    # assign an incrementing id to all positionfixes that start a tripleg
    # create triplegs
    stps_tpls["trip_id"] = np.nan
    stps_tpls.loc[cond_all, "trip_id"] = np.arange(cond_all.sum())

    stps_tpls["trip_id"] = stps_tpls["trip_id"].fillna(method="ffill")

    # exclude activities

    stps_tpls_no_act = stps_tpls[stps_tpls["activity"] == False]
    stps_tpls_only_act = stps_tpls[stps_tpls["activity"] == True]
    stps_tpls_only_act["activity_id"] = stps_tpls_only_act["id"]

    # create trips by grouping
    trips_grouper = stps_tpls_no_act.groupby("trip_id")
    trips = trips_grouper.agg(
        {
            "user_id": ["mean"],
            "started_at": min,
            "finished_at": max,
            "type": list,
            "id": list,
        }
    )

    def _seperate_ids(row):

        stps_ids = []
        tpls_ids = []
        for ix, type_ in enumerate(row[("type", "list")]):
            if type_ == "staypoint":
                stps_ids.append(row[("id", "list")][ix])
            else:
                tpls_ids.append(row[("id", "list")][ix])
        return [stps_ids, tpls_ids]

    trips[["stps", "tpls"]] = trips.apply(_seperate_ids, axis=1, result_type="expand")

    trips.columns = trips.columns.droplevel(1)

    # save trip_id as column
    trips.reset_index(inplace=True)

    # merge with activitie
    trips = pd.concat((trips, stps_tpls_only_act), axis=0, ignore_index=True)
    trips = trips.sort_values(["user_id", "started_at"])

    # add origin/destination ids by shifting
    trips["origin_staypoint_id"] = trips["activity_id"].shift(1)
    trips["destination_staypoint_id"] = trips["activity_id"].shift(-1)

    # check for gaps and delete origin destination ids
    # add a gap_before flag and a gap after flag and then delete these rows
    trips["gap_origin"] = (
                                  trips["started_at"] - trips["finished_at"].shift(1)
                          ).dt.total_seconds() / 60 > gap_threshold
    trips["gap_destination"] = (
                                       trips["started_at"].shift(-1) - trips["finished_at"]
                               ).dt.total_seconds() / 60 > gap_threshold

    # todo: delete ids in case of gap
    # todo: create gap if user_id changes (in case that there is no temporal gap for some reason)

    # clean up

    # delete activities
    # transform column to binary
    trips["activity"] = trips["activity"] == True
    trips = trips[~trips["activity"]]

    trips.drop(
        [
            "type",
            "id",
            "activity",
            "started_at_next",
            "activity_next",
            "new_trip",
            "activity_id",
            "gap_origin",
            "gap_destination",
        ],
        inplace=True,
        axis=1,
    )

    trips.rename({"trip_id": "id"}, inplace=True, axis=1)

    # index management
    # trips["id"] = np.arange(len(trips))
    trips.set_index("id", inplace=True)

    # assign trip_id to tpls
    trip2tpl_map = trips[["tpls"]].to_dict()["tpls"]
    ls = []
    for key, values in trip2tpl_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[tpls.index.name, "trip_id"]).set_index(
        tpls.index.name
    )
    tpls = tpls.join(temp, how="left")

    # assign trip_id to stps, for non-activity stps
    trip2spt_map = trips[["stps"]].to_dict()["stps"]
    ls = []
    for key, values in trip2spt_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[stps.index.name, "trip_id"]).set_index(
        stps.index.name
    )
    stps = stps.join(temp, how="left")

    # assign prev_trip_id to stps
    temp = trips[["destination_staypoint_id"]].copy()
    temp.rename(columns={"destination_staypoint_id": stps.index.name}, inplace=True)
    temp.index.name = "prev_trip_id"
    temp = temp.reset_index().set_index(stps.index.name)
    stps = stps.join(temp, how="left")

    # assign next_trip_id to stps
    temp = trips[["origin_staypoint_id"]].copy()
    temp.rename(columns={"origin_staypoint_id": stps.index.name}, inplace=True)
    temp.index.name = "next_trip_id"
    temp = temp.reset_index().set_index(stps.index.name)
    stps = stps.join(temp, how="left")

    # final cleaning
    tpls.drop(columns=["type"], inplace=True)
    stps.drop(columns=["type"], inplace=True)
    trips.drop(columns=["tpls", "stps"], inplace=True)

    ## dtype consistency
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
