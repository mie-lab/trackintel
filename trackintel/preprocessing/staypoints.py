import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings

from trackintel import Staypoints, Locations
from trackintel.geogr import meters_to_decimal_degrees, check_gdf_planar
from trackintel.preprocessing.util import applyParallel, angle_centroid_multipoints


def generate_locations(
    staypoints,
    method="dbscan",
    epsilon=100,
    num_samples=1,
    distance_metric="haversine",
    agg_level="user",
    activities_only=False,
    print_progress=False,
    n_jobs=1,
):
    """
    Generate locations from the staypoints.

    Parameters
    ----------
    staypoints : Staypoints

    method : {'dbscan'}
        Method to create locations.

        - `dbscan` : Uses the DBSCAN algorithm to cluster staypoints.

    epsilon : float, default 100
        The epsilon for the 'dbscan' method. if 'distance_metric' is 'haversine'
        or 'euclidean', the unit is in meters.

    num_samples : int, default 1
        The minimal number of samples in a cluster.

    distance_metric: {'haversine', 'euclidean'}
        The distance metric used by the applied method. Any mentioned below are possible:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    agg_level: {'user','dataset'}
        The level of aggregation when generating locations:

        - `user`: locations are generated independently per-user.
        - `dataset`: shared locations are generated for all users.

    activities_only: bool, default False (requires "activity" column)
        Flag to set if locations should be generated only from staypoints on which the value for "activity" is True.
        Useful if activites represent more significant places.

    print_progress : bool, default False
        If print_progress is True, the progress bar is displayed

    n_jobs: int, default 1
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description

    Returns
    -------
    sp: Staypoints
        The original staypoints with a new column ``[`location_id`]``.

    locs: Locations
        The generated locations.

    Examples
    --------
    >>> sp.generate_locations(method='dbscan', epsilon=100, num_samples=1)
    """
    Staypoints.validate(staypoints)
    if agg_level not in ["user", "dataset"]:
        raise AttributeError(f"agg_level '{agg_level}' is unknown. Supported values are ['user', 'dataset'].")
    if method not in ["dbscan"]:
        raise AttributeError(f"method '{method}' is unknown. Supported value is ['dbscan'].")

    # initialize the return GeoDataFrames
    sp = staypoints.copy()
    non_activities = None
    if activities_only:
        if "activity" not in sp.columns:
            raise KeyError('staypoints must contain column "activity" if "activities_only" flag is set.')
        non_activities = sp[~sp["activity"]]
        sp = sp[sp["activity"]]
    sp = sp.sort_values(["user_id", "started_at"])
    geo_col = sp.geometry.name

    if method == "dbscan":
        eps = epsilon / 6371000 if distance_metric == "haversine" else epsilon
        # scikit haversine_distance wants radian. (We assume that this is good enough)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
        db = DBSCAN(eps=eps, min_samples=num_samples, algorithm="ball_tree", metric=distance_metric)

        if agg_level == "user":
            sp = applyParallel(
                sp.groupby("user_id", as_index=False),
                _gen_locs_dbscan,
                n_jobs=n_jobs,
                print_progress=print_progress,
                distance_metric=distance_metric,
                db=db,
            )

            # keeping track of noise labels
            sp_non_noise_labels = sp[sp["location_id"] != -1]
            sp_noise_labels = sp[sp["location_id"] == -1]

            # sort so that the last location id of a user = max(location id)
            sp_non_noise_labels = sp_non_noise_labels.sort_values(["user_id", "location_id"])

            # identify start positions of new user_ids
            start_of_user_id = sp_non_noise_labels["user_id"] != sp_non_noise_labels["user_id"].shift(1)

            # calculate the offset (= last location id of the previous user)
            # multiplication is to mask all positions where no new user starts and addition is to have a +1 when a
            # new user starts
            loc_id_offset = sp_non_noise_labels["location_id"].shift(1) * start_of_user_id + start_of_user_id

            # fill first nan with 0 and create the cumulative sum
            loc_id_offset = loc_id_offset.fillna(0).cumsum()

            sp_non_noise_labels["location_id"] = sp_non_noise_labels["location_id"] + loc_id_offset
            sp = gpd.GeoDataFrame(pd.concat([sp_non_noise_labels, sp_noise_labels]), geometry=geo_col)
            sp.sort_values(["user_id", "started_at"], inplace=True)

        else:
            _gen_locs_dbscan(sp, db=db, distance_metric=distance_metric)

        ### create locations as grouped staypoints
        temp_sp = sp[["user_id", "location_id", sp.geometry.name]]
        if agg_level == "user":
            # directly dissolve by 'user_id' and 'location_id'
            locs = temp_sp.dissolve(by=["user_id", "location_id"], as_index=False)
        else:
            ## generate user-location pairs with same geometries across users
            # get user-location pairs
            locs = temp_sp[["user_id", "location_id"]].drop_duplicates(ignore_index=True)
            # get location geometries
            geom_gdf = temp_sp.dissolve(by=["location_id"], as_index=False).drop(columns={"user_id"})
            # merge pairs with location geometries
            locs = geom_gdf.merge(locs, on="location_id", how="right")

        # filter staypoints not belonging to locations
        locs = locs.loc[locs["location_id"] != -1]

        if check_gdf_planar(locs):
            locs["center"] = locs.geometry.centroid
        else:
            # error of wrapping e.g. mean([-180, +180]) -> own function needed
            locs["center"] = angle_centroid_multipoints(locs.geometry)

        # extent is the convex hull of the geometry
        locs["extent"] = locs.geometry.convex_hull
        # convex_hull of Point is Point, and MultiPoint with two Points is a LineString
        # we change them into Polygon by creating a buffer of epsilon around them.
        pointLine_idx = (locs["extent"].geom_type == "LineString") | (locs["extent"].geom_type == "Point")

        # Perform meter to decimal conversion if the distance metric is haversine
        if distance_metric == "haversine":
            locs.loc[pointLine_idx, "extent"] = locs.loc[pointLine_idx].apply(
                lambda p: p["extent"].buffer(meters_to_decimal_degrees(epsilon, p["center"].y)), axis=1
            )
        else:
            locs.loc[pointLine_idx, "extent"] = locs.loc[pointLine_idx, "extent"].buffer(epsilon)

        locs = locs.set_geometry("center", crs=sp.crs)
        locs = locs[["user_id", "location_id", "center", "extent"]]

        # index management
        locs.rename(columns={"location_id": "id"}, inplace=True)
        locs.set_index("id", inplace=True)

        # staypoints not linked to a location receive np.nan in 'location_id'
        sp.loc[sp["location_id"] == -1, "location_id"] = np.nan

    # merge non_activities back if "activities_only" flag is set
    sp = pd.concat([sp, non_activities])

    ## dtype consistency
    # locs id (generated by this function) should be int64
    locs.index = locs.index.astype("int64")
    # location_id of staypoints can only be in Int64 (missing values)
    sp["location_id"] = sp["location_id"].astype("Int64")
    # user_id of locs should be the same as sp
    locs["user_id"] = locs["user_id"].astype(sp["user_id"].dtype)
    if len(locs) == 0:
        warnings.warn("No locations can be generated, returning empty locs.")
        return sp, locs
    # keep class of staypoints
    sp = Staypoints(sp) if isinstance(staypoints, Staypoints) else sp
    return sp, Locations(locs)


def _gen_locs_dbscan(sp, distance_metric, db):
    """Small helper function that takes staypoints and apply them to DBSCAN.

    Parameters
    ----------
    sp : Staypoints
    distance_metric : str
    db : sklearn.cluster.DBSCAN

    Returns
    -------
    sp : Staypoints
        Staypoints with new column "location_id"
    """
    p = np.array([sp.geometry.x, sp.geometry.y]).transpose()
    if distance_metric == "haversine":
        p = np.deg2rad(p)  # haversine distance metric assumes input is in rad
    labels = db.fit_predict(p)
    sp["location_id"] = labels
    return sp


def merge_staypoints(staypoints, triplegs, max_time_gap="10min", agg={}):
    """
    Aggregate staypoints horizontally via time threshold.

    Parameters
    ----------
    staypoints : Staypoints
        The staypoints must contain a column `location_id` (see `generate_locations` function)

    triplegs: Triplegs

    max_time_gap : str or pd.Timedelta, default "10min"
        Maximum duration between staypoints to still be merged.
        If str must be parsable by pd.to_timedelta.

    agg: dict, optional
        Dictionary to aggregate the rows after merging staypoints. This dictionary is used as input to the pandas
        aggregate function: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
        If empty, only the required columns of staypoints (which are ['user_id', 'started_at', 'finished_at']) are
        aggregated and returned. In order to return for example also the geometry column of the merged staypoints, set
        'agg={"geom":"first"}' to return the first geometry of the merged staypoints, or 'agg={"geom":"last"}' to use
        the last one.

    Returns
    -------
    sp: Staypoints
        The new staypoints with the default columns and columns in `agg`, where staypoints at same location and close in
        time are aggregated.

    Notes
    -----
    - Due to the modification of the staypoint index, the relation between the staypoints and the corresponding
      positionfixes **is broken** after execution of this function! In explanation, the staypoint_id column of pfs does
      not necessarily correspond to an id in the new sp table that is returned from this function. The same holds for
      trips (if generated yet) where the staypoints contained in a trip might be merged in this function.
    - If there is a tripleg between two staypoints, the staypoints are not merged. If you for some reason want to merge
      such staypoints, simply pass an empty DataFrame for the tpls argument.

    Examples
    --------
    >>> # direct function call
    >>> ti.preprocessing.merge_staypoints(staypoints=sp, triplegs=tpls)
    >>> # or using the trackintel datamodel
    >>> sp.merge_staypoints(triplegs, max_time_gap="1h", agg={"geom":"first"})
    """
    if isinstance(max_time_gap, str):
        max_time_gap = pd.to_timedelta(max_time_gap)
    # otherwise check if it's a Timedelta already, and raise error if not
    elif not isinstance(max_time_gap, pd.Timedelta):
        raise TypeError("Parameter max_time_gap must be either of type String or pd.Timedelta!")
    assert "location_id" in staypoints.columns, "Staypoints must contain column location_id"

    sp_merge = staypoints.copy()
    index_name = staypoints.index.name

    # concat sp and tpls to get information whether there is a tripleg between to staypoints
    tpls_merge = triplegs.copy()
    tpls_merge["type"] = "tripleg"
    sp_merge["type"] = "staypoint"
    # convert datatypes in order to preserve the datatypes (especially ints) despite of NaNs during concat
    sp_merge = sp_merge.convert_dtypes()

    # a joined dataframe sp_tpls is constructed to add the columns 'type' and 'next_type' to the 'sp_merge' table
    # concat and sort by time
    sp_tpls = pd.concat([sp_merge, tpls_merge]).sort_values(by=["user_id", "started_at"])
    # TODO: we want to make tpls as argumtent optional and adapt this logic here. See issue #463.
    sp_tpls.index.rename(index_name, inplace=True)
    # get information whether the there is a tripleg after a staypoint
    sp_tpls["next_type"] = sp_tpls["type"].shift(-1)
    # get only staypoints, but with next type information
    sp_merge = sp_tpls[sp_tpls["type"] == "staypoint"]

    # reset index and make temporary index
    sp_merge = sp_merge.reset_index()
    # copy index to use it in the end (id is modified)
    sp_merge["index_temp"] = sp_merge[index_name]

    # roll by 1 to get next row info
    sp_merge[["next_user_id", "next_started_at", "next_location_id"]] = sp_merge[
        ["user_id", "started_at", "location_id"]
    ].shift(-1)
    # Conditions to keep on merging
    cond = pd.Series(data=False, index=sp_merge.index)
    cond_old = pd.Series(data=True, index=sp_merge.index)
    cond_diff = cond != cond_old

    while np.sum(cond_diff) >= 1:
        # .values is important otherwise the "=" would imply a join via the new index
        sp_merge["next_id"] = sp_merge["index_temp"].shift(-1).values

        # identify rows to merge
        cond0 = sp_merge["next_user_id"] == sp_merge["user_id"]
        cond1 = sp_merge["next_started_at"] - sp_merge["finished_at"] <= max_time_gap  # time constraint
        cond2 = sp_merge["location_id"] == sp_merge["next_location_id"]
        cond3 = sp_merge["index_temp"] != sp_merge["next_id"]  # already merged
        cond4 = sp_merge["next_type"] != "tripleg"  # no tripleg inbetween two staypoints
        cond = cond0 & cond1 & cond2 & cond3 & cond4

        # assign index to next row
        sp_merge.loc[cond, "index_temp"] = sp_merge.loc[cond, "next_id"]
        # check whether anything was changed
        cond_diff = cond != cond_old
        cond_old = cond.copy()

    # Staypoint-required columnsare aggregated in the following manner:
    agg_dict = {
        index_name: "first",
        "user_id": "first",
        "started_at": "first",
        "finished_at": "last",
        "location_id": "first",
    }
    # User-defined further aggregation
    agg_dict.update(agg)

    # aggregate values
    sp = sp_merge.groupby(by="index_temp").agg(agg_dict).sort_values(by=["user_id", "started_at"])

    # clean
    sp = sp.set_index(index_name)
    return sp
