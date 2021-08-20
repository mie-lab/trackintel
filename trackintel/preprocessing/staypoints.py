from math import radians

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from trackintel.geogr.distances import meters_to_decimal_degrees
from trackintel.preprocessing.util import applyParallel


def generate_locations(
    staypoints,
    method="dbscan",
    epsilon=100,
    num_samples=1,
    distance_metric="haversine",
    agg_level="user",
    print_progress=False,
    n_jobs=1,
):
    """
    Generate locations from the staypoints.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        The staypoints have to follow the standard definition for staypoints DataFrames.

    method : {'dbscan'}
        Method to create locations.

        - 'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

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
        - 'user'      : locations are generated independently per-user.
        - 'dataset'   : shared locations are generated for all users.

    print_progress : bool, default False
        If print_progress is True, the progress bar is displayed

    n_jobs: int, default 1
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description

    Returns
    -------
    sp: GeoDataFrame (as trackintel staypoints)
        The original staypoints with a new column ``[`location_id`]``.

    locs: GeoDataFrame (as trackintel locations)
        The generated locations.

    Examples
    --------
    >>> sp.as_staypoints.generate_locations(method='dbscan', epsilon=100, num_samples=1)
    """
    if agg_level not in ["user", "dataset"]:
        raise AttributeError("The parameter agg_level must be one of ['user', 'dataset'].")
    if method not in ["dbscan"]:
        raise AttributeError("The parameter method must be one of ['dbscan'].")

    # initialize the return GeoDataFrames
    sp = staypoints.copy()
    sp = sp.sort_values(["user_id", "started_at"])
    geo_col = sp.geometry.name

    if method == "dbscan":

        if distance_metric == "haversine":
            # The input and output of sklearn's harvarsine metrix are both in radians,
            # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
            # here the 'epsilon' is directly applied to the metric's output.
            # convert to radius
            db = DBSCAN(eps=epsilon / 6371000, min_samples=num_samples, algorithm="ball_tree", metric=distance_metric)
        else:
            db = DBSCAN(eps=epsilon, min_samples=num_samples, algorithm="ball_tree", metric=distance_metric)

        if agg_level == "user":
            sp = applyParallel(
                sp.groupby("user_id", as_index=False),
                _generate_locations_per_user,
                n_jobs=n_jobs,
                print_progress=print_progress,
                geo_col=geo_col,
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
            if distance_metric == "haversine":
                # the input is converted to list of (lat, lon) tuples in radians unit
                p = np.array([[radians(g.y), radians(g.x)] for g in sp.geometry])
            else:
                p = np.array([[g.x, g.y] for g in sp.geometry])
            labels = db.fit_predict(p)

            sp["location_id"] = labels

        ### create locations as grouped staypoints
        temp_sp = sp[["user_id", "location_id", sp.geometry.name]]
        if agg_level == "user":
            # directly dissolve by 'user_id' and 'location_id'
            locs = temp_sp.dissolve(by=["user_id", "location_id"], as_index=False)
        else:
            ## generate user-location pairs with same geometries across users
            # get user-location pairs
            locs = temp_sp.dissolve(by=["user_id", "location_id"], as_index=False).drop(columns={temp_sp.geometry.name})
            # get location geometries
            geom_df = temp_sp.dissolve(by=["location_id"], as_index=False).drop(columns={"user_id"})
            # merge pairs with location geometries
            locs = locs.merge(geom_df, on="location_id", how="left")

        # filter staypoints not belonging to locations
        locs = locs.loc[locs["location_id"] != -1]

        locs["center"] = None  # initialize
        # locations with only one staypoints is of type "Point"
        point_idx = locs.geom_type == "Point"
        if not locs.loc[point_idx].empty:
            locs.loc[point_idx, "center"] = locs.loc[point_idx, locs.geometry.name]
        # locations with multiple staypoints is of type "MultiPoint"
        if not locs.loc[~point_idx].empty:
            locs.loc[~point_idx, "center"] = locs.loc[~point_idx, locs.geometry.name].apply(
                lambda p: Point(np.array(p)[:, 0].mean(), np.array(p)[:, 1].mean())
            )

        # extent is the convex hull of the geometry
        locs["extent"] = None  # initialize
        if not locs.empty:
            locs["extent"] = locs[locs.geometry.name].apply(lambda p: p.convex_hull)
            # convex_hull of one point would be a Point and two points a Linestring,
            # we change them into Polygon by creating a buffer of epsilon around them.
            pointLine_idx = (locs["extent"].geom_type == "LineString") | (locs["extent"].geom_type == "Point")

            if not locs.loc[pointLine_idx].empty:
                # Perform meter to decimal conversion if the distance metric is haversine
                if distance_metric == "haversine":
                    locs.loc[pointLine_idx, "extent"] = locs.loc[pointLine_idx].apply(
                        lambda p: p["extent"].buffer(meters_to_decimal_degrees(epsilon, p["center"].y)), axis=1
                    )
                else:
                    locs.loc[pointLine_idx, "extent"] = locs.loc[pointLine_idx].apply(
                        lambda p: p["extent"].buffer(epsilon), axis=1
                    )

        locs = locs.set_geometry("center")
        locs = locs[["user_id", "location_id", "center", "extent"]]

        # index management
        locs.rename(columns={"location_id": "id"}, inplace=True)
        locs.set_index("id", inplace=True)

    # staypoints not linked to a location receive np.nan in 'location_id'
    sp.loc[sp["location_id"] == -1, "location_id"] = np.nan

    ## dtype consistency
    # locs id (generated by this function) should be int64
    locs.index = locs.index.astype("int64")
    # location_id of staypoints can only be in Int64 (missing values)
    sp["location_id"] = sp["location_id"].astype("Int64")
    # user_id of locs should be the same as sp
    locs["user_id"] = locs["user_id"].astype(sp["user_id"].dtype)

    return sp, locs


def _generate_locations_per_user(user_staypoints, distance_metric, db, geo_col):
    """function called after groupby: should only contain records of one user;
    see generate_locations() function for parameter meaning."""

    if distance_metric == "haversine":
        # the input is converted to list of (lat, lon) tuples in radians unit
        p = np.array([[radians(q.y), radians(q.x)] for q in (user_staypoints[geo_col])])
    else:
        p = np.array([[q.x, q.y] for q in (user_staypoints[geo_col])])
    labels = db.fit_predict(p)

    # add staypoint - location matching to original staypoints
    user_staypoints["location_id"] = labels
    user_staypoints = gpd.GeoDataFrame(user_staypoints, geometry=geo_col)

    return user_staypoints
