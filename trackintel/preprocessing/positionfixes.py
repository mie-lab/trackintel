import datetime
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from trackintel.geogr.distances import haversine_dist
from trackintel.preprocessing.util import applyParallel


def generate_staypoints(
    positionfixes,
    method="sliding",
    distance_metric="haversine",
    dist_threshold=100,
    time_threshold=5.0,
    gap_threshold=15.0,
    include_last=False,
    print_progress=False,
    exclude_duplicate_pfs=True,
    n_jobs=1,
):
    """
    Generate staypoints from positionfixes.

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    method : {'sliding'}
        Method to create staypoints. 'sliding' applies a sliding window over the data.

    distance_metric : {'haversine'}
        The distance metric used by the applied method.

    dist_threshold : float, default 100
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint. Units depend on the dist_func parameter. If 'distance_metric' is 'haversine' the
        unit is in meters

    time_threshold : float, default 5.0 (minutes)
        The time threshold for the 'sliding' method in minutes.

    gap_threshold : float, default 15.0 (minutes)
        The time threshold of determine whether a gap exists between consecutive pfs. Consecutive pfs with
        temporal gaps larger than 'gap_threshold' will be excluded from staypoints generation.
        Only valid in 'sliding' method.

    include_last: boolean, default False
        The algorithm in Li et al. (2008) only detects staypoint if the user steps out
        of that staypoint. This will omit the last staypoint (if any). Set 'include_last'
        to True to include this last staypoint.

    print_progress: boolean, default False
        Show per-user progress if set to True.

    exclude_duplicate_pfs: boolean, default True
        Filters duplicate positionfixes before generating staypoints. Duplicates can lead to problems in later
        processing steps (e.g., when generating triplegs). It is not recommended to set this to False.

    n_jobs: int, default 1
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`staypoint_id`]``.

    sp: GeoDataFrame (as trackintel staypoints)
        The generated staypoints.

    Notes
    -----
    The 'sliding' method is adapted from Li et al. (2008). In the original algorithm, the 'finished_at'
    time for the current staypoint lasts until the 'tracked_at' time of the first positionfix outside
    this staypoint. Users are assumed to be stationary during this missing period and potential tracking
    gaps may be included in staypoints. To avoid including too large missing signal gaps, set 'gap_threshold'
    to a small value, e.g., 15 min.

    Examples
    --------
    >>> pfs.as_positionfixes.generate_staypoints('sliding', dist_threshold=100)

    References
    ----------
    Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems
    and Technology (TIST), 6(3), 29.

    Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., & Ma, W. Y. (2008, November). Mining user
    similarity based on location history. In Proceedings of the 16th ACM SIGSPATIAL international
    conference on Advances in geographic information systems (p. 34). ACM.
    """
    # copy the original pfs for adding 'staypoint_id' column
    pfs = positionfixes.copy()

    if exclude_duplicate_pfs:
        len_org = pfs.shape[0]
        pfs = pfs.drop_duplicates()
        nb_dropped = len_org - pfs.shape[0]
        if nb_dropped > 0:
            warn_str = (
                f"{nb_dropped} duplicates were dropped from your positionfixes. Dropping duplicates is"
                + " recommended but can be prevented using the 'exclude_duplicate_pfs' flag."
            )
            warnings.warn(warn_str)

    # if the positionfixes already have a column "staypoint_id", we drop it
    if "staypoint_id" in pfs:
        pfs.drop(columns="staypoint_id", inplace=True)

    elevation_flag = "elevation" in pfs.columns  # if there is elevation data

    geo_col = pfs.geometry.name
    if elevation_flag:
        sp_column = ["user_id", "started_at", "finished_at", "elevation", geo_col]
    else:
        sp_column = ["user_id", "started_at", "finished_at", geo_col]

    # TODO: tests using a different distance function, e.g., L2 distance
    if method == "sliding":
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        sp = applyParallel(
            pfs.groupby("user_id", as_index=False),
            _generate_staypoints_sliding_user,
            n_jobs=n_jobs,
            print_progress=print_progress,
            geo_col=geo_col,
            elevation_flag=elevation_flag,
            dist_threshold=dist_threshold,
            time_threshold=time_threshold,
            gap_threshold=gap_threshold,
            distance_metric=distance_metric,
            include_last=include_last,
        ).reset_index(drop=True)

        # index management
        sp["id"] = np.arange(len(sp))
        sp.set_index("id", inplace=True)

        # Assign staypoint_id to ret_pfs if sp is detected
        if not sp.empty:
            sp2pfs_map = sp[["pfs_id"]].to_dict()["pfs_id"]

            ls = []
            for key, values in sp2pfs_map.items():
                for value in values:
                    ls.append([value, key])
            temp = pd.DataFrame(ls, columns=["id", "staypoint_id"]).set_index("id")
            # pfs with no sp receives nan in 'staypoint_id'
            pfs = pfs.join(temp, how="left")
            sp.drop(columns={"pfs_id"}, inplace=True)

        # if no staypoint at all is identified
        else:
            pfs["staypoint_id"] = np.nan

    pfs = gpd.GeoDataFrame(pfs, geometry=geo_col, crs=pfs.crs)
    sp = gpd.GeoDataFrame(sp, columns=sp_column, geometry=geo_col, crs=pfs.crs)
    # rearange column order
    sp = sp[sp_column]

    if len(sp) > 0:
        sp.as_staypoints
    else:
        warnings.warn("No staypoints can be generated, returning empty sp.")

    ## dtype consistency
    # sp id (generated by this function) should be int64
    sp.index = sp.index.astype("int64")
    # ret_pfs['staypoint_id'] should be Int64 (missing values)
    pfs["staypoint_id"] = pfs["staypoint_id"].astype("Int64")

    # user_id of sp should be the same as ret_pfs
    sp["user_id"] = sp["user_id"].astype(pfs["user_id"].dtype)

    return pfs, sp


def generate_triplegs(
    positionfixes,
    staypoints=None,
    method="between_staypoints",
    gap_threshold=15,
):
    """Generate triplegs from positionfixes.

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.
        If 'staypoint_id' column is not found, 'staypoints' needs to be given.

    staypoints : GeoDataFrame (as trackintel staypoints), optional
        The staypoints (corresponding to the positionfixes). If this is not passed, the
        positionfixes need 'staypoint_id' associated with them.

    method: {'between_staypoints'}
        Method to create triplegs. 'between_staypoints' method defines a tripleg as all positionfixes
        between two staypoints (no overlap). This method requires either a column 'staypoint_id' on
        the positionfixes or passing staypoints as an input.

    gap_threshold: float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
        `gap_threshold` minutes, a new tripleg will be generated.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`tripleg_id`]``.

    tpls: GeoDataFrame (as trackintel triplegs)
        The generated triplegs.

    Notes
    -----
    Methods 'between_staypoints' requires either a column 'staypoint_id' on the
    positionfixes or passing some staypoints that correspond to the positionfixes!
    This means you usually should call ``generate_staypoints()`` first.

    The first positionfix after a staypoint is regarded as the first positionfix of the
    generated tripleg. The generated tripleg will not have overlapping positionfix with
    the existing staypoints. This means a small temporal gap in user's trace will occur
    between the first positionfix of staypoint and the last positionfix of tripleg:
    pfs_stp_first['tracked_at'] - pfs_tpl_last['tracked_at'].

    Examples
    --------
    >>> pfs.as_positionfixes.generate_triplegs('between_staypoints', gap_threshold=15)
    """
    # copy the original pfs for adding 'tripleg_id' column
    pfs = positionfixes.copy()

    # if the positionfixes already have a column "tripleg_id", we drop it
    if "tripleg_id" in pfs:
        pfs.drop(columns="tripleg_id", inplace=True)

    # we need to ensure pfs is properly ordered
    pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)

    if method == "between_staypoints":

        # get case:
        # Case 1: pfs have a column 'staypoint_id'
        # Case 2: pfs do not have a column 'staypoint_id' but staypoint are provided

        if "staypoint_id" not in pfs.columns:
            case = 2
        else:
            case = 1

        # Preprocessing for case 2:
        # - step 1: Assign staypoint ids to positionfixes by matching timestamps (per user)
        # - step 2: Find first positionfix after a staypoint
        # (relevant if the pfs of sp are not provided, and we can only infer the pfs after sp through time)
        if case == 2:
            # initialize the index list of pfs where a tpl will begin
            insert_index_ls = []
            pfs["staypoint_id"] = pd.NA
            for user_id_this in pfs["user_id"].unique():
                sp_user = staypoints[staypoints["user_id"] == user_id_this]
                pfs_user = pfs[pfs["user_id"] == user_id_this]

                # step 1
                # All positionfixes with timestamp between staypoints are assigned the value 0
                # Intersect all positionfixes of a user with all staypoints of the same user
                intervals = pd.IntervalIndex.from_arrays(sp_user["started_at"], sp_user["finished_at"], closed="left")
                is_in_interval = pfs_user["tracked_at"].apply(lambda x: intervals.contains(x).any()).astype("bool")
                pfs.loc[is_in_interval[is_in_interval].index, "staypoint_id"] = 0

                # step 2
                # Identify first positionfix after a staypoint
                # find index of closest positionfix with equal or greater timestamp.
                tracked_at_sorted = pfs_user["tracked_at"].sort_values()
                insert_position_user = tracked_at_sorted.searchsorted(sp_user["finished_at"])
                insert_index_user = tracked_at_sorted.iloc[insert_position_user].index

                # store the insert insert_position_user in an array
                insert_index_ls.extend(list(insert_index_user))
            #
            cond_staypoints_case2 = pd.Series(False, index=pfs.index)
            cond_staypoints_case2.loc[insert_index_ls] = True

        # initialize tripleg_id with pd.NA and fill all pfs that belong to staypoints with -1
        # pd.NA will be replaced later with tripleg ids
        pfs["tripleg_id"] = pd.NA
        pfs.loc[~pd.isna(pfs["staypoint_id"]), "tripleg_id"] = -1

        # get all conditions that trigger a new tripleg.
        # condition 1: a positionfix belongs to a new tripleg if the user changes. For this we need to sort pfs.
        # The first positionfix of the new user is the start of a new tripleg (if it is no staypoint)
        cond_new_user = ((pfs["user_id"] - pfs["user_id"].shift(1)) != 0) & pd.isna(pfs["staypoint_id"])

        # condition 2: Temporal gaps
        # if there is a gap that is longer than gap_threshold minutes, we start a new tripleg
        cond_gap = pfs["tracked_at"] - pfs["tracked_at"].shift(1) > datetime.timedelta(minutes=gap_threshold)

        # condition 3: staypoint
        # By our definition the pf after a stp is the first pf of a tpl.
        # this works only for numeric staypoint ids, TODO: can we change?
        _stp_id = (pfs["staypoint_id"] + 1).fillna(0)
        cond_stp = (_stp_id - _stp_id.shift(1)) != 0

        # special check for case 2: pfs that belong to stp might not present in the data.
        # We need to select these pfs using time.
        if case == 2:
            cond_stp = cond_stp | cond_staypoints_case2

        # combine conditions
        cond_all = cond_new_user | cond_gap | cond_stp
        # make sure not to create triplegs within staypoints:
        cond_all = cond_all & pd.isna(pfs["staypoint_id"])

        # get the start position of tpls
        tpls_starts = np.where(cond_all)[0]
        tpls_diff = np.diff(tpls_starts)

        # get the start position of staypoint
        # pd.NA causes error in boolean comparision, replace to -1
        sp_id = pfs["staypoint_id"].copy().fillna(-1)
        unique_sp, sp_starts = np.unique(sp_id, return_index=True)
        # get the index of where the tpls_starts belong in sp_starts
        sp_starts = sp_starts[unique_sp != -1]
        tpls_place_in_sp = np.searchsorted(sp_starts, tpls_starts)

        # get the length between each stp and tpl
        try:
            # pfs ends with stp
            sp_tpls_diff = sp_starts[tpls_place_in_sp] - tpls_starts

            # tpls_lengths is the minimum of tpls_diff and sp_tpls_diff
            # sp_tpls_diff one larger than tpls_diff
            tpls_lengths = np.minimum(tpls_diff, sp_tpls_diff[:-1])

            # the last tpl has length (last stp begin - last tpl begin)
            tpls_lengths = np.append(tpls_lengths, sp_tpls_diff[-1])
        except IndexError:
            # pfs ends with tpl
            # ignore the tpls after the last sp sp_tpls_diff
            ignore_index = tpls_place_in_sp == len(sp_starts)
            sp_tpls_diff = sp_starts[tpls_place_in_sp[~ignore_index]] - tpls_starts[~ignore_index]

            # tpls_lengths is the minimum of tpls_diff and sp_tpls_diff
            tpls_lengths = np.minimum(tpls_diff[: len(sp_tpls_diff)], sp_tpls_diff)
            tpls_lengths = np.append(tpls_lengths, tpls_diff[len(sp_tpls_diff) :])

            # add the length of the last tpl
            tpls_lengths = np.append(tpls_lengths, len(pfs) - tpls_starts[-1])

        # a valid linestring needs 2 points
        cond_to_remove = np.take(tpls_starts, np.where(tpls_lengths < 2)[0])
        cond_all.iloc[cond_to_remove] = False
        # Note: cond_to_remove is the array index of pfs.index and not pfs.index itself
        pfs.loc[pfs.index[cond_to_remove], "tripleg_id"] = -1

        # assign an incrementing id to all positionfixes that start a tripleg
        # create triplegs
        pfs.loc[cond_all, "tripleg_id"] = np.arange(cond_all.sum())

        # fill the pd.NAs with the previously observed tripleg_id
        # pfs not belonging to tripleg are also propagated (with -1)
        pfs["tripleg_id"] = pfs["tripleg_id"].fillna(method="ffill")
        # assign back pd.NA to -1
        pfs.loc[pfs["tripleg_id"] == -1, "tripleg_id"] = pd.NA

        posfix_grouper = pfs.groupby("tripleg_id")

        tpls = posfix_grouper.agg(
            {"user_id": ["mean"], "tracked_at": [min, max], pfs.geometry.name: list}
        )  # could add a "number of pfs": can be any column "count"

        # prepare dataframe: Rename columns; read/set geometry/crs;
        # Order of column has to correspond to the order of the groupby statement
        tpls.columns = ["user_id", "started_at", "finished_at", "geom"]
        tpls["geom"] = tpls["geom"].apply(LineString)
        tpls = tpls.set_geometry("geom")
        tpls.crs = pfs.crs

        # assert validity of triplegs
        tpls, pfs = _drop_invalid_triplegs(tpls, pfs)

        if len(tpls) > 0:
            tpls.as_triplegs
        else:
            warnings.warn("No triplegs can be generated, returning empty tpls.")

        if case == 2:
            pfs.drop(columns="staypoint_id", inplace=True)

        # dtype consistency
        pfs["tripleg_id"] = pfs["tripleg_id"].astype("Int64")
        tpls.index = tpls.index.astype("int64")
        tpls.index.name = "id"

        # user_id of tpls should be the same as pfs
        tpls["user_id"] = tpls["user_id"].astype(pfs["user_id"].dtype)

        return pfs, tpls

    else:
        raise AttributeError(f"Method unknown. We only support 'between_staypoints'. You passed {method}")


def _generate_staypoints_sliding_user(
    df, geo_col, elevation_flag, dist_threshold, time_threshold, gap_threshold, distance_metric, include_last=False
):
    """User level staypoint geenration using sliding method, see generate_staypoints() function for parameter meaning."""
    if distance_metric == "haversine":
        dist_func = haversine_dist
    else:
        raise AttributeError("distance_metric unknown. We only support ['haversine']. " f"You passed {distance_metric}")

    df = df.sort_index(kind="mergesort").sort_values(by=["tracked_at"], kind="mergesort")
    # pfs id should be in index, create separate idx for storing the matching
    pfs = df.to_dict("records")
    idx = df.index.to_list()

    ret_sp = []
    start = 0

    # as start begin from 0, curr begin from 1
    for i in range(1, len(pfs)):
        curr = i

        # the duration of gap in the last two pfs
        gap_t = (pfs[curr]["tracked_at"] - pfs[curr - 1]["tracked_at"]).total_seconds()
        # the gap of two consecutive positionfixes should not be too long
        if gap_t > gap_threshold * 60:
            start = curr
            continue

        delta_dist = dist_func(pfs[start][geo_col].x, pfs[start][geo_col].y, pfs[curr][geo_col].x, pfs[curr][geo_col].y)

        if delta_dist >= dist_threshold:
            # the total duration of the staypoints
            delta_t = (pfs[curr]["tracked_at"] - pfs[start]["tracked_at"]).total_seconds()

            # we want the staypoint to have long duration,
            # but the gap of two consecutive positionfixes should not be too long
            if (delta_t >= (time_threshold * 60)) and (gap_t < gap_threshold * 60):
                new_sp = __create_new_staypoints(start, curr, pfs, idx, elevation_flag, geo_col)
                # add staypoint
                ret_sp.append(new_sp)

            # distance larger but time too short -> not a stay point
            # also initializer when new stp is added
            start = curr

        # if we arrive at the last positionfix, and want to include the last staypoint
        if (curr == len(pfs) - 1) and include_last:
            # additional control: we want to create staypoints with duration larger than time_threshold
            delta_t = (pfs[curr]["tracked_at"] - pfs[start]["tracked_at"]).total_seconds()
            if delta_t >= (time_threshold * 60):
                new_sp = __create_new_staypoints(start, curr, pfs, idx, elevation_flag, geo_col, last_flag=True)

                # add staypoint
                ret_sp.append(new_sp)

    ret_sp = pd.DataFrame(ret_sp)
    ret_sp["user_id"] = df["user_id"].unique()[0]
    return ret_sp


def __create_new_staypoints(start, end, pfs, idx, elevation_flag, geo_col, last_flag=False):
    """Create a staypoint with relevant infomation from start to end pfs."""
    new_sp = {}

    # Here we consider pfs[end] time for stp 'finished_at', but only include
    # pfs[end - 1] for stp geometry and pfs linkage.
    new_sp["started_at"] = pfs[start]["tracked_at"]
    new_sp["finished_at"] = pfs[end]["tracked_at"]

    # if end is the last pfs, we want to include the info from it as well
    if last_flag:
        end = len(pfs)

    new_sp[geo_col] = Point(
        np.median([pfs[k][geo_col].x for k in range(start, end)]),
        np.median([pfs[k][geo_col].y for k in range(start, end)]),
    )
    if elevation_flag:
        new_sp["elevation"] = np.median([pfs[k]["elevation"] for k in range(start, end)])
    # store matching, index should be the id of pfs
    new_sp["pfs_id"] = [idx[k] for k in range(start, end)]

    return new_sp


def _drop_invalid_triplegs(tpls, pfs):
    """Remove triplegs with invalid geometries. Also remove the corresponding invalid tripleg ids from positionfixes.

    Parameters
    ----------
    tpls : GeoDataFrame (as trackintel triplegs)
    pfs : GeoDataFrame (as trackintel positionfixes)

    Returns
    -------
    tpls: GeoDataFrame (as trackintel triplegs)
        original tpls with invalid geometries removed.

    pfs: GeoDataFrame (as trackintel positionfixes)
        original pfs with invalid tripleg id set to pd.NA.

    Notes
    -----
    Valid is defined using shapely (https://shapely.readthedocs.io/en/stable/manual.html#object.is_valid) via
    the geopandas accessor.
    """
    invalid_tpls = tpls[~tpls.geometry.is_valid]
    if invalid_tpls.shape[0] > 0:
        # identify invalid tripleg ids
        invalid_tpls_ids = invalid_tpls.index.to_list()

        # reset tpls id in pfs
        invalid_pfs_ixs = pfs[pfs.tripleg_id.isin(invalid_tpls_ids)].index
        pfs.loc[invalid_pfs_ixs, "tripleg_id"] = pd.NA
        warn_string = (
            f"The positionfixes with ids {invalid_pfs_ixs.values} lead to invalid tripleg geometries. The "
            f"resulting triplegs were omitted and the tripleg id of the positionfixes was set to nan"
        )
        warnings.warn(warn_string)

        # return valid triplegs
        tpls = tpls[tpls.geometry.is_valid]
    return tpls, pfs
