import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN

import datetime
from math import radians

from trackintel.geogr.distances import haversine_dist


def generate_staypoints(positionfixes,
                        method='sliding',
                        dist_func=haversine_dist,
                        dist_threshold=50,
                        time_threshold=300,
                        epsilon=100,
                        num_samples=1):
    """
    Generate staypoints from positionfixes.
    
    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    method : {'sliding' or 'dbscan'}
        Method to create staypoints. 
        
        - 'sliding' : Applies a sliding window over the data.
        - 'dbscan' : Uses the DBSCAN algorithm to find clusters of staypoints.
        
    dist_func : {'haversine_dist'}
        The distance metric used by the applied method.
        
    dist_threshold : float, default 50
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint. Units depend on the dist_func parameter.

    time_threshold : float, default 300 (seconds)
        The time threshold for the 'sliding' method in seconds
        
    epsilon : float, default 100
        The epsilon for the 'dbscan' method. Units depend on the dist_func parameter.
        
    num_samples : int, default 1
        The num_samples for the 'dbscan' method. The minimal number of samples in a cluster. 
    
    Returns
    -------
    ret_pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`staypoint_id`]``.
        
    ret_spts: GeoDataFrame (as trackintel staypoints)
        The generated staypoints.

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
    ret_pfs = positionfixes.copy()

    elevation_flag = 'elevation' in ret_pfs.columns  # if there is elevation data

    name_geocol = ret_pfs.geometry.name
    ret_spts = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])

    # TODO: tests using a different distance function, e.g., L2 distance
    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        ret_spts = ret_pfs.groupby('user_id', as_index=False).apply(_generate_staypoints_sliding_user,
                                                                    name_geocol,
                                                                    elevation_flag,
                                                                    dist_threshold,
                                                                    time_threshold,
                                                                    dist_func).reset_index(drop=True)
        # index management
        ret_spts['id'] = np.arange(len(ret_spts))
        ret_spts.set_index('id', inplace=True)

        # Assign staypoint_id to ret_pfs if spts is detected 
        if not ret_spts.empty:
            stps2pfs_map = ret_spts[['pfs_id']].to_dict()['pfs_id']

            ls = []
            for key, values in stps2pfs_map.items():
                for value in values:
                    ls.append([value, key])
            temp = pd.DataFrame(ls, columns=['id', 'staypoint_id']).set_index('id')
            # pfs with no stps receives nan in 'staypoint_id'
            ret_pfs = ret_pfs.join(temp, how='left')
            ret_spts.drop(columns={'pfs_id'}, inplace=True)
        # if no staypoint is identified
        else:
            ret_pfs['staypoint_id'] = np.nan

    # TODO: create tests for dbscan method
    # TODO: currently only support haversine distance, provode support for other distances, 
    # we could use the same structure as generate_location function
    elif method == 'dbscan':
        # TODO: Make sure time information is included in the clustering!
        # time information is in the column 'started at', however the user should be able to
        # adjust the distance metric e.g. chebychev

        # TODO: fix bug: generated staypoints has id starting from 0 for each user
        ret_pfs = ret_pfs.groupby("user_id").apply(_generate_staypoints_dbscan_user,
                                                   name_geocol,
                                                   epsilon,
                                                   num_samples)

        # TODO: staypoint 'elevation' field
        # TODO: using dissolve for staypoint generation
        # create staypoints as the center of the grouped positionfixes
        grouped_df = ret_pfs.groupby(['user_id', 'staypoint_id'])
        for combined_id, group in grouped_df:
            user_id, staypoint_id = combined_id

            if int(staypoint_id) != -1:
                staypoint = {}
                staypoint['user_id'] = user_id
                staypoint['id'] = staypoint_id

                # point geometry of staypoint
                staypoint[name_geocol] = Point(group[name_geocol].x.mean(),
                                               group[name_geocol].y.mean())

                ret_spts = ret_spts.append(staypoint, ignore_index=True)
        ret_spts.set_index('id', inplace=True)
        
    ret_pfs = gpd.GeoDataFrame(ret_pfs, geometry=name_geocol,crs=ret_pfs.crs)
    ret_spts = gpd.GeoDataFrame(ret_spts, geometry=name_geocol,crs=ret_pfs.crs)
    
    ## dtype consistency 
    # stps id (generated by this function) should be int64
    ret_spts.index = ret_spts.index.astype('int64')
    # ret_pfs['staypoint_id'] should be Int64 (missing values)
    ret_pfs['staypoint_id'] = ret_pfs['staypoint_id'].astype('Int64')

    # user_id of spts should be the same as ret_pfs
    ret_spts['user_id'] = ret_spts['user_id'].astype(ret_pfs['user_id'].dtype)

    return ret_pfs, ret_spts

def generate_triplegs(pfs_input, stps_input, method="between_staypoints", gap_threshold=15):
    """Generate triplegs from positionfixes.

    Parameters
    ----------
    pfs_input : GeoDataFrame (as trackintel positionfixes)
        The pfs have to follow the standard definition for positionfixes DataFrames. 
        If 'staypoint_id' column is not found, stps_input needs to be given.

    stps_input : GeoDataFrame (as trackintel staypoints), optional
        The stps (corresponding to the positionfixes). If this is not passed, the
        positionfixes need 'staypoint_id' associated with them.

    method: {'between_staypoints'}
        Method to create triplegs. 'between_staypoints' method defines a tripleg as all pfs 
        between two stps. This method requires either a column 'staypoint_id' on 
        the pfs or passing stps as an input.
            
    gap_threshold: float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than 
        `gap_threshold` minutes, then a new tripleg will be generated.

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
    
    The first pfs after a stp is regarded as the first pfs of the generated tpl.

    Examples
    --------
    >>> pfs.as_positionfixes.generate_triplegs('between_staypoints', gap_threshold=15)
    """
    # copy the original pfs for adding 'staypoint_id' column
    pfs = pfs_input.copy()

    if method == "between_staypoints":

        # get case:
        # Case 1: pfs have a column 'staypoint_id'
        # Case 2: pfs do not have a column 'staypoint_id' but stps_input are provided

        if "staypoint_id" not in pfs.columns:
            case = 2
        else:
            case = 1

        # Preprocessing for case 2:
        # - step 1: Assign staypoint ids to positionfixes by matching timestamps (per user)
        # - step 2: Find first positionfix after a staypoint
        # (relevant if the pfs of stps are not provided, and we can only infer the pfs after stps through time)
        if case == 2:
            # initialize the index list of pfs where a tpl will begin
            insert_index_ls = []
            pfs["staypoint_id"] = pd.NA
            for user_id_this in pfs["user_id"].unique():
                spts_user = stps_input[stps_input["user_id"] == user_id_this]
                pfs_user = pfs[pfs["user_id"] == user_id_this]

                # step 1
                # All positionfixes with timestamp between staypoints are assigned the value 0
                # Intersect all positionfixes of a user with all staypoints of the same user
                intervals = pd.IntervalIndex.from_arrays(spts_user["started_at"], spts_user["finished_at"], closed="both")
                is_in_interval = pfs_user["tracked_at"].apply(lambda x: intervals.contains(x).any()).astype("bool")
                pfs.loc[is_in_interval[is_in_interval].index, "staypoint_id"] = 0

                # step 2
                # Identify first positionfix after a staypoint
                # find index of closest positionfix with equal or greater timestamp.
                tracked_at_sorted = pfs_user["tracked_at"].sort_values()
                insert_position_user = tracked_at_sorted.searchsorted(spts_user["finished_at"])
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

        # we need to ensure pfs is properly ordered
        pfs.sort_values(by=["user_id", "tracked_at"], inplace=True)
        # get all conditions that trigger a new tripleg.
        # condition 1: a positionfix belongs to a new tripleg if the user changes. For this we need to sort pfs.
        # The first positionfix of the new user is the start of a new tripleg (if it is no staypoint)
        cond_new_user = ((pfs["user_id"] - pfs["user_id"].shift(1)) != 0) & pd.isna(pfs["staypoint_id"])

        # condition 2: Temporal gaps
        # if there is a gap that is longer than gap_threshold minutes, we start a new tripleg
        cond_gap = pfs["tracked_at"] - pfs["tracked_at"].shift(1) > datetime.timedelta(minutes=gap_threshold)

        # condition 3: stps
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

        cond_all.sort_index(inplace=True)
        # get the start position of tpls
        tpls_starts = np.where(cond_all)[0]

        # a valid linestring needs 2 points
        tpls_lengths = np.diff(tpls_starts)
        cond_to_remove = np.take(tpls_starts, np.where(tpls_lengths < 2)[0])
        cond_all.iloc[cond_to_remove] = False
        pfs.loc[pfs.index.isin(cond_to_remove), "tripleg_id"] = -1

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
            {"user_id": ["mean"], "tracked_at": [min, max], "geom": list}
        )  # could add a "number of pfs": can be any column "count"

        # prepare dataframe: Rename columns; read/set geometry/crs;
        # Order of column has to correspond to the order of the groupby statement
        tpls.columns = ["user_id", "started_at", "finished_at", "geom"]
        tpls["geom"] = tpls["geom"].apply(LineString)
        tpls = tpls.set_geometry("geom")
        tpls.crs = pfs.crs

        # check the correctness of the generated tpls
        assert tpls.as_triplegs

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


def _generate_staypoints_sliding_user(df,
                                      name_geocol,
                                      elevation_flag,
                                      dist_threshold=50,
                                      time_threshold=300,
                                      dist_func=haversine_dist):
    ret_spts = pd.DataFrame(columns=['user_id', 'started_at', 'finished_at', 'geom'])
    df.sort_values('tracked_at', inplace=True)

    # pfs id should be in index, create separate idx for storing the matching
    pfs = df.to_dict('records')
    idx = df.index.to_list()

    num_pfs = len(pfs)

    i = 0
    j = 0  # is zero because it gets incremented in the beginning
    while i < num_pfs:
        if j == num_pfs:
            # We're at the end, this can happen if in the last "bin", 
            # the dist_threshold is never crossed anymore.
            break
        else:
            j = i + 1

        while j < num_pfs:
            # TODO: Can we make distance function independent of projection?
            dist = dist_func(pfs[i][name_geocol].x, pfs[i][name_geocol].y,
                             pfs[j][name_geocol].x, pfs[j][name_geocol].y)

            if dist > dist_threshold:
                delta_t = pfs[j]['tracked_at'] - pfs[i]['tracked_at']
                if delta_t.total_seconds() > time_threshold:
                    staypoint = {}
                    staypoint['user_id'] = pfs[i]['user_id']
                    staypoint[name_geocol] = Point(np.mean([pfs[k][name_geocol].x for k in range(i, j)]),
                                                   np.mean([pfs[k][name_geocol].y for k in range(i, j)]))
                    if elevation_flag:
                        staypoint['elevation'] = np.mean([pfs[k]['elevation'] for k in range(i, j)])
                    staypoint['started_at'] = pfs[i]['tracked_at']
                    staypoint['finished_at'] = pfs[j - 1]['tracked_at']

                    # store matching, index should be the id of pfs
                    staypoint['pfs_id'] = [idx[k] for k in range(i, j)]

                    # add staypoint
                    ret_spts = ret_spts.append(staypoint, ignore_index=True)

                    # TODO Discussion: Is this last point really a staypoint? As we don't know if the
                    #      person "moves on" afterwards...
                    if j == num_pfs - 1:
                        staypoint = {}
                        staypoint['user_id'] = pfs[j]['user_id']
                        staypoint[name_geocol] = Point(pfs[j][name_geocol].x, pfs[j][name_geocol].y)
                        if elevation_flag:
                            staypoint['elevation'] = pfs[j]['elevation']
                        staypoint['started_at'] = pfs[j]['tracked_at']
                        staypoint['finished_at'] = pfs[j]['tracked_at']
                        # store matching, index should be the id of pfs
                        staypoint['pfs_id'] = [idx[j]]

                        ret_spts = ret_spts.append(staypoint, ignore_index=True)
                i = j
                break
            j = j + 1

    return ret_spts


def _generate_staypoints_dbscan_user(pfs,
                                     name_geocol,
                                     epsilon=100,
                                     num_samples=1):
    db = DBSCAN(eps=epsilon / 6371000, min_samples=num_samples, algorithm='ball_tree', metric='haversine')

    # TODO: enable transformations to temporary (metric) system
    transform_crs = None
    if transform_crs is not None:
        pass

    # get staypoint matching
    p = np.array([[radians(g.y), radians(g.x)] for g in pfs[name_geocol]])
    labels = db.fit_predict(p)

    # add positionfixes - staypoint matching to original positionfixes
    pfs['staypoint_id'] = labels

    return pfs