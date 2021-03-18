import datetime
from math import radians

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN

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

    ret_pfs = gpd.GeoDataFrame(ret_pfs, geometry=name_geocol, crs=ret_pfs.crs)
    ret_spts = gpd.GeoDataFrame(ret_spts, geometry=name_geocol, crs=ret_pfs.crs)

    ## dtype consistency 
    # stps id (generated by this function) should be int64
    ret_spts.index = ret_spts.index.astype('int64')
    # ret_pfs['staypoint_id'] should be Int64 (missing values)
    ret_pfs['staypoint_id'] = ret_pfs['staypoint_id'].astype('Int64')

    # user_id of spts should be the same as ret_pfs
    ret_spts['user_id'] = ret_spts['user_id'].astype(ret_pfs['user_id'].dtype)

    return ret_pfs, ret_spts


def generate_triplegs(positionfixes_in, spts=None, method='between_staypoints'):
    """
    Generate triplegs from positionfixes.

    A tripleg is (for now) defined as anything that happens between two consecutive staypoints.

    **Attention**: This function requires either a column ``staypoint_id`` on the 
    positionfixes or passing some staypoints that correspond to the positionfixes! 
    This means you usually should call ``extract_staypoints()`` first.

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    staypoints : GeoDataFrame (as trackintel staypoints), optional
        The staypoints (corresponding to the positionfixes). If this is not passed, the
        positionfixes need staypoint_id associated with them.

    method: {'between_staypoints'}
        Method to create triplegs. 
        
        - 'between_staypoints': A tripleg is defined as all positionfixes \
            between two staypoints. This method requires either a column ``staypoint_id`` on \
            the positionfixes or passing staypoints as an input.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`tripleg_id`]``.
        
    ret_tpls: GeoDataFrame (as trackintel triplegs)
        The generated triplegs.

    Notes
    -----
    Methods ``between_staypoints`` creates a tripleg from all positionfixes between two sequential
    staypoinst. The latest positionfix of a staypoint is at the same time the first positionfix of corresponding
    tripleg. This means that the a staypoint and the following tripleg share 1 trackpoint.
    To use the method 'between_staypoints' you need to provide staypoints, positionfixes with a column 'staypoint_id'
    or both. It is recommended to provide both as it increases the performance.

    Examples
    --------
    """
    # copy the original pfs for adding 'staypoint_id' column
    pfs = positionfixes_in.copy()

    if method == 'between_staypoints':

        # get case: # todo: @SVen I think only case 2 is now relevant
        # Case 1: Staypoints are provided and are connected to positionfixes which have a column 'staypoint_id'
        # Case 2: Staypoints are provided but positionfixes do not have a column 'staypoint_id'
        # case 3: Staypoints are not provided but positionfixes have a column 'staypoint_id'

        if spts is not None and "staypoint_id" in pfs:
            case = 1
        elif spts is not None:
            case = 2
        elif "staypoint_id" in pfs:
            case = 3
        else:
            raise Exception('unknown case')

        # Preprocessing for case 2: Assign staypoint ids to positionfixes by matching timestamps
        if case == 2:
            insert_index_psf_list = []
            pfs['staypoint_id'] = np.nan
            for user_id_this in pfs['user_id'].unique():
                spts_u = spts[spts['user_id'] == user_id_this]
                pfs_u = pfs[pfs['user_id'] == user_id_this]

                # all positionfixes with timestamp between staypoints are assigned the value 0
                # Here we intersect all positionfixes of a user with all staypoins of the same user
                intervals = pd.IntervalIndex.from_arrays(spts_u['started_at'], spts_u['finished_at'], closed='both')
                is_in_interval = pfs_u['tracked_at'].apply(lambda x: intervals.contains(x).any()).astype('bool')
                pfs.loc[is_in_interval[is_in_interval].index, 'staypoint_id'] = 0

                #
                # find closest positionfix (gleich oder später)
                tracked_at_sorted = pfs_u['tracked_at'].sort_values()
                insert_position_psf_u = tracked_at_sorted.searchsorted(spts_u['finished_at'])
                insert_index_psf = tracked_at_sorted.iloc[insert_position_psf_u].index

                # store the insert insert_position_spts_u in an array
                insert_index_psf_list.extend(list(tracked_at_sorted.iloc[insert_position_psf_u].index))

            cond_staypoins_case2 = pd.Series(False, index=pfs.index)
            cond_staypoins_case2.loc[insert_index_psf_list] = True

        # initialize tripleg_id with np.nan and fill all psfs that belong to staypoins with -1 (nans will be
        # replaced later with tripleg ids)
        pfs['tripleg_id'] = np.nan
        pfs.loc[~pd.isna(pfs['staypoint_id']), 'tripleg_id'] = -1

        # get all conditions that trigger a new tripleg.

        # condition 1: a positionfix belongs to a new tripleg if the user changes.
        # The first positionfix of the new user is the start of a new tripleg (if it is no staypoint)
        cond_new_user = ((pfs['user_id'] - pfs['user_id'].shift(1)) != 0) & pd.isna(pfs['staypoint_id'])

        # condition 2: Temporal gaps
        # if there is a gap that is longer than gap_threshold minutes, we start a new tripleg
        cond_gap = pfs['tracked_at'] - pfs['tracked_at'].shift(1) > datetime.timedelta(
            minutes=15)

        # condition 3: staypoints
        # By our definition the last positionfix of a staypoint is the first positionfix of a tripleg.
        # todo This definition needs some more debate
        stp_id = (pfs['staypoint_id'] + 1).fillna(0)
        cond_stp = ((stp_id - stp_id.shift(-1)) != 0) & (stp_id != 0)



        # todo: add a "True" to cond_stp at the position that is given in the values of insert_position_spts

        # combine conditions and assign an incrementing id to all positionfixes that start a tripleg
        cond_all = cond_new_user | cond_gap | cond_stp

        # special check for case 2: positionfixes that belong to staypoints might not be present in the data.
        if case == 2:
            cond_all | cond_staypoins_case2

        # assign an incrementing id to all positionfixes that start a tripleg
        # create triplegs
        nb_tpls = cond_all.sum()
        pfs.loc[cond_all, 'tripleg_id'] = np.arange(nb_tpls)

        pfs['tripleg_id'] = pfs['tripleg_id'].fillna(method='ffill')
        pfs.loc[pfs['tripleg_id'] == -1, 'tripleg_id'] = np.nan

        posfix_grouper = pfs.groupby('tripleg_id')

        ret_tpls = posfix_grouper.agg({'user_id': 'mean', 'tracked_at': [min, max],
                                       'geom': list, 'user_id': 'count'}) # count can be any column

        # todo: check if all relevant columns are present
        # todo: tripleg list of geometries to linestring
        # todo: renaming of columns
        # todo: renaming and datatype of index
        # todo: drop too short triplegs
        # todo: what do we still need from the code below (e.g., dtype consistency)

        # # case 3: Only positionfixes with staypoint id for tripleg generation
        # elif case == 3:
        #     generated_triplegs.extend(_triplegs_between_staypoints_case3(positionfixes_user_this, user_id_this))

    #     # create tripleg dataframe
    #     columns_triplegs = ['user_id', 'started_at', 'finished_at', 'geom', 'pfs_ids']
    #     if len(generated_triplegs) == 0:
    #         ret_tpls = gpd.GeoDataFrame(columns=columns_triplegs,
    #                                     geometry='geom', crs=pfs.crs)
    #     else:
    #         ret_tpls = gpd.GeoDataFrame(generated_triplegs, geometry='geom', crs=pfs.crs)
    #         # sanity check for tripleg generation
    #         assert len(columns_triplegs) == len(ret_tpls.columns), "Unexpected or missing column in tripleg generation"
    #         for col in columns_triplegs:
    #             assert col in ret_tpls.columns, "Unexpected columns in tripleg generation."
    #
    #     # index management
    #     ret_tpls['id'] = np.arange(len(ret_tpls))
    #     ret_tpls.set_index('id', inplace=True)
    #
    #     # assign tripleg_id to positionfixes
    #     if not ret_tpls.empty:
    #         tripleg_ids = ret_tpls['pfs_ids'].explode()
    #         # swap index and values
    #         tripleg_ids = pd.Series(tripleg_ids.index, index=tripleg_ids.values, name='tripleg_id')
    #         pfs = pfs.join(tripleg_ids, how='left')
    #     else:
    #         pfs['tripleg_id'] = np.nan
    # ret_tpls = ret_tpls.drop('pfs_ids', axis=1)

        ## dtype consistency
        # # tpls id (generated by this function) should be int
        # ret_tpls.index = ret_tpls.index.astype('int64')
        # # pfs['tripleg_id'] should be Int64 (missing values)
        # pfs['tripleg_id'] = pfs['tripleg_id'].astype('Int64')
        # # user_id of tpls should be the same as pfs
        # ret_tpls['user_id'] = ret_tpls['user_id'].astype(pfs['user_id'].dtype)

        return pfs, ret_tpls

    else:
        raise NameError('Chosen method is not defined')


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

