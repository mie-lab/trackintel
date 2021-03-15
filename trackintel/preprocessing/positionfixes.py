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


def generate_triplegs(positionfixes, staypoints=None, method='between_staypoints'):
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
    ret_pfs: GeoDataFrame (as trackintel positionfixes)
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
    >>> pfs.as_positionfixes.generate_triplegs(staypoints)
    """
    # copy the original pfs for adding 'staypoint_id' column
    ret_pfs = positionfixes.copy()

    ret_tpls = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])
    if method == 'between_staypoints':

        # get case:
        # Case 1: Staypoints are provided and are connected to positionfixes which have a column 'staypoint_id'
        # Case 2: Staypoints are provided but positionfixes do not have a column 'staypoint_id'
        # case 3: Staypoints are not provided but positionfixes have a column 'staypoint_id'


        if staypoints is not None and "staypoint_id" in ret_pfs:
            case = 1
        elif staypoints is not None:
            case = 2
        elif "staypoint_id" in ret_pfs:
            case = 3
        else:
            raise Exception('unknown case')

        # generated_triplegs is a list to which we will append tripleg records. Every tripleg record is a dictionary
        # with the following keys: ['id', 'user_id', 'started_at', 'finished_at', 'geom', 'pfs_ids']
        generated_triplegs = []
        for user_id_this in ret_pfs['user_id'].unique():

            positionfixes_user_this = ret_pfs.loc[ret_pfs['user_id'] == user_id_this].sort_values(
                'tracked_at')  # this is no copy
            if positionfixes_user_this.empty:
                continue

            # Case 1: Staypoints exist and are connected to positionfixes by user id
            if case == 1:
                generated_triplegs.extend(_triplegs_between_staypoints_case1(positionfixes_user_this, staypoints,
                                                                             user_id_this))

            # Case 2: Staypoints exist but there is no user_id given
            elif case == 2:
                generated_triplegs.extend(_triplegs_between_staypoints_case2(positionfixes_user_this, staypoints,
                                                                             user_id_this))

            # case 3: Only positionfixes with staypoint id for tripleg generation
            elif case == 3:
                generated_triplegs.extend(_triplegs_between_staypoints_case3(positionfixes_user_this, user_id_this))

        # create tripleg dataframe
        columns_triplegs = ['user_id', 'started_at', 'finished_at', 'geom', 'pfs_ids']
        if len(generated_triplegs) == 0:
            ret_tpls = gpd.GeoDataFrame(columns=columns_triplegs,
                                        geometry='geom', crs=ret_pfs.crs)
        else:
            ret_tpls = gpd.GeoDataFrame(generated_triplegs, geometry='geom', crs=ret_pfs.crs)
            # sanity check for tripleg generation
            assert len(columns_triplegs) == len(ret_tpls.columns), "Unexpected or missing column in tripleg generation"
            for col in columns_triplegs:
                assert col in ret_tpls.columns, "Unexpected columns in tripleg generation."
        
        # index management    
        ret_tpls['id'] = np.arange(len(ret_tpls))
        ret_tpls.set_index('id', inplace=True)

        # assign tripleg_id to positionfixes
        if not ret_tpls.empty:
            tripleg_ids = ret_tpls['pfs_ids'].explode()
            # swap index and values
            tripleg_ids = pd.Series(tripleg_ids.index, index=tripleg_ids.values, name='tripleg_id')
            ret_pfs = ret_pfs.join(tripleg_ids, how='left')
        else:
            ret_pfs['tripleg_id'] = np.nan
        ret_tpls = ret_tpls.drop('pfs_ids', axis=1)

        ## dtype consistency
        # tpls id (generated by this function) should be int
        ret_tpls.index = ret_tpls.index.astype('int64')
        # ret_pfs['tripleg_id'] should be Int64 (missing values)
        ret_pfs['tripleg_id'] = ret_pfs['tripleg_id'].astype('Int64')
        # user_id of tpls should be the same as ret_pfs
        ret_tpls['user_id'] = ret_tpls['user_id'].astype(ret_pfs['user_id'].dtype)

        return ret_pfs, ret_tpls

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


def _triplegs_between_staypoints_case1(positionfixes, staypoints, user_id_this):
    """
    This function uses the staypoints and the column 'staypoint_id' in the positionfixes, to identify all
    positionfixes that lie in between two staypoints.

    Parameters
    ----------
    positionfixes: trackintel positionfixes
    staypoints: trackintel staypoints
    user_id_this:

    Returns
    --------
    list
        a list of dictionaries with individual triplegs
    """

    generated_triplegs_list = []
    spts = staypoints.loc[staypoints['user_id'] == user_id_this].sort_values('started_at')
    if spts.empty:
        return []

    spts = spts.reset_index().to_dict('records')

    for spt1, spt2 in zip(list(spts), list(spts)[1:]):
        # - Go through all pairs of consecutive staypoints.
        # - identify end of first and start of second staypoint.
        # - assign all positionfixes in between (including bounds) to a tripleg

        # get the last posfix of the first staypoint
        index_first_posfix_tl = positionfixes[positionfixes.staypoint_id == spt1['id']].index[-1]
        position_first_posfix_tl = positionfixes.index.get_loc(index_first_posfix_tl)

        # get first posfix of the second staypoint
        index_last_posfix_tl = positionfixes[positionfixes.staypoint_id == spt2['id']].index[0]
        position_last_posfix_tl = positionfixes.index.get_loc(index_last_posfix_tl)

        # create tripleg from all positionfixes in between the two staypoints
        pfs_tripleg = positionfixes.iloc[position_first_posfix_tl:position_last_posfix_tl + 1]
        generated_triplegs_list.append(__get_tripleg_record_from_psfs(pfs_tripleg, user_id_this, min_nb_of_points=3))

    # add first tripleg to the beginning of generated_tripleg_list
    index_first_posfix_first_stp = positionfixes[positionfixes.staypoint_id == spts[0]['id']].index[0]
    position_first_posfix_first_stp = positionfixes.index.get_loc(index_first_posfix_first_stp)

    pfs_tripleg = positionfixes.iloc[0:position_first_posfix_first_stp + 1]
    generated_triplegs_list = [__get_tripleg_record_from_psfs(pfs_tripleg, user_id_this, min_nb_of_points=2)] + \
                              generated_triplegs_list

    # add last tripleg to the end of generated_triplegs
    index_last_posfix_last_stp = positionfixes[positionfixes.staypoint_id == spts[-1]['id']].index[-1]
    position_last_posfix_last_stp = positionfixes.index.get_loc(index_last_posfix_last_stp)

    pfs_tripleg = positionfixes.iloc[position_last_posfix_last_stp:]
    generated_triplegs_list.append(__get_tripleg_record_from_psfs(pfs_tripleg, user_id_this, min_nb_of_points=2))

    # filter None values
    return list(filter(None, generated_triplegs_list))


def _triplegs_between_staypoints_case2(positionfixes, staypoints, user_id_this):
    """
    This function uses the timestamps of staypoints to identify all positionfixes that lie in between two staypoints.

    Parameters
    ----------
    positionfixes: trackintel positionfixes
    staypoints: trackintel staypoints
    user_id_this:

    Returns
    --------
    list
        a list of dictionaries with individual triplegs

    """
    generated_triplegs_list = []
    spts = staypoints.loc[staypoints['user_id'] == user_id_this].sort_values('started_at')
    if spts.empty:
        return []

    spts = spts.reset_index().to_dict('records')
    positionfixes = positionfixes.sort_values('tracked_at')
    for stp1, stp2 in zip(list(spts), list(spts)[1:]):
        # - Get all positionfixes that lie between these two staypoints by comparing timestamps.
        # - generate tripleg

        # Not so efficient, always matching on the time (as things are sorted anyways).
        pfs_tripleg = positionfixes[(stp1['finished_at'] <= positionfixes['tracked_at']) &
                                    (positionfixes['tracked_at'] <= stp2['started_at'])]
        generated_triplegs_list.append(__get_tripleg_record_from_psfs(pfs_tripleg, user_id_this, min_nb_of_points=3))

    # add first tripleg
    pfs_first_tripleg = positionfixes[positionfixes['tracked_at'] <= spts[0]['started_at']]
    generated_triplegs_list = [__get_tripleg_record_from_psfs(pfs_first_tripleg, user_id_this, min_nb_of_points=2
                                                              )] + generated_triplegs_list

    # add last tripleg
    pfs_first_tripleg = positionfixes[positionfixes['tracked_at'] >= spts[-1]['finished_at']]
    generated_triplegs_list.append(__get_tripleg_record_from_psfs(pfs_first_tripleg, user_id_this, min_nb_of_points=2))

    # filter None values
    return list(filter(None, generated_triplegs_list))


def __get_tripleg_record_from_psfs(pfs_tripleg, user_id_this, min_nb_of_points):
    """
    Create a tripleg from a collection of positionfixes

    Parameters
    ----------
    pfs_tripleg: geodataframe
        All positionfixes that are part of the tripleg
    user_id_this
    min_nb_of_points: int
        Minimum number of positionfixes required for a valid tripleg.
        3 positionfixes are required for a tripleg in between two staypoints to have at least 1 positionfix that is
        no part of a staypoint.
        2 positionfixes are required for a tripleg in the beginning or the end of the dataset (or a gap) as the
        first/last positionfix does then not belong to a staypoint.

    Returns
    -------
    dict or None
    """
    coords = list(pfs_tripleg.geometry.apply(lambda r: (r.x, r.y)))

    if len(coords) < min_nb_of_points:  # at least 1 posfix that is not part of a staypoint
        return None
    else:
        tripleg_entry = {
            'user_id': user_id_this,
            'started_at': pfs_tripleg['tracked_at'].iloc[0],
            'finished_at': pfs_tripleg['tracked_at'].iloc[-1],
            'geom': LineString(coords),
            'pfs_ids': list(pfs_tripleg.index)
        }
        return tripleg_entry


def _triplegs_between_staypoints_case3(positionfixes, user_id_this):
    """
    This function uses column 'staypoint_id' to identify all positionfixes that lie in between two staypoints.

    Parameters
    ----------
    positionfixes: trackintel positionfixes
    user_id_this:

    Returns
    --------
    list
        a list of dictionaries with individual triplegs
    """

    name_geocol = positionfixes.geometry.name
    generated_triplegs = []
    # initialize first tripleg
    curr_tripleg = {
        'user_id': user_id_this,
        'started_at': positionfixes['tracked_at'].iloc[0],
        'finished_at': None,
        'geom': [],
        'pfs_ids': []
    }

    first_iteration = True
    for idx, pf in positionfixes.iterrows():

        if first_iteration:
            first_iteration = False

            if pd.isna(pf['staypoint_id']):
                status = 'in_tripleg'
            elif not pd.isna(pf['staypoint_id']):
                status = 'in_staypoint'
        else:
            # - loop through all positionfixes
            # - identify the current situation and define the variable 'status'
            # - store or skip the current positionfix based on the state of 'status'

            # During the loop the status of a positionfix can be {'in_tripleg', 'in_staypoint', 'tripleg_starts',
            # 'tripleg_ends'}

            if not pd.isna(prev_pf['staypoint_id']):
                if pd.isna(pf['staypoint_id']):
                    status = 'tripleg_starts'
                else:
                    status = 'in_staypoint'
                    if prev_pf['staypoint_id'] != pf['staypoint_id']:
                        status = 'tripleg_starts'

            elif pd.isna(prev_pf['staypoint_id']):
                if not pd.isna(pf['staypoint_id']):
                    status = 'tripleg_ends'
                elif pd.isna(pf['staypoint_id']):
                    status = 'in_tripleg'
                else:
                    raise Exception("case not defined")
            else:
                raise Exception("case not defined")

        # take action depending on status
        if status == 'tripleg_starts':
            # initialize tripleg with last staypoint
            curr_tripleg = {
                'user_id': user_id_this,
                'started_at': prev_pf['tracked_at'],
                'finished_at': None,
                'geom': [(prev_pf[name_geocol].x, prev_pf[name_geocol].y), ],
                'pfs_ids': [prev_idx, ]
            }
            status = 'in_tripleg'

        if status == 'in_tripleg':
            curr_tripleg['geom'].append((pf[name_geocol].x, pf[name_geocol].y))
            curr_tripleg['pfs_ids'].append(idx)

        elif status == 'tripleg_ends':
            curr_tripleg['finished_at'] = pf['tracked_at']
            curr_tripleg['geom'].append((pf[name_geocol].x, pf[name_geocol].y))
            curr_tripleg['pfs_ids'].append(idx)
            curr_tripleg['geom'] = LineString([(x, y) for x, y in curr_tripleg['geom']])
            generated_triplegs.append(curr_tripleg)

            del curr_tripleg
        elif status == 'in_staypoint':
            pass

        prev_idx = idx
        prev_pf = pf

    # add a potential tripleg after the last staypoint
    if status == 'in_tripleg' and len(curr_tripleg) > 1:
        curr_tripleg['finished_at'] = pf['tracked_at']

        # NB: geom and id where already added during the loop
        curr_tripleg['geom'] = LineString([(x, y) for x, y in curr_tripleg['geom']])
        generated_triplegs.append(curr_tripleg)

    return generated_triplegs
