import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN

from trackintel.geogr.distances import haversine_dist


def generate_staypoints(positionfixes, method='sliding',
                        dist_threshold=50, time_threshold= 300, epsilon=100,
                        dist_func=haversine_dist, num_samples=None):
    """Generates staypoints from positionfixes.

    Parameters
    ----------
    
    positionfixes : GeoDataFrame
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    method : str, {'sliding' or 'dbscan'}, default 'sliding'
        - 'sliding' : Applies a sliding window over the data.
        - 'dbscan' : Uses the DBSCAN algorithm to find clusters of staypoints.

    dist_threshold : float, default 50
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint.

    time_threshold : float, default 300 (seconds)
        The time threshold for the 'sliding' method in seconds, i.e., how long someone has to 
        stay within an area to consider it as a staypoint.

    epsilon : float, default 100
        The epsilon for the 'dbscan' method.

    dist_func : function, defaut haversine_dist
        A function that expects (lon_1, lat_1, lon_2, lat_2) and computes a distance in meters.
        
    num_samples :
    
    Returns
    -------
    (GeoDataFrame, GeoDataFrame)
        Tuple of (positionfixes, staypoints). Positionfixes is the original GeoDataFrame with 
        a new column 'staypoint_id'. 

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
    
    if 'id' not in ret_pfs.columns:
        ret_pfs['id'] = ret_pfs.index
    
    elevation_flag = 'elevation' in ret_pfs.columns # if there is elevation data

    name_geocol = ret_pfs.geometry.name
    ret_spts = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])

    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        staypoint_id_counter = 0
        ret_pfs['staypoint_id'] = -1  # this marks all that are not part of a SP

        for user_id_this in ret_pfs['user_id'].unique():

            positionfixes_user_this = ret_pfs.loc[ret_pfs['user_id'] == user_id_this]  # this is no copy

            pfs = positionfixes_user_this.sort_values('tracked_at').to_dict('records')
            num_pfs = len(pfs)

            posfix_staypoint_matching = {}

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
                            staypoint['finished_at'] = pfs[j - 1][
                                'tracked_at']  # TODO: should this not be j-1? because j is not part of the staypoint. DB: Changed.
                            staypoint['id'] = staypoint_id_counter

                            # store matching 
                            posfix_staypoint_matching[staypoint_id_counter] = [pfs[k]['id'] for k in range(i, j)]
                            staypoint_id_counter += 1

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
                                staypoint['id'] = staypoint_id_counter

                                # store matching
                                posfix_staypoint_matching[staypoint_id_counter] = [
                                    pfs[j]['id']]  # rather [k for k in range(i, j)]?
                                staypoint_id_counter += 1
                                ret_spts = ret_spts.append(staypoint, ignore_index=True)
                        i = j
                        break
                    j = j + 1

            # add matching to original positionfixes (for every user)

            for staypoints_id, posfix_idlist in posfix_staypoint_matching.items():
                # note that we use .loc because above we have saved the id 
                # of the positionfixes not thier absolut position
                ret_pfs.loc[posfix_idlist, 'staypoint_id'] = staypoints_id


    elif method == 'dbscan':
        # TODO: Make sure time information is included in the clustering!
        # time information is in the column 'started at', however the user should be able to
        # adjust the distance metric e.g. chebychev

        db = DBSCAN(eps=epsilon, min_samples=num_samples)
        for user_id_this in ret_pfs['user_id'].unique():

            user_positionfixes = ret_pfs[ret_pfs['user_id'] == user_id_this]  # this is not a copy!

            # TODO: enable transformations to temporary (metric) system
            transform_crs = None
            if transform_crs is not None:
                pass

            # get staypoint matching
            coordinates = np.array([[g.x, g.y] for g in user_positionfixes['geom']])
            labels = db.fit_predict(coordinates)

            # add positionfixes - staypoint matching to original positionfixes
            ret_pfs.loc[user_positionfixes.index, 'staypoint_id'] = labels

        # create staypoints as the center of the grouped positionfixes
        grouped_df = ret_pfs.groupby(['user_id', 'staypoint_id'])
        for combined_id, group in grouped_df:
            user_id, staypoint_id = combined_id

            if int(staypoint_id) != -1:
                staypoint = {}
                staypoint['user_id'] = user_id
                staypoint['id'] = staypoint_id

                # point geometry of staypoint
                staypoint.geometry = Point(group[name_geocol].x.mean(),
                                           group[name_geocol].y.mean())

                ret_spts = ret_spts.append(staypoint, ignore_index=True)

    ret_spts = gpd.GeoDataFrame(ret_spts, geometry='geom',crs=ret_pfs.crs)
    
    ## ensure dtype consistency 
    # stps id (generated by this function) should be int64
    ret_spts['id'] = ret_spts['id'].astype('int64')
    ret_pfs['staypoint_id'] = ret_pfs['staypoint_id'].astype('int64')
    
    # user_id of spts should be the same as ret_pfs
    ret_spts['user_id'] = ret_spts['user_id'].astype(ret_pfs['user_id'].dtype)

    return ret_pfs, ret_spts


def generate_triplegs(positionfixes, staypoints=None, *args, **kwargs):
    """Generates triplegs from positionfixes. A tripleg is (for now) defined as anything
    that happens between two consecutive staypoints.

    **Attention**: This function requires either a column ``staypoint_id`` on the 
    positionfixes or passing some staypoints that correspond to the positionfixes! 
    This means you usually should call ``extract_staypoints()`` first.

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    staypoints : GeoDataFrame, optional
        The staypoints (corresponding to the positionfixes). If this is not passed, the 
        positionfixes need staypoint_id associated with them.

    Returns
    -------
    (GeoDataFrame, GeoDataFrame)
        Tuple of (positionfixes, triplegs). Positionfixes is the original GeoDataFrame with 
        a new column 'tripleg_id'.

    Examples
    --------
    >>> pfs.as_positionfixes.generate_triplegs(staypoints)
    """
    # copy the original pfs for adding 'staypoint_id' column
    ret_pfs = positionfixes.copy()
    
    name_geocol = ret_pfs.geometry.name
    # Check that data adheres to contract.
    if staypoints is None and len(ret_pfs['staypoint_id'].unique()) < 2:
        raise ValueError("If staypoints is not defined, positionfixes must have more than 1 staypoint_id.")

    # if staypoints is not None:
    #     raise NotImplementedError("Splitting up positionfixes by timestamp is not available yet. " + \
    #         "Use extract_staypoints and the thus generated staypoint_ids.")

    ret_tpls = pd.DataFrame(columns=['id', 'user_id', 'started_at', 'finished_at', 'geom'])
    curr_tripleg_id = 0
    # Do this for each user.
    for user_id_this in ret_pfs['user_id'].unique():

        positionfixes_user_this = ret_pfs.loc[ret_pfs['user_id'] == user_id_this]  # this is no copy
        pfs = positionfixes_user_this.sort_values('tracked_at')
        generated_triplegs = []

        # Case 1: Staypoints exist and are connected to positionfixes by user id
        if staypoints is not None and "staypoint_id" in pfs:
            stps = staypoints.loc[staypoints['user_id'] == user_id_this].sort_values('started_at')
            stps = stps.to_dict('records')
            for stp1, stp2 in zip(list(stps), list(stps)[1:]):
                # Get all positionfixes that lie between these two staypoints.

                # get the last posfix of the first staypoint
                index_first_posfix_tl = pfs[pfs.staypoint_id == stp1['id']].index[-1]
                position_first_posfix_tl = pfs.index.get_loc(index_first_posfix_tl)

                # get first posfix of the second staypoint
                index_last_posfix_tl = pfs[pfs.staypoint_id == stp2['id']].index[0]
                position_last_posfix_tl = pfs.index.get_loc(index_last_posfix_tl)

                pfs_tripleg = pfs.iloc[position_first_posfix_tl:position_last_posfix_tl + 1]

                # include every positionfix that brings you closer to the center 
                # of the staypoint

                started_at = pfs_tripleg['tracked_at'].iloc[0]
                finished_at = pfs_tripleg['tracked_at'].iloc[-1]

                coords = list(pfs_tripleg.geometry.apply(lambda r: (r.x, r.y)))

                if len(coords) > 1:
                    generated_triplegs.append({
                        'id': curr_tripleg_id,
                        'user_id': user_id_this,
                        'started_at': started_at,  # pfs_tripleg['tracked_at'].iloc[0],
                        'finished_at': finished_at,  # pfs_tripleg['tracked_at'].iloc[-1],
                        'geom': LineString(coords)
                    })
                    curr_tripleg_id += 1

        # Case 2: Staypoints exist but there is no user_id given
        # TODO Not so efficient, always matching on the time (as things are sorted anyways).
        elif staypoints is not None:
            stps = staypoints.loc[staypoints['user_id'] == user_id_this].sort_values('started_at')
            stps = stps.to_dict('records')
            for stp1, stp2 in zip(list(stps), list(stps)[1:]):
                # Get all positionfixes that lie between these two staypoints.
                pfs_tripleg = pfs[(stp1['finished_at'] <= pfs['tracked_at']) & \
                                  (pfs['tracked_at'] <= stp2['started_at'])].sort_values('tracked_at')

                coords = list(pfs_tripleg.geometry.apply(lambda r: (r.x, r.y)))
                if len(coords) > 1:
                    generated_triplegs.append({
                        'id': curr_tripleg_id,
                        'user_id': user_id_this,
                        'started_at': pfs_tripleg['tracked_at'].iloc[0],
                        'finished_at': pfs_tripleg['tracked_at'].iloc[-1],
                        'geom': LineString(list(pfs_tripleg.geometry.apply(lambda r: (r.x, r.y))))
                    })
                    curr_tripleg_id += 1

        # case 3: Only positionfixes with staypoint id for tripleg generation
        else:
            prev_pf = None
            curr_tripleg = {
                'id': curr_tripleg_id,
                'user_id': user_id_this,
                'started_at': pfs['tracked_at'].iloc[0],
                'finished_at': None,
                'coords': []
            }
            for idx, pf in pfs.iterrows():
                if prev_pf is not None and prev_pf['staypoint_id'] == -1 and pf['staypoint_id'] != -1:
                    # This tripleg ends. 
                    pfs.loc[idx, 'tripleg_id'] = curr_tripleg_id
                    curr_tripleg['finished_at'] = pf['tracked_at']
                    curr_tripleg['coords'].append((pf[name_geocol].x, pf[name_geocol].y))

                elif (prev_pf is not None and prev_pf['staypoint_id'] != -1 and pf['staypoint_id'] == -1):
                    # A new tripleg starts (due to a staypoint_id switch from -1 to x).
                    if len(curr_tripleg['coords']) > 1:
                        curr_tripleg[name_geocol] = LineString(curr_tripleg['coords'])
                        del curr_tripleg['coords']
                        generated_triplegs.append(curr_tripleg)
                        curr_tripleg_id += 1

                    curr_tripleg = {'id': curr_tripleg_id, 'user_id': user_id_this, 'started_at': None,
                                    'finished_at': None, 'coords': []}
                    prev_pf['tripleg_id'] = curr_tripleg_id
                    pfs.loc[idx, 'tripleg_id'] = curr_tripleg_id
                    curr_tripleg['started_at'] = pf['tracked_at']
                    curr_tripleg['coords'].append((pf[name_geocol].x, pf[name_geocol].y))

                elif prev_pf is not None and prev_pf['staypoint_id'] != -1 and \
                        pf['staypoint_id'] != -1 and prev_pf['staypoint_id'] != pf['staypoint_id']:
                    # A new tripleg starts (due to a staypoint_id switch from x to y).
                    pfs.loc[idx, 'tripleg_id'] = curr_tripleg_id
                    curr_tripleg['finished_at'] = pf['tracked_at']
                    curr_tripleg['coords'].append((pf[name_geocol].x, pf[name_geocol].y))

                    if len(curr_tripleg['coords']) > 1:
                        curr_tripleg[name_geocol] = LineString(curr_tripleg['coords'])
                        del curr_tripleg['coords']
                        generated_triplegs.append(curr_tripleg)
                        curr_tripleg_id += 1

                    curr_tripleg = {
                        'id': curr_tripleg_id,
                        'user_id': user_id_this,
                        'started_at': None,
                        'finished_at': None,
                        'coords': []
                    }
                    prev_pf['tripleg_id'] = curr_tripleg_id
                    pfs.loc[idx, 'tripleg_id'] = curr_tripleg_id
                    curr_tripleg['started_at'] = pf['tracked_at']
                    curr_tripleg['coords'].append((pf[name_geocol].x, pf[name_geocol].y))

                elif prev_pf is not None and prev_pf['staypoint_id'] != -1 and \
                        prev_pf['staypoint_id'] == pf['staypoint_id']:
                    # This is still "at the same staypoint". Do nothing.
                    pass

                else:
                    pfs.loc[idx, 'tripleg_id'] = curr_tripleg_id
                    curr_tripleg['coords'].append((pf[name_geocol].x, pf[name_geocol].y))

                prev_pf = pf
        if len(generated_triplegs) > 0:
            ret_tpls = ret_tpls.append(generated_triplegs)

    ret_tpls = gpd.GeoDataFrame(ret_tpls, geometry='geom', crs=ret_pfs.crs)
    
    ## ensure dtype consistency 
    # tpls id (generated by this function) should be int64
    ret_tpls['id'] = ret_tpls['id'].astype('int64')
    # ISSUE 56: no tripleg_id is assigned for case 1
    # ret_pfs['tripleg_id'] = ret_pfs['tripleg_id'].astype('int64')
    
    # user_id of tpls should be the same as ret_pfs
    ret_tpls['user_id'] = ret_tpls['user_id'].astype(ret_pfs['user_id'].dtype)
    
    
    # todo: triplegs dataframe has use the index as id
    # todo: proposed fix: ret_triplegs = ret_triplegs.set_index('id')
    return ret_pfs, ret_tpls
