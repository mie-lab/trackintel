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