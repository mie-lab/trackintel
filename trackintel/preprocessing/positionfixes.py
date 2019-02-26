import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

from shapely.geometry import Point

import trackintel.geogr.distances
from trackintel.geogr.distances import haversine_dist


def extract_staypoints(positionfixes, method='sliding', 
                       dist_threshold=100, time_threshold=5*60, epsilon=100,
                       dist_func=haversine_dist):
    """Extract staypoints from positionfixes.

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    method : {'sliding' or 'dbscan'}
        The following methods are available to extract staypoints from positionfixes:

        'sliding' : Applies a sliding window over the data.
        'dbscan' : Uses the DBSCAN algorithm to find clusters of staypoints.

    dist_threshold : float
        The distance threshold for the 'sliding' method, i.e., how far someone has to travel to
        generate a new staypoint.

    time_threshold : float
        The time threshold for the 'sliding' method in seconds, i.e., how long someone has to 
        stay within an area to consider it as a staypoint.

    epsilon : float
        The epsilon for the 'dbscan' method.

    dist_func : function
        A function that expects (lon_1, lat_1, lon_2, lat_2) and computes a distance in meters.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing points where a person spent some time.

    Examples
    --------
    >>> extract_staypoints(...)

    References
    ----------
    Zheng, Y. (2015). Trajectory data mining: an overview. ACM Transactions on Intelligent Systems 
    and Technology (TIST), 6(3), 29.

    Li, Q., Zheng, Y., Xie, X., Chen, Y., Liu, W., & Ma, W. Y. (2008, November). Mining user 
    similarity based on location history. In Proceedings of the 16th ACM SIGSPATIAL international 
    conference on Advances in geographic information systems (p. 34). ACM.
    """
    ret_staypoints = pd.DataFrame(columns=['arrival_at', 'departure_at', 'geom'])

    # TODO We have to make sure that the user_id is taken into account.

    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        pfs = positionfixes.sort_values('tracked_at').to_dict('records')
        num_pfs = len(positionfixes)

        i = 0
        j = 1
        while i < num_pfs:
            if j == num_pfs:
                # We're at the end, this can happen if in the last "bin", 
                # the dist_threshold is never crossed anymore.
                break
            else:
                j = i + 1
            while j < num_pfs:
                dist = dist_func(pfs[i]['geom'].x, pfs[i]['geom'].y, 
                                 pfs[j]['geom'].x, pfs[j]['geom'].y)

                if dist > dist_threshold:
                    delta_t = pfs[j]['tracked_at'] - pfs[i]['tracked_at']
                    if delta_t.total_seconds() > time_threshold:
                        staypoint = {}
                        staypoint['user_id'] = pfs[i]['user_id']
                        staypoint['geom'] = Point(np.mean([pfs[k]['geom'].x for k in range(i, j)]), 
                                                      np.mean([pfs[k]['geom'].y for k in range(i, j)]))
                        staypoint['elevation'] = np.mean([pfs[k]['elevation'] for k in range(i, j)])
                        staypoint['started_at'] = pfs[i]['tracked_at']
                        staypoint['finished_at'] = pfs[j]['tracked_at']
                        ret_staypoints = ret_staypoints.append(staypoint, ignore_index=True)
                        # TODO Discussion: Is this last point really a staypoint? As we don't know if the
                        #      person "moves on" afterwards...
                        if j == num_pfs - 1:
                            staypoint = {}
                            staypoint['user_id'] = pfs[j]['user_id']
                            staypoint['geom'] = Point(pfs[j]['geom'].x, pfs[j]['geom'].y)
                            staypoint['elevation'] = pfs[j]['elevation']
                            staypoint['started_at'] = pfs[j]['tracked_at']
                            staypoint['finished_at'] = pfs[j]['tracked_at']
                            ret_staypoints = ret_staypoints.append(staypoint, ignore_index=True)
                    i = j
                    break
                j = j + 1

    elif method == 'dbscan':
        pass


    ret_staypoints = gpd.GeoDataFrame(ret_staypoints, geometry='geom')

    return ret_staypoints