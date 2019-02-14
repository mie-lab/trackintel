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
    ret_staypoints = pd.DataFrame(columns=['longitude', 'latitude', 'arrival_at', 'departure_at'])

    if method == 'sliding':
        # Algorithm from Li et al. (2008). For details, please refer to the paper.
        pfs = positionfixes.sort_values('tracked_at').to_dict('records')
        num_pfs = len(positionfixes)

        i = 0
        while i < num_pfs - 1:
            j = i + 1
            while j < num_pfs:
                dist = dist_func(pfs[i]['longitude'], pfs[i]['latitude'], 
                                 pfs[j]['longitude'], pfs[j]['latitude'])

                if dist > dist_threshold:
                    delta_t = pfs[j]['tracked_at'] - pfs[i]['tracked_at']
                    if delta_t.total_seconds() > time_threshold:
                        staypoint = {}
                        staypoint['longitude'] = np.mean([pfs[k]['longitude'] for k in range(i, j)])
                        staypoint['latitude'] = np.mean([pfs[k]['latitude'] for k in range(i, j)])
                        staypoint['arrival_at'] = pfs[i]['tracked_at']
                        staypoint['departure_at'] = pfs[j]['tracked_at']
                        ret_staypoints = ret_staypoints.append(staypoint, ignore_index=True)
                    i = j
                    break
                j = j + 1

    elif method == 'dbscan':
        pass

    ret_staypoints['geom'] = list(zip(ret_staypoints.longitude, ret_staypoints.latitude))
    ret_staypoints['geom'] = ret_staypoints['geom'].apply(Point)
    ret_staypoints = gpd.GeoDataFrame(ret_staypoints, geometry='geom')

    return ret_staypoints