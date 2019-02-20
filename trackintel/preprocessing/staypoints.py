import numpy as np
import pandas as pd
import geopandas as gpd
import sklearn
import shapely

from shapely.geometry import Point
from sklearn.cluster import DBSCAN


def cluster_staypoints(staypoints, method='dbscan',
                       epsilon=100, num_samples=3):
    """Clusters staypoints to get places.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints have to follow the standard definition for staypoints DataFrames.

    method : {'dbscan'}
        The following methods are available to cluster staypoints into places:

        'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

    epsilon : float
        The epsilon for the 'dbscan' method.

    num_samples : int
        The minimal number of samples in a cluster.

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing places that a person visited multiple times.

    Examples
    --------
    >>> cluster_staypoints(...)    
    """
    ret_places = pd.DataFrame(columns=['user_id', 'geom'])

    # TODO We have to make sure that the user_id is taken into account.
    db = DBSCAN(eps=epsilon, min_samples=num_samples)
    coordinates = np.array([[g.x, g.y] for g in staypoints['geom']])
    labels = db.fit_predict(coordinates)
    labeled_staypoints = staypoints
    labeled_staypoints['cluster_id'] = labels

    grouped_df = labeled_staypoints.groupby('cluster_id')
    for cluster_id, group in grouped_df:
        if cluster_id is not -1:
            stps = group.to_dict('records')
            ret_place = {}
            ret_place['user_id'] = stps[0]['user_id']
            ret_place['geom'] = Point(np.mean([k['geom'].x for k in stps]), 
                                          np.mean([k['geom'].y for k in stps]))
            ret_places = ret_places.append(ret_place, ignore_index=True)

    ret_places = gpd.GeoDataFrame(ret_places, geometry='geom')
    return ret_places

