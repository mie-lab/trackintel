import numpy as np
import pandas as pd
import geopandas as gpd
import sklearn
import shapely

from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN

from trackintel.geogr.distances import calculate_distance_matrix

def cluster_staypoints(staypoints, method='dbscan',
                       epsilon=100, num_samples=3, distance_matrix_metric=None):
    """Clusters staypoints to get places.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints have to follow the standard definition for staypoints DataFrames.

    method : str, {'dbscan'}, default 'dbscan'
        The following methods are available to cluster staypoints into places:
        'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

    epsilon : float
        The epsilon for the 'dbscan' method.

    num_samples : int
        The minimal number of samples in a cluster. 

    distance_matrix_metric: string (optional)
        When given, dbscan will work on a precomputed a distance matrix that is
        created using the staypoints based on the given metric. Possible metrics
        are: {'haversine', 'euclidean'} or any mentioned in: 
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.pairwise_distances.html

    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing places that a person visited multiple times.
        
    Examples
    --------
    >>> spts.as_staypoints.cluster_staypoints(method='dbscan', epsilon=50, num_samples=3)
    """
    ret_places = pd.DataFrame(columns=['user_id', 'place_id','center', 'extent'])

    if method=='dbscan':

        if distance_matrix_metric is not None:
            db = DBSCAN(eps=epsilon, min_samples=num_samples,
                        metric='precomputed')
        else:    
            db = DBSCAN(eps=epsilon, min_samples=num_samples)
            
        place_id_counter = 0
            
        for user_id_this in staypoints["user_id"].unique():
            # Slice staypoints array by user. This is not a copy!
            user_staypoints = staypoints[staypoints["user_id"] == user_id_this]  
            
            if distance_matrix_metric is not None:
                sp_distance_matrix = calculate_distance_matrix(
                        user_staypoints, dist_metric=distance_matrix_metric)
                labels = db.fit_predict(sp_distance_matrix)
            
            else:  
                coordinates = np.array([[g.x, g.y] for g in user_staypoints['geom']])
                labels = db.fit_predict(coordinates)
                
            # enforce unique lables across all users without changing noise
            # labels
            max_label = np.max(labels)
            labels[labels != -1] = labels[labels != -1] + place_id_counter +1
            if max_label > -1:
                place_id_counter = place_id_counter + max_label + 1
            
            # add staypoint - place matching to original staypoints
            staypoints.loc[user_staypoints.index,'place_id'] = labels

            
    
        # create places as grouped staypoints
        grouped_df = staypoints.groupby(['user_id','place_id'])
        for combined_id, group in grouped_df:
            user_id, place_id = combined_id
    
            if int(place_id) != -1:
                ret_place = {}
                ret_place['user_id'] = user_id
                ret_place['place_id'] = place_id
                
                # point geometry of place
                ret_place['center'] = Point(group.geometry.x.mean(),
                     group.geometry.y.mean())
                # polygon geometry of place
                ret_place['extent'] = MultiPoint(points=list(group.geometry)).convex_hull
    
                ret_places = ret_places.append(ret_place, ignore_index=True)
    
        ret_places = gpd.GeoDataFrame(ret_places, geometry='center',
                                      crs=staypoints.crs)
        ret_places['place_id'] = ret_places['place_id'].astype('int')
        
    return ret_places

