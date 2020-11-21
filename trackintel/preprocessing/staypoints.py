import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN

from trackintel.geogr.distances import calculate_distance_matrix, meters_to_decimal_degrees

def cluster_staypoints(staypoints, 
                       method='dbscan',
                       epsilon=100, 
                       num_samples=1, 
                       distance_matrix_metric=None,
                       agg_level='user'):
    """Clusters staypoints to get locations.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints have to follow the standard definition for staypoints DataFrames.

    method : str, {'dbscan'}, default 'dbscan'
        The following methods are available to cluster staypoints into locations:
        'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

    epsilon : float, default 100
        The epsilon for the 'dbscan' method.

    num_samples : int, default 1
        The minimal number of samples in a cluster. 

    distance_matrix_metric: str (optional)
        When given, dbscan will work on a precomputed a distance matrix that is
        created using the staypoints based on the given metric. Possible metrics
        are: {'haversine', 'euclidean'} or any mentioned in: 
        https://scikit-learn.org/stable/modules/generated/
        sklearn.metrics.pairwise_distances.html
        
    agg_level: str, {'user' or 'dataset'}, default 'user'
        The level of aggregation when generating locations:
        'user'      : locations are generated independently per-user.
        'dataset'   : shared locations are generated for all users.
    
    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing locations that a person visited multiple times.
        
    Examples
    --------
    >>> spts.as_staypoints.cluster_staypoints(method='dbscan', epsilon=100, num_samples=1)
    """
    
    if agg_level not in ['user', 'dataset']:
        raise AttributeError("The parameter agg_level must be one of ['user', 'dataset'].")
    
    ret_sp = staypoints.copy()
    if method=='dbscan':

        if distance_matrix_metric is not None:
            db = DBSCAN(eps=epsilon, min_samples=num_samples, metric='precomputed')
        else:    
            db = DBSCAN(eps=epsilon, min_samples=num_samples)
            
        if agg_level == 'user':
            location_id_counter = 0
            for user_id_this in ret_sp["user_id"].unique():
                # Slice staypoints array by user. This is not a copy!
                user_staypoints = ret_sp[ret_sp["user_id"] == user_id_this]  
                
                if distance_matrix_metric is not None:
                    sp_distance_matrix = calculate_distance_matrix(user_staypoints, 
                                                                   dist_metric=distance_matrix_metric)
                    labels = db.fit_predict(sp_distance_matrix)
                else:
                    coordinates = np.array([[g.x, g.y] for g in user_staypoints.geometry])
                    labels = db.fit_predict(coordinates)
                    
                # enforce unique lables across all users without changing noise labels
                max_label = np.max(labels)
                labels[labels != -1] = labels[labels != -1] + location_id_counter +1
                if max_label > -1:
                    location_id_counter = location_id_counter + max_label + 1
                
                # add staypoint - location matching to original staypoints
                ret_sp.loc[user_staypoints.index, 'location_id'] = labels
        else:
            if distance_matrix_metric is not None:
                sp_distance_matrix = calculate_distance_matrix(ret_sp, 
                                                               dist_metric=distance_matrix_metric)
                labels = db.fit_predict(sp_distance_matrix)
            else:
                coordinates = np.array([[g.x, g.y] for g in ret_sp.geometry])
                labels = db.fit_predict(coordinates)
            
            # add 1 to match the 'user' level result
            ret_sp['location_id'] = labels + 1
            
        # create locations as grouped staypoints
        temp_sp = ret_sp[['user_id', 'location_id', ret_sp.geometry.name]]
        ret_loc = temp_sp.dissolve(by=['user_id', 'location_id'],as_index=False)
        # filter outlier
        ret_loc = ret_loc.loc[ret_loc['location_id'] != -1]
        
        # locations with only one staypoints is of type "Point"
        point_idx = ret_loc.geom_type == 'Point'
        ret_loc['center'] = 0 # initialize
        ret_loc.loc[point_idx, 'center'] = ret_loc.loc[point_idx, 'geom']
        # locations with multiple staypoints is of type "MultiPoint"
        ret_loc.loc[~point_idx, 'center'] = ret_loc.loc[~point_idx, 'geom'].apply(
            lambda p: Point(np.array(p)[:,0].mean(), np.array(p)[:,1].mean()))
        
        # extent is the convex hull of the geometry
        ret_loc['extent'] = ret_loc['geom'].apply(lambda p: p.convex_hull)                                                                                        
        # convex_hull of one point would be a Point and two points a Linestring, 
        # we change them into Polygon by creating a buffer of epsilon around them.
        pointLine_idx = (ret_loc['extent'].geom_type == 'LineString') | (ret_loc['extent'].geom_type == 'Point')
        
        # Perform meter to decimal conversion if the distance metric is haversine
        if distance_matrix_metric == 'haversine':
            ret_loc.loc[pointLine_idx, 'extent'] = ret_loc.loc[pointLine_idx].apply(
                lambda p: p['extent'].buffer(meters_to_decimal_degrees(epsilon, p['center'].y)), axis=1)
        else:
            ret_loc.loc[pointLine_idx, 'extent'] = ret_loc.loc[pointLine_idx].apply(
                lambda p: p['extent'].buffer(epsilon), axis=1)
        
        ret_loc = ret_loc.set_geometry('center')
        ret_loc = ret_loc[['user_id', 'location_id', 'center', 'extent']]
        ret_loc['location_id'] = ret_loc['location_id'].astype('int')
        
    return ret_sp, ret_loc


def create_activity_flag(staypoints, method='time_threshold', time_threshold=5, activity_column_name='activity'):
    """

    Parameters
    ----------
    staypoints
    method
    time_threshold
    activity_column_name

    Returns
    -------

    """


    if method == 'time_threshold':
        staypoints[activity_column_name] = staypoints['finished_at'] - staypoints['started_at'] \
                                           > datetime.timedelta(minutes=time_threshold)
    else:
        raise NameError("Method {} is not known".format(method))

    return staypoints
