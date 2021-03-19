import datetime
from math import radians

import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from trackintel.geogr.distances import meters_to_decimal_degrees


def generate_locations(staypoints,
                       method='dbscan',
                       epsilon=100,
                       num_samples=1,
                       distance_matrix_metric='euclidean',
                       agg_level='user'):
    """
    Generate locations from the staypoints.

    Parameters
    ----------
    staypoints : GeoDataFrame (as trackintel staypoints)
        The staypoints have to follow the standard definition for staypoints DataFrames.

    method : {'dbscan'}
        Method to create locations.
        
        - 'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

    epsilon : float, default 100
        The epsilon for the 'dbscan' method. if 'distance_matrix_metric' is 'haversine' 
        or 'euclidean', the unit is in meters.

    num_samples : int, default 1
        The minimal number of samples in a cluster. 

    distance_matrix_metric: {'haversine', 'euclidean'}
        The distance matrix used by the applied method. Any mentioned below are possible: 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        
    agg_level: {'user','dataset'}
        The level of aggregation when generating locations:
        
        - 'user'      : locations are generated independently per-user.
        - 'dataset'   : shared locations are generated for all users.
    
    Returns
    -------
    ret_sp: GeoDataFrame (as trackintel staypoints)
        The original staypoints with a new column ``[`location_id`]``.
        
    ret_loc: GeoDataFrame (as trackintel locations)
        The generated locations. 
        
    Examples
    --------
    >>> spts.as_staypoints.generate_locations(method='dbscan', epsilon=100, num_samples=1)
    """
    if agg_level not in ['user', 'dataset']:
        raise AttributeError("The parameter agg_level must be one of ['user', 'dataset'].")
    if method not in ['dbscan']:
        raise AttributeError("The parameter method must be one of ['dbscan'].")
    
    # initialize the return GeoDataFrames
    ret_stps = staypoints.copy()
    
    if method=='dbscan':

        if distance_matrix_metric == 'haversine':
            # The input and output of sklearn's harvarsine metrix are both in radians,
            # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
            # here the 'epsilon' is directly applied to the metric's output.
            # convert to radius
            db = DBSCAN(eps=epsilon / 6371000 , min_samples=num_samples, algorithm='ball_tree', metric=distance_matrix_metric)
        else:
            db = DBSCAN(eps=epsilon, min_samples=num_samples, algorithm='ball_tree', metric=distance_matrix_metric)
            
        if agg_level == 'user':
            location_id_counter = 0
            # TODO: change into groupby
            for user_id_this in ret_stps["user_id"].unique():
                # Slice staypoints array by user. This is not a copy!
                user_staypoints = ret_stps[ret_stps["user_id"] == user_id_this]  
                
                if distance_matrix_metric == 'haversine':
                    # the input is converted to list of (lat, lon) tuples in radians unit
                    p = np.array([[radians(g.y), radians(g.x)] for g in user_staypoints.geometry])
                else:
                    p = np.array([[g.x, g.y] for g in user_staypoints.geometry])
                labels = db.fit_predict(p)
                    
                # enforce unique lables across all users without changing noise labels
                max_label = np.max(labels)
                labels[labels != -1] = labels[labels != -1] + location_id_counter
                if max_label > -1:
                    location_id_counter = location_id_counter + max_label + 1
                
                # add staypoint - location matching to original staypoints
                ret_stps.loc[user_staypoints.index, 'location_id'] = labels
        else:
            if distance_matrix_metric == 'haversine':
                # the input is converted to list of (lat, lon) tuples in radians unit
                p = np.array([[radians(g.y), radians(g.x)] for g in ret_stps.geometry])
            else:
                p = np.array([[g.x, g.y] for g in ret_stps.geometry])
            labels = db.fit_predict(p)
            
            ret_stps['location_id'] = labels
            
        ### create locations as grouped staypoints
        temp_sp = ret_stps[['user_id', 'location_id', ret_stps.geometry.name]]
        if agg_level == 'user':
            # directly dissolve by 'user_id' and 'location_id'
            ret_loc = temp_sp.dissolve(by=['user_id', 'location_id'],as_index=False)
        else:
            ## generate user-location pairs with same geometries across users
            # get user-location pairs
            ret_loc = temp_sp.dissolve(by=['user_id', 'location_id'], as_index=False).drop(columns={'geom'})
            # get location geometries
            geom_df = temp_sp.dissolve(by=['location_id'], as_index=False).drop(columns={'user_id'})
            # merge pairs with location geometries
            ret_loc = ret_loc.merge(geom_df, on='location_id', how='left')
            
        # filter stps not belonging to locations
        ret_loc = ret_loc.loc[ret_loc['location_id'] != -1]
        
        ret_loc['center'] = None # initialize
        # locations with only one staypoints is of type "Point"
        point_idx = ret_loc.geom_type == 'Point'
        if not ret_loc.loc[point_idx].empty:
            ret_loc.loc[point_idx, 'center'] = ret_loc.loc[point_idx, 'geom']
        # locations with multiple staypoints is of type "MultiPoint"
        if not ret_loc.loc[~point_idx].empty:
            ret_loc.loc[~point_idx, 'center'] = ret_loc.loc[~point_idx, 'geom'].apply(
                lambda p: Point(np.array(p)[:,0].mean(), np.array(p)[:,1].mean()))
        
        # extent is the convex hull of the geometry
        ret_loc['extent'] = None # initialize
        if not ret_loc.empty:
            ret_loc['extent'] = ret_loc['geom'].apply(lambda p: p.convex_hull)                                                                                        
            # convex_hull of one point would be a Point and two points a Linestring, 
            # we change them into Polygon by creating a buffer of epsilon around them.
            pointLine_idx = (ret_loc['extent'].geom_type == 'LineString') | (ret_loc['extent'].geom_type == 'Point')
        
            if not ret_loc.loc[pointLine_idx].empty:
                # Perform meter to decimal conversion if the distance metric is haversine
                if distance_matrix_metric == 'haversine':
                    ret_loc.loc[pointLine_idx, 'extent'] = ret_loc.loc[pointLine_idx].apply(
                        lambda p: p['extent'].buffer(meters_to_decimal_degrees(epsilon, p['center'].y)), axis=1)
                else:
                    ret_loc.loc[pointLine_idx, 'extent'] = ret_loc.loc[pointLine_idx].apply(
                        lambda p: p['extent'].buffer(epsilon), axis=1)
        
        ret_loc = ret_loc.set_geometry('center')
        ret_loc = ret_loc[['user_id', 'location_id', 'center', 'extent']]

        # index management
        ret_loc.rename(columns={'location_id':'id'}, inplace=True)
        ret_loc.set_index('id', inplace=True)
    
    # stps not linked to a location receive np.nan in 'location_id'
    ret_stps.loc[ret_stps['location_id'] == -1, 'location_id'] = np.nan
    
    ## dtype consistency
    # locs id (generated by this function) should be int64
    ret_loc.index = ret_loc.index.astype('int64')
    # location_id of spts can only be in Int64 (missing values)
    ret_stps['location_id'] = ret_stps['location_id'].astype('Int64')
    # user_id of ret_loc should be the same as ret_stps
    ret_loc['user_id'] = ret_loc['user_id'].astype(ret_stps['user_id'].dtype)
    
    return ret_stps, ret_loc


def create_activity_flag(staypoints, method='time_threshold', time_threshold=5., activity_column_name='activity'):
    """
    Add a flag whether or not a staypoint is considered an activity.

    Parameters
    ----------
    staypoints: GeoDataFrame (as trackintel staypoints)
        The original input staypoints

    method: {'time_threshold'}, default = 'time_threshold'

        - 'time_threshold' : All staypoints with a duration greater than the time_threshold are considered an activity.

    time_threshold : float, default = 5 (minutes)
        The time threshold for which a staypoint is considered an activity in minutes. Used by method 'time_threshold'
    
    activity_column_name : str , default = 'activity'
        The name of the newly created column that holds the activity flag.

    Returns
    -------
    staypoints : GeoDataFrame (as trackintel staypoints)
        Original staypoints with the additional activity column

    Examples
    --------
    >>> spts  = spts.as_staypoints.create_activity_flag(method='time_threshold', time_threshold=5)
    >>> print(spts['activity'])
    """
    if method == 'time_threshold':
        staypoints[activity_column_name] = staypoints['finished_at'] - staypoints['started_at'] \
                                           > datetime.timedelta(minutes=time_threshold)
    else:
        raise NameError("Method {} is not known".format(method))

    return staypoints
