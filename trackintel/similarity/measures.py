"""
Algorithms e_dtw and e_edr adapted from https://github.com/bguillouet/traj-dist

"""


import numpy as np
import geopandas as gpd


def e_dtw(t0, t1,**kwargs):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.
    Distance Function: Euclidian Distance
    
    Properties:
        - DTW is affected by outliers in the data
        - DTW can deal with different sampling rates, but therefore not distinguish different movement speeds
        - DTW distance is normalized by the (bigger) number of trajectory points
    

    Parameters
    ----------
    t0 : GeoDataFrame with n0 Points
    t1 : GeoDataFrame with n1 Points

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = t0.shape[0]
    n1 = t1.shape[0]
    c = np.zeros((n0 + 1, n1 + 1))
    c[1:, 0] = float('inf')
    c[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            c[i, j] = t0.iloc[i - 1]['geom'].distance(t1.iloc[j - 1]['geom']) + min(c[i, j - 1], c[i - 1, j - 1], c[i - 1, j])
    dtw = float(c[n0, n1])/max([n0,n1])
    return dtw



def e_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1. 
    Distance Function: Euclidian Distance

    Parameters
    ----------
    t0 : GeoDataFrame with n0 Points
    t1 : GeoDataFrame with n1 Points
    eps : float
        threshold for two points to be considered as equal (Epsilon parameter)

    Returns
    -------
    edr : float
           The Edit Distance on real Sequence between trajectory t0 and t1
    """
    n0 = t0.shape[0]
    n1 = t1.shape[0]
    
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if t0.iloc[i-1].geom.distance(t1.iloc[j-1].geom) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr


def start_end_dist(data, dist_trsh, time_trsh, field='tripleg_id', w=[0.35, 0.35, 0.1,  0.2], **kwargs):
    """
    Method that calculates the start_end_distance of either two points or a trajectory to all trajectories of a data set.
    
    Parameters
    ----------
        data            GeoDataFrame of positionfixes with a field (e.g. tripleg_id) to distinguish the trajectories
        dist_trsh       Threshold for points to be considered as near to the start or end. Unit dependent of projection of data.
        time_trsh       Threshold for time differences in seconds.
        
        field           The field of data to distinguish the trajectories
        w               Weighting function [start distance, end distance, start time difference, end time difference]
        
    OPTIONS
    -------
        id_to_compare   The id (in data.field) of the trajectory to compare
        
        OR
        
        start           The origin of a trajectory to compare, that is not in data. [point]
        end             The destination "           "               "               [point]
        start_time      The timestamp of start                                      
        end_time        The timestamp of end
        
    Returns
    -------
        traj_dist       Array of length max(field.values)+1 with the trajectory distances weighted by w. 
                        Be aware that also non existing ids have distance inf. If possible the ids of field should be continuous.
        
    """
    
    try:
        assert data.as_positionfixes
        assert field in data.columns
       
    except:
        raise Exception('data must be positionfixes with ' + field + ' in columns')
             
    try:
         assert data['tracked_at'].dtype == np.dtype('<M8[ns]') 
    except:
        raise Exception('data[tracked_at] must be dtype Timestamp, you may need to call data[tracked_at].dt.tz_localize(None)') #check other data types
        
             
    if 'id_to_compare' in kwargs: #case the tripleg to compare is in the data set
        id_to_compare = kwargs.get('id_to_compare')
        tpl = data[data[field]==id_to_compare]
        start, start_time, end, end_time = ses_extract(tpl)
        
    elif ('start' in kwargs and 'end' in kwargs and 'start_time' in kwargs and 'end_time' in kwargs):  #case the start, end, etc. are set manually
        start = kwargs.get('start')
        start_time = kwargs.get('start_time')
        end =   kwargs.get('end')
        end_time = kwargs.get('end_time')
        
    else:
        raise Exception('id_to_compare or start, end, start_time and end_time must be specified')
        
        
    start_buffer = start.buffer(dist_trsh)
    end_buffer = end.buffer(dist_trsh)
    start_neighbours = data[data.intersects(start_buffer)]
    end_neighbours = data[data.intersects(end_buffer)]
    traj_dist = np.ones((int(max(data[field].unique())+1)))*np.inf
    
    it = start_neighbours[field].unique().tolist()
    if not it:
        try:
           it.remove(-1) #catch case where all points belong to a tripleg
        except KeyError:
            pass
        it.remove(id_to_compare)
        
            
    
    for i in it: #iterate over all triplegs within the start_buffer
        if i in end_neighbours[field].values: #only continue with triplegs that have also a point in the end_buffer
            neighbour_points_start_this_tpl = start_neighbours[start_neighbours[field]==i] #extract the points in the start_buffer
            neighbour_points_end_this_tpl = end_neighbours[end_neighbours[field]==i]
            start_distances = neighbour_points_start_this_tpl.geom.distance(start)
            end_distances = neighbour_points_end_this_tpl.geom.distance(end)
            start_min_dist_id = start_distances.argmin()
            end_min_dist_id = end_distances.argmin()
            
            start_diff = start_distances.iloc[start_min_dist_id]
            end_diff = end_distances.iloc[end_min_dist_id]
            
            
            start_time_diff = (start_time - neighbour_points_start_this_tpl.iloc[start_min_dist_id]['tracked_at']).seconds
            end_time_diff = (end_time - neighbour_points_end_this_tpl.iloc[end_min_dist_id]['tracked_at']).seconds
            
            rel_start_diff = start_diff/dist_trsh
            rel_end_diff = end_diff/dist_trsh
            rel_start_time_diff = start_time_diff/time_trsh
            rel_end_time_diff = end_time_diff/time_trsh
            
            d = w[0]*rel_start_diff+w[1]*rel_end_diff+w[2]*rel_start_time_diff+w[3]*rel_end_time_diff
            traj_dist[int(i)] = d    
    
    try:
        traj_dist[int(id_to_compare)]=0 #depending on the method chosen by the user
    except UnboundLocalError:
        pass
    return traj_dist



def ses_extract(tpl):
    start = tpl.iloc[0].geom
    start_time = tpl.iloc[0]['tracked_at']
    end = tpl.iloc[-1].geom
    end_time = tpl.iloc[-1]['tracked_at']
    return start, start_time, end, end_time

#def ses_calc(start, start_t, end, end_t, dist_trsh, time_trsh, w):
    


















