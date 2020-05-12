"""
Algorithm adapted from https://github.com/bguillouet/traj-dist

"""


import numpy as np
import geopandas as gpd


def e_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1. The timestamps
    in the GeoDataFrames are not (yet) considered.

    Parameters
    ----------
    param t0 : GeoDataFrame with n0 Points
    param t1 : GeoDataFrame with n1 Points

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
    dtw = c[n0, n1]
    return dtw



def e_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Edit Distance on real Sequence between trajectory t0 and t1
    """
    n0 = t0.shape[0]
    n1 = t1.shape[0]
    # An (m+1) times (n+1) matrix
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

#
#def start_end_sim(data, dist_trsh, time_trsh, field='tripleg_id', w=[0.4, 0.4, 0.2], matrix=False **kwargs):
#    try:
#        assert data.as_positionfixes
#        assert field in data.columns
#    except:
#        raise Exception('data must be positionfixes with ' + field + ' in columns')
#    
#    if matrix:
#        it = data[field].unique()
#        for tpl in it:
#             start, start_time, end, end_time = ses_extract(tpl)
#             
#             
#    if 'id_to_compare' in kwargs:
#        id_to_compare = kwargs.get('id_to_compare')
#        tpl = data[data[field]==id_to_compare]
#        start, start_time, end, end_time = ses_extract(tpl)
#        
#    elif all('start' and 'end' and 'end_time' in kwargs): #check syntax:
#        start = kwargs.get('start')
#        start_time = kwargs.get('start_time')
#        end =   kwargs.get('end')
#        end_time = kwargs.get('end_time')
#        
#        
#    start_buffer = start.buffer(dist_trsh)
#    end_buffer = end.buffer(dist_trsh)
#    start_neighbours = data[data.intersects(start_buffer)]
#    end_neighbours = data[data.intersects(end_buffer)]
#    traj_ids = []
#    
#    for i in list(start_neighbours[field].unique()).remove(-1):
#        if i in end_neighbours[field]:
#            neighbour_points_start_this_tpl = start_neighbours[start_neighbours[field]==i]
#            neighbour_points_end_this_tpl = end_neighbours[end_neighbours[field]==i]
#            start_dist = min(neighbour_points_start_this_tpl.geom.distance(start))
#            end_dist = min(neighbour_points_end_this_tpl.geom.distance(end))
#            
#                
#                time_diff = 
#                d = w[0]*start_diff+w[1]*end_diff+w[2]*
#                traj_ids.append(start_neighbours.iloc[i][field])
#     traj_ids = list(set(traj_ids))       
#    
#    
#    return traj_ids
#
#
#
#def ses_extract(tpl):
#    start = tpl.iloc[0].geom
#    start_time = tpl.iloc[0]['tracked_at']
#    end = tpl.iloc[-1].geom
#    end_time = tpl.iloc[-1]['tracked_at']
#    return start, start_time, end, end_time
#
#def ses_calc(start, start_t, end, end_t, dist_trsh, time_trsh, w):
#    
#
#
#
#
#
#
#
#










