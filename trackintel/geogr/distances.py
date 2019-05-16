from math import radians, cos, sin, asin, sqrt, pi
import numpy as np
from scipy.spatial.distance import euclidean as euclidean_dist
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
# todo: calculate distance matrix
#def distance_matrix():
#   pass
# for all that is euclidean (or minkowski) we can use scipy.spatial.distance_matrix
#
# There is a sklearn function that supports many different metrics:
# sklearn.metrics.pairwise_distances(X, Y=None, metric=’euclidean’, n_jobs=None, **kwds)
#

# todo: check the sklearn format for distances matrices and try to use it
def calculate_distance_matrix(points, dist_metric='haversine', n_jobs=0, *args, **kwds):
    """
    Calculate a distance matrix based on a specific distance metric.
    
    Parameters
    ----------
    points : geopandas dataframe
        geopandas dataframe in trackintel staypoints format
    dist_metric : str, optional
        ('haversine', 'euclidean')
    *args
        Description
    **kwds
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    x = points.geometry.x.values
    y = points.geometry.x.values
    xy = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
    
    if dist_metric == 'euclidean':
        D = pairwise_distances(xy)
        
#    # super slow!
#    elif dist_metric == 'haversine':
#        D = cdist(xy, xy, metric=haversine_dist_cdist)
        
    elif dist_metric == 'haversine':
#        x = points.geometry.x
#        y = points.geometry.x
    
        # create point pairs to calculate distance from
        n = len(x)
        x1 = []
        y1 = []
        x2 = []
        y2 = []
    
        for i in range(n):
            for j in np.arange(i+1,n):
                x1.append(x[i])
                y1.append(y[i])
                x2.append(x[j])
                y2.append(y[j])
        
        d = haversine_dist(x1, y1, x2, y2)
        
        # rebuild matrix from vector
        D = np.zeros((n,n))
        k = 0
        for i in range(n):
            for j in np.arange(i+1,n):
                D[i,j] = d[k]
                k = k+1
                
        # mirror triangle matrix to be conform with scikit-learn format and to 
        # allow for non-symmetric distances in the future
        
        for i in range(D.shape[0]):
            for j in range(i, D.shape[1]):
                D[j][i] = D[i][j]
        
        
    else:
        D = pairwise_distances(xy, metric=dist_metric, n_jobs=n_jobs)


    return D
    
def haversine_dist_cdist(XA, XB):
    """Applies the haversine_dist function for the scipy cdist function.
    
    Parameters
    ----------
    XA : numpy array
        2d numpy array with [lon1,lat1]
    XB : TYPE
        2d numpy array with [lon2,lat2]
    
    Returns
    -------
    float
        haversine distance between two points
    """
    
    
    return haversine_dist(XA[0], XA[1], XB[0], XB[1])

def haversine_dist(lon_1, lat_1, lon_2, lat_2, r=6371000):
    """Computes the great circle or haversine distance between two coordinates in WGS84.

    # todo: test different input formats, especially different vector
    shapes
    # define output format. 

    Parameters
    ----------
    lon_1 : float or numpy.array of shape (-1,)
        The longitude of the first point.
    
    lat_1 : float or numpy.array of shape (-1,)
        The latitude of the first point.
        
    lon_2 : float or numpy.array of shape (-1,)
        The longitude of the second point.
    
    lat_2 : float or numpy.array of shape (-1,)
        The latitude of the second point.

    r     : float
        Radius of the reference sphere for the calculation. 
        The average Earth radius is 6'371'000 m. 

    Returns
    -------
    float
        An approximation of the distance between two points in WGS84 given in meters.

    Examples
    --------
    >>> haversine_dist(8.5, 47.3, 8.7, 47.2)
    18749.056277719905

    References
    ----------
    https://en.wikipedia.org/wiki/Haversine_formula
    https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
    """ 
    
    lon_1 = np.asarray(lon_1).ravel() * np.pi / 180
    lat_1 = np.asarray(lat_1).ravel() * np.pi / 180
    lon_2 = np.asarray(lon_2).ravel() * np.pi / 180
    lat_2 = np.asarray(lat_2).ravel() * np.pi / 180
    
    cos_lat1 = np.cos(lat_1)
    cos_lat2 = np.cos(lat_2)
    cos_lat_d = np.cos(lat_1 - lat_2)
    cos_lon_d = np.cos(lon_1 - lon_2)

    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def meters_to_decimal_degrees(meters, latitude):
    """Converts meters to decimal degrees (approximately).

    Parameters
    ----------
    meters : float
        The meters to convert to degrees.

    latitude : float
        As the conversion is dependent (approximatively) on the latitude where 
        the conversion happens, this needs to be specified. Use 0 for the equator.

    Returns
    -------
    float
        An approximation of a distance (given in meters) in degrees.
    """
    return meters / (111.32 * 1000.0 * cos(latitude * (pi / 180.0)))