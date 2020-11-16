from math import radians, cos, sin, asin, sqrt, pi
import numpy as np
from scipy.spatial.distance import euclidean as euclidean_dist
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from trackintel.geogr.point_distances import haversine_dist
from trackintel.geogr.trajectory_distances import dtw, frechet_dist
from scipy.sparse import coo_matrix
from functools import partial
import multiprocessing
import geopandas as gpd
from shapely.geometry import Point

# todo: check the sklearn format for distances matrices and try to use it
# todo: think about possibilities for efficient implementation for sparse point distance matrices using knn (or geopy?)
def calculate_distance_matrix(x, y=None, dist_metric='haversine', n_jobs=0, **kwds):
    """
    Calculate a distance matrix based on a specific distance metric.
    
    Parameters
    ----------
    points : GeoDataFrame
        GeoPandas DataFrame in trackintel staypoints format.
    dist_metric : str, {'haversine', 'euclidean'}, default 'haversine'
        The distance metric to be used for calculating the matrix. This function wratps around the ``pairwise_distance``
        function from scikit-learn. Therefore the following metrics are also accepted:
        via scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
        via scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’,
         ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
          ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

    n_jobs : int, optional
        Number of jobs to be passed to the ``sklearn.metrics`` function ``pairwise_distances``.
    *args
        Description
    **kwds
        Description
    
    Returns
    -------
    array
        An array of size [n_points, n_points].
    """
    
    if y is None:
        y = x
    geom_type = x.geometry.iat[0].geom_type
    assert x.geometry.iat[0].geom_type == y.geometry.iat[0].geom_type, "x and y need same geometry type " \
                                                                       "(only first column checked)"
    # todo: check if metrics are valid for input geom type

    if geom_type == 'Point':
        x = x.geometry.x.values
        y = y.geometry.y.values

        if dist_metric == 'haversine':

            # create point pairs to calculate distance from
            n = len(x)
            ix_1, ix_2 = np.triu_indices(n, k=1)
            trilix = np.tril_indices(n, k=-1)

            x1 = x[ix_1]
            y1 = y[ix_1]
            x2 = x[ix_2]
            y2 = y[ix_2]

            d = haversine_dist(x1, y1, x2, y2)

            D = np.zeros((n, n))
            D[(ix_1, ix_2)] = d

            # mirror triangle matrix to be conform with scikit-learn format and to
            # allow for non-symmetric distances in the future
            D[trilix] = D.T[trilix]

        else:

            D = pairwise_distances(x=x, y=y, metric=dist_metric, n_jobs=n_jobs)

        return D

    elif geom_type == 'LineString':

        if dist_metric in ['dtw', 'frechet']:
            # these are the preparation steps for all distance functions based only on coordinates

            if dist_metric == 'dtw':
                d_fun = partial(dtw, **kwds)

            elif dist_metric == 'frechet':
                d_fun = partial(frechet_dist, **kwds)

            # get combinations of distances that have to be calculated
            n = len(x)
            ix_1, ix_2 = np.triu_indices(n, k=1)
            trilix = np.tril_indices(n, k=-1)

            left = list(x.iloc[ix_1].geometry)
            right = list(y.iloc[ix_2].geometry)

            # map the combinations to the distance function
            if n_jobs == -1 or n_jobs > 1:
                if n_jobs == -1:
                    n_jobs = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=n_jobs) as pool:
                    left_right = list(zip(left, right))
                    d = np.array(list(pool.starmap(d_fun, left_right)))
            else:
                d = np.array(list(map(d_fun, left, right)))

            # write results to (symmetric) distance matrix
            D = np.zeros((n, n))
            D[(ix_1, ix_2)] = d
            D[trilix] = D.T[trilix]

            return D

        else:
            raise AttributeError("Metric unknown. We only support ['dtw', 'frechet'] for LineStrings. "
                                 f"You passed {dist_metric}")
    else:
        raise AttributeError(f"We only support 'Point' and 'LineString'. Your geometry is {geom_type}")



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


# # Todo: support sparse trajectory distances with prefiltering (maybe based on the center of the trajectory?:
# # Todo: or using a top-k approach using knn
# def makeSparseDM(x, y=None, radius=None, reference='center', metric='euclidean'):
#     if y is None:
#         y = x
#
#     if radius is None:
#         raise AttributeError("threshold parameter has to be provided for prefiltering")
#
#     if reference == 'first':
#         points_x = [Point(line.coords[0]) for line in x]
#         points_y = [Point(line.coords[0]) for line in y]
#     elif reference == 'last':
#         points_x = [Point(line.coords[0]) for line in x]
#         points_y = [Point(line.coords[0]) for line in y]
#     else:
#         points_x = [Point(line.centroids) for line in x]
#         points_y = [Point(line.centroids) for line in y]
#
#     points_x = gpd.GeoDataFrame(geometry=points_x)
#     points_y = gpd.GeoDataFrame(geometry=points_y)
#
#     D = calculate_distance_matrix(X, metric=metric)
#
#     [I, J] = np.meshgrid(np.arange(N), np.arange(N))
#     I = I[D <= radius]
#     J = J[D <= radius]
#     V = D[D <= radius]
#     return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
# # https://ripser.scikit-tda.org/notebooks/Sparse%20Distance%20Matrices.html