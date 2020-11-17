import multiprocessing
from functools import partial
from math import cos, pi

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from trackintel.geogr.point_distances import haversine_dist
from trackintel.geogr.trajectory_distances import dtw, frechet_dist


def calculate_distance_matrix(X, Y=None, dist_metric='haversine', n_jobs=0, **kwds):
    """
    Calculate a distance matrix based on a specific distance metric.

    If only X is given, the pair-wise distances between all elements in X are calculated. If X and Y are given, the
     distances between all combinations of X and Y are calculated. Distances between elements of X and X, and distances
     between elements of Y and Y are not calculated.

    Parameters
    ----------
    X : GeoDataFrame
         GeoPandas DataFrame in trackintel staypoints or triplegs format.
    Y : GeoDataFrame
         [optional] GeoPandas DataFrame in trackintel staypoints or triplegs format.
    dist_metric: str, {'haversine', 'euclidean', 'dtw', 'frechet'}, default 'haversine'
         The distance metric to be used for calculating the matrix. This function wraps around the
         ``pairwise_distance`` function from scikit-learn if only `X` is given and wraps around the
         `scipy.spatial.distance.cdist` function if X and Y are given. Therefore the following metrics are also
         accepted:
         via scikit-learn: `[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]`
         via scipy.spatial.distance: `[‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’,
         ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
         ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]`
         triplegs can only be used in combination with `['dtw', 'frechet']`
    n_jobs: int
         Number of cores to use: 'dtw', 'frechet' and all distance metrics from `pairwise_distance` (only available
         if only X is given) are parallelized
    kwds: optional keywords passed to the distance functions

    Returns numpy array
         returns matrix of shape (len(X), len(X)) or of shape (len(X), len(Y))
    -------
    """

    geom_type = X.geometry.iat[0].geom_type
    if Y is None:
        Y = X
    assert Y.geometry.iat[0].geom_type == Y.geometry.iat[0].geom_type, "x and y need same geometry type " \
                                                                       "(only first column checked)"

    if geom_type == 'Point':
        x1 = X.geometry.x.values
        y1 = X.geometry.y.values
        x2 = Y.geometry.x.values
        y2 = Y.geometry.y.values


        if dist_metric == 'haversine':
            # create point pairs for distance calculation
            nx = len(X)
            ny = len(Y)

            # if y != x they could have different dimensions
            if ny >= nx:
                ix_1, ix_2 = np.triu_indices(nx, k=1, m=ny)
                trilix = np.tril_indices(nx, k=-1, m=ny)
            else:
                ix_1, ix_2 = np.tril_indices(nx, k=-1, m=ny)
                trilix = np.triu_indices(nx, k=1, m=ny)

            x1 = x1[ix_1]
            y1 = y1[ix_1]
            x2 = x2[ix_2]
            y2 = y2[ix_2]

            d = haversine_dist(x1, y1, x2, y2)

            D = np.zeros((nx, ny))
            D[(ix_1, ix_2)] = d

            # mirror triangle matrix to be conform with scikit-learn format and to
            # allow for non-symmetric distances in the future
            D[trilix] = D.T[trilix]

        else:
            xy1 = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=1)

            if Y is not None:
                xy2 = np.concatenate((x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
                D = cdist(xy1, xy2, metric=dist_metric, **kwds)
            else:
                D = pairwise_distances(xy1, metric=dist_metric, n_jobs=n_jobs)

        return D

    elif geom_type == 'LineString':

        if dist_metric in ['dtw', 'frechet']:
            # these are the preparation steps for all distance functions based only on coordinates

            if dist_metric == 'dtw':
                d_fun = partial(dtw, **kwds)

            elif dist_metric == 'frechet':
                d_fun = partial(frechet_dist, **kwds)

            # get combinations of distances that have to be calculated
            nx = len(X)
            ny = len(Y)

            if ny >= nx:
                ix_1, ix_2 = np.triu_indices(nx, k=1, m=ny)
                trilix = np.tril_indices(nx, k=-1, m=ny)
            else:
                ix_1, ix_2 = np.tril_indices(nx, k=-1, m=ny)
                trilix = np.triu_indices(nx, k=1, m=ny)

            left = list(X.iloc[ix_1].geometry)
            right = list(Y.iloc[ix_2].geometry)

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
            D = np.zeros((nx, ny))
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