import multiprocessing
import warnings
from functools import partial
from math import cos, pi

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import similaritymeasures

from trackintel.geogr.point_distances import haversine_dist


def calculate_distance_matrix(X, Y=None, dist_metric="haversine", n_jobs=0, **kwds):
    """
    Calculate a distance matrix based on a specific distance metric.

    If only X is given, the pair-wise distances between all elements in X are calculated. If X and Y are given, the
    distances between all combinations of X and Y are calculated. Distances between elements of X and X, and distances
    between elements of Y and Y are not calculated.

    Parameters
    ----------
    X : GeoDataFrame (as trackintel staypoints or triplegs)

    Y : GeoDataFrame (as trackintel staypoints or triplegs), optional

    dist_metric: {'haversine', 'euclidean', 'dtw', 'frechet'}
        The distance metric to be used for calculating the matrix.

        For staypoints, common choice is 'haversine' or 'euclidean'. This function wraps around
        the ``pairwise_distance`` function from scikit-learn if only `X` is given and wraps around the
        ``scipy.spatial.distance.cdist`` function if X and Y are given.
        Therefore the following metrics are also accepted:

        via ``scikit-learn``: `[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]`

        via ``scipy.spatial.distance``: `[‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’,
        ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
        ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]`

        For triplegs, common choice is 'dtw' or 'frechet'. This function uses the implementation
        from similaritymeasures.

    n_jobs: int
        Number of cores to use: 'dtw', 'frechet' and all distance metrics from `pairwise_distance` (only available
        if only X is given) are parallelized.

    **kwds:
        optional keywords passed to the distance functions.

    Returns
    -------
    D: np.array
        matrix of shape (len(X), len(X)) or of shape (len(X), len(Y)) if Y is provided.

    """
    geom_type = X.geometry.iat[0].geom_type
    if Y is None:
        Y = X
    assert Y.geometry.iat[0].geom_type == Y.geometry.iat[0].geom_type, (
        "x and y need same geometry type " "(only first column checked)"
    )

    if geom_type == "Point":
        x1 = X.geometry.x.values
        y1 = X.geometry.y.values
        x2 = Y.geometry.x.values
        y2 = Y.geometry.y.values

        if dist_metric == "haversine":
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

    elif geom_type == "LineString":

        if dist_metric in ["dtw", "frechet"]:
            # these are the preparation steps for all distance functions based only on coordinates

            if dist_metric == "dtw":
                d_fun = partial(similaritymeasures.dtw, **kwds)
            else:
                d_fun = partial(similaritymeasures.frechet_dist, **kwds)

            # get combinations of distances that have to be calculated
            nx = len(X)
            ny = len(Y)

            if ny >= nx:
                ix_1, ix_2 = np.triu_indices(nx, k=1, m=ny)
                trilix = np.tril_indices(nx, k=-1, m=ny)
            else:
                ix_1, ix_2 = np.tril_indices(nx, k=-1, m=ny)
                trilix = np.triu_indices(nx, k=1, m=ny)

            # get the coordinates as list of each LineString
            left = list(X.iloc[ix_1].geometry.apply(lambda x: x.coords))
            right = list(Y.iloc[ix_2].geometry.apply(lambda x: x.coords))

            # map the combinations to the distance function
            if n_jobs == -1 or n_jobs > 1:
                if n_jobs == -1:
                    n_jobs = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=n_jobs) as pool:
                    left_right = list(zip(left, right))
                    res = list(pool.starmap(d_fun, left_right))
            else:
                res = list(map(d_fun, left, right))

            if dist_metric == "dtw":
                # the first return is the dtw distance, see docs of similaritymeasures.dtw
                d = [dist[0] for dist in res]
            else:
                d = res

            # write results to (symmetric) distance matrix
            D = np.zeros((nx, ny))
            D[(ix_1, ix_2)] = d
            D[trilix] = D.T[trilix]
            return D

        else:
            raise AttributeError(
                "Metric unknown. We only support ['dtw', 'frechet'] for LineStrings. " f"You passed {dist_metric}"
            )
    else:
        raise AttributeError(f"We only support 'Point' and 'LineString'. Your geometry is {geom_type}")


def meters_to_decimal_degrees(meters, latitude):
    """
    Convert meters to decimal degrees (approximately).

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


def check_gdf_crs(gdf, transform=False):
    """
    Check if GeoDataFrame has CRS or is already in WGS84.

    Additionally transform a GeoDataFrame into WGS84.

    Parameters
    ----------
    gdf : GeoDataFrame
        input GeoDataFrame for checking or transform

    transform : bool, default False
        whether to transform gdf into WGS84.

    Returns
    -------
    if_planer : bool
        True if the returned gdf has planar crs.

    gdf : GeoDataFrame
        if transform is True, return the re-projected gdf.


    Examples
    --------
    >>> from trackintel.geogr.distances import check_gdf_crs
    >>> check_gdf_crs(triplegs, transform=False)
    """
    if_planer = False
    if gdf.crs is None:
        # projection is not defined
        if_planer = False
        if transform:
            gdf.crs = "EPSG:4326"
        else:
            warnings.warn("Your data is not projected.")

    elif gdf.crs == "EPSG:4326":
        # if projection is defined as WGS84
        if_planer = False

    else:
        # if projection is defined but not as WGS84
        if_planer = True
        if transform:
            if_planer = False
            gdf = gdf.to_crs("EPSG:4326")

    if transform:
        return if_planer, gdf
    else:
        return if_planer


def calculate_haversine_length(gdf):
    """
    Calculate the length of linestrings using the haversine distance.

    Parameters
    ----------
    gdf : GeoDataFrame with linestring geometry
        The coordinates are expected to be in WGS84

    Returns
    -------
    length: Pandas Series
        The length of each linestring in meters

    Examples
    --------
    >>> from trackintel.geogr.distances import calculate_haversine_length
    >>> triplegs['length'] = calculate_haversine_length(triplegs)
    """
    assert all(gdf.geom_type == "LineString")

    length = gdf.geometry.apply(_calculate_haversine_length_single)
    return length


def _calculate_haversine_length_single(linestring):
    """
    calculate the length of a single linestring using the haversine distance.

    Parameters
    ----------
    linestring : 'shapely.geometry.linestring.LineString'
        Coordinates of the linestring are expected to be in WGS84

    Returns
    -------
    int
        length of the linestring in meter

    Examples
    --------
    >>> from shapely.geometry import LineString
    >>> from trackintel.geogr.distances import _calculate_haversine_length_single
    >>> ls = LineString([(13.476808430, 48.573711823), (11.5675446, 48.1485459), (8.5067847, 47.4084269)])
    >>> _calculate_haversine_length_single(ls)
    """
    coords_df = pd.DataFrame(linestring.xy, index=["x_0", "y_0"]).transpose()
    coords_df["x_1"] = coords_df["x_0"].shift(-1)
    coords_df["y_1"] = coords_df["y_0"].shift(-1)
    coords_df.dropna(axis=0, inplace=True)

    distances = haversine_dist(coords_df.x_0, coords_df.y_0, coords_df.x_1, coords_df.y_1)
    return np.sum(distances)
