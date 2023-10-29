import math
import multiprocessing
import warnings
from functools import partial
from math import cos, pi

import numpy as np
import pandas as pd
import shapely
import similaritymeasures
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

from trackintel import Positionfixes, Triplegs


def point_haversine_dist(lon_1, lat_1, lon_2, lat_2, r=6371000, float_flag=False):
    """
    Compute the great circle or haversine distance between two coordinates in WGS84.

    Serialized version of the haversine distance.

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

    float_flag : bool, default False
        Optimization flag. Set to True if you are sure that you are only using floats as args.

    Returns
    -------
    float or numpy.array
        An approximation of the distance between two points in WGS84 given in meters.

    Examples
    --------
    >>> point_haversine_dist(8.5, 47.3, 8.7, 47.2)
    18749.056277719905

    References
    ----------
    https://en.wikipedia.org/wiki/Haversine_formula
    https://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
    """
    if float_flag:
        lon_1 = math.radians(lon_1)
        lat_1 = math.radians(lat_1)
        lon_2 = math.radians(lon_2)
        lat_2 = math.radians(lat_2)

        cos_lat2 = math.cos(lat_2)
        cos_lat1 = math.cos(lat_1)
        cos_lat_d = math.cos(lat_1 - lat_2)
        cos_lon_d = math.cos(lon_1 - lon_2)

        return r * math.acos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    lon_1 = np.deg2rad(lon_1).ravel()
    lat_1 = np.deg2rad(lat_1).ravel()
    lon_2 = np.deg2rad(lon_2).ravel()
    lat_2 = np.deg2rad(lat_2).ravel()

    cos_lat1 = np.cos(lat_1)
    cos_lat2 = np.cos(lat_2)
    cos_lat_d = np.cos(lat_1 - lat_2)
    cos_lon_d = np.cos(lon_1 - lon_2)

    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def calculate_distance_matrix(X, Y=None, dist_metric="haversine", n_jobs=0, **kwds):
    """
    Calculate a distance matrix based on a specific distance metric.

    If only X is given, the pair-wise distances between all elements in X are calculated.
    If X and Y are given, the distances between all combinations of X and Y are calculated.
    Distances between elements of X and X, and distances between elements of Y and Y are not calculated.

    Parameters
    ----------
    X : Trackintel class

    Y : Trackintel class, optional
        Should be of the same type as X

    dist_metric: {{'haversine', 'euclidean', 'dtw', 'frechet'}}, optional
        The distance metric to be used for calculating the matrix. By default 'haversine.

        For staypoints or positionfixes, a common choice is 'haversine' or 'euclidean'. This function wraps around
        the ``pairwise_distance`` function from scikit-learn if only `X` is given and wraps around the
        ``scipy.spatial.distance.cdist`` function if X and Y are given.
        Therefore the following metrics are also accepted:

        via ``scikit-learn``: `['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']`

        via ``scipy.spatial.distance``: `['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
        'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule']`

        For triplegs, common choice is 'dtw' or 'frechet'. This function uses the implementation
        from similaritymeasures.

    n_jobs: int, optional
        Number of cores to use: 'dtw', 'frechet' and all distance metrics from `pairwise_distance` (only available
        if only X is given) are parallelized. By default 1.

    **kwds:
        optional keywords passed to the distance functions.

    Returns
    -------
    D: np.array
        matrix of shape (len(X), len(X)) or of shape (len(X), len(Y)) if Y is provided.

    Examples
    --------
    >>> calculate_distance_matrix(staypoints, dist_metric="haversine")
    >>> calculate_distance_matrix(triplegs_1, triplegs_2, dist_metric="dtw")
    >>> pfs.calculate_distance_matrix(dist_metric="haversine")
    """
    geom_type = X.geometry.iat[0].geom_type
    if Y is None:
        Y = X
    assert (
        Y.geometry.iat[0].geom_type == Y.geometry.iat[0].geom_type
    ), "x and y need same geometry type (only first column checked)"

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

            d = point_haversine_dist(x1, y1, x2, y2)

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

    Examples
    --------
    >>> meters_to_decimal_degrees(500.0, 47.410)
    """
    return meters / (111.32 * 1000.0 * cos(latitude * (pi / 180.0)))


def check_gdf_planar(gdf, transform=False):
    """
    Check if a GeoDataFrame has a planar or projected coordinate system.

    Optionally transform a GeoDataFrame into WGS84.

    Parameters
    ----------
    gdf : GeoDataFrame
        input GeoDataFrame for checking or transform

    transform : bool, default False
        whether to transform gdf into WGS84.

    Returns
    -------
    is_planer : bool
        True if the returned gdf has planar crs.

    gdf : GeoDataFrame
        if transform is True, return the re-projected gdf.

    Examples
    --------
    >>> from trackintel.geogr import check_gdf_planar
    >>> check_gdf_planar(triplegs, transform=False)
    """
    wgs84 = "EPSG:4326"
    if gdf.crs != wgs84:
        if transform:
            gdf = gdf.set_crs(wgs84) if gdf.crs is None else gdf.to_crs(wgs84)

    if gdf.crs is None:
        warnings.warn("The CRS of your data is not defined.")

    if transform:
        return False, gdf
    return not (gdf.crs is None or gdf.crs.is_geographic)


def calculate_haversine_length(gdf):
    """
    Calculate the length of linestrings using the haversine distance.

    Parameters
    ----------
    gdf : GeoDataFrame with linestring geometry
        The coordinates are expected to be in WGS84

    Returns
    -------
    length: np.array
        The length of each linestring in meters

    Examples
    --------
    >>> from trackintel.geogr import calculate_haversine_length
    >>> triplegs['length'] = calculate_haversine_length(triplegs)
    """
    geom = gdf.geometry
    assert np.any(shapely.get_type_id(geom) == 1)  # 1 is LineStrings
    geom, index = shapely.get_coordinates(geom, return_index=True)
    no_mix = index[:-1] == index[1:]  # mask where LineStrings are not overlapping
    dist = point_haversine_dist(geom[:-1, 0], geom[:-1, 1], geom[1:, 0], geom[1:, 1])
    return np.bincount((index[:-1])[no_mix], weights=dist[no_mix])


def get_speed_positionfixes(positionfixes):
    """
    Compute speed per positionfix (in m/s)

    Parameters
    ----------
    positionfixes : Positionfixes

    Returns
    -------
    pfs: Positionfixes
        Copy of the original positionfixes with a new column ``[`speed`]``. The speed is given in m/s

    Notes
    -----
    The speed at one positionfix is computed from the distance and time since the previous positionfix.
    For the first positionfix, the speed is set to the same value as for the second one.
    """
    pfs = positionfixes.copy()
    is_planar_crs = check_gdf_planar(pfs)

    g = pfs.geometry
    # get distance and time difference
    if is_planar_crs:
        dist = g.distance(g.shift(1)).to_numpy()
    else:
        x = g.x.to_numpy()
        y = g.y.to_numpy()
        dist = np.zeros(len(pfs), dtype=np.float64)
        dist[1:] = point_haversine_dist(x[:-1], y[:-1], x[1:], y[1:])

    time_delta = (pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds().to_numpy()
    # compute speed (in m/s)
    speed = dist / time_delta
    speed[0] = speed[1]  # The first point speed is imputed
    pfs["speed"] = speed
    return pfs


def get_speed_triplegs(triplegs, positionfixes=None, method="tpls_speed"):
    """
    Compute the average speed per positionfix for each tripleg (in m/s)

    Parameters
    ----------
    triplegs: Triplegs

    positionfixes: Positionfixes, optional
        Only required if the method is 'pfs_mean_speed'.
        In addition to the standard columns positionfixes must include the column ``[`tripleg_id`]``.

    method: {'tpls_speed', 'pfs_mean_speed'}, optional
        Method how of speed calculation, default is "tpls_speed"
        The 'tpls_speed' method divides the tripleg distance by its duration,
        the 'pfs_mean_speed' method calculates the speed via the mean speed of the positionfixes of a tripleg.

    Returns
    -------
    tpls: Triplegs
        The original triplegs with a new column ``[`speed`]``. The speed is given in m/s.
    """
    Triplegs.validate(triplegs)
    # Simple method: Divide overall tripleg distance by overall duration
    if method == "tpls_speed":
        if check_gdf_planar(triplegs):
            distance = triplegs.length
        else:
            distance = calculate_haversine_length(triplegs)
        duration = (triplegs["finished_at"] - triplegs["started_at"]).dt.total_seconds()
        # The unit of the speed is m/s
        tpls = triplegs.copy()
        tpls["speed"] = distance / duration
        return tpls

    # Pfs-based method: compute speed per positionfix and average then
    elif method == "pfs_mean_speed":
        if positionfixes is None:
            raise AttributeError('Method "pfs_mean_speed" requires positionfixes as input.')
        if "tripleg_id" not in positionfixes:
            raise AttributeError('Positionfixes must include column "tripleg_id".')
        # group positionfixes by triplegs and compute average speed for each collection of positionfixes
        grouped_pfs = positionfixes.groupby("tripleg_id").apply(_single_tripleg_mean_speed)
        # add the speed values to the triplegs column
        tpls = pd.merge(triplegs, grouped_pfs.rename("speed"), how="left", left_index=True, right_index=True)
        tpls.index = tpls.index.astype("int64")
        return tpls

    else:
        raise AttributeError(f"Method {method} not known for speed computation.")


def _single_tripleg_mean_speed(positionfixes):
    pfs_sorted = positionfixes.sort_values(by="tracked_at")
    pfs_speed = get_speed_positionfixes(pfs_sorted)
    return np.mean(pfs_speed["speed"].values[1:])
