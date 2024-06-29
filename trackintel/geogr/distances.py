import itertools
import math
import warnings
from math import cos, pi

import numpy as np
import pandas as pd
import shapely
import similaritymeasures
from sklearn.metrics import pairwise_distances

from trackintel import Triplegs


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

    if all(isinstance(x, pd.Series) for x in [lon_1, lat_1, lon_2, lat_2]):
        lon_1 = lon_1.to_numpy()
        lat_1 = lat_1.to_numpy()
        lon_2 = lon_2.to_numpy()
        lat_2 = lat_2.to_numpy()

    lon_1 = np.deg2rad(lon_1)
    lat_1 = np.deg2rad(lat_1)
    lon_2 = np.deg2rad(lon_2)
    lat_2 = np.deg2rad(lat_2)

    cos_lat1 = np.cos(lat_1)
    cos_lat2 = np.cos(lat_2)
    cos_lat_d = np.cos(lat_1 - lat_2)
    cos_lon_d = np.cos(lon_1 - lon_2)

    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def calculate_distance_matrix(X, Y=None, dist_metric="haversine", n_jobs=None, **kwds):
    """
    Compute the distance matrix from a vector array X and optional Y.

    X and Y can either be Point geometries or LineString geometries.
    If only X is given, the pair-wise distances between all elements in X are calculated.
    If X and Y are given, the distances between all combinations of X and Y are calculated.

    Parameters
    ----------
    X : GeoDataFrame

    Y : GeoDataFrame, optional
        Should have same geometry type as X

    dist_metric: {{'haversine', 'euclidean', 'dtw', 'frechet'}}, optional
        The distance metric to be used for calculating the matrix. By default 'haversine.

        For Point geometries we provide the 'haversine' metric.
        This function then wraps `sklearn.metrics.pairwise_distances`.
        Therefore the following metrics are also accepted:

        - via ``scikit-learn``: `['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']`

        - via ``scipy.spatial.distance``: `['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
        'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule']`

        For LineStrings, we provide the metrics {'dtw', 'frechet'} via the implementation from similaritymeasures.

    n_jobs: int, optional
        The number of jobs to use for the computation. Ignored for LineStrings.
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        See `sklearn.metrics.pairwise_distances` for more informations.

    **kwds:
        Optional keywords passed to the distance functions.

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
    if Y is not None and Y.geometry.iloc[0].geom_type != geom_type:
        raise ValueError("X and Y need to have same geometry type.")
    if geom_type not in ["Point", "LineString"]:
        raise ValueError(f"We only support 'Point' and 'LineString'. Your geometry is {geom_type}")

    if geom_type == "Point":
        if dist_metric == "haversine":
            # curry our haversine distance
            def haversine_curry(a, b, **_):
                return point_haversine_dist(*a, *b, float_flag=True)

            dist_metric = haversine_curry
        X = shapely.get_coordinates(X.geometry)
        Y = shapely.get_coordinates(Y.geometry) if Y is not None else X
        return pairwise_distances(X, Y, metric=dist_metric, n_jobs=n_jobs, **kwds)

    # geom_type == "LineString"
    # for LineStrings we cannot use pairwise_distance because it enforces float in its array
    if dist_metric == "dtw":

        def dtw(a, b, **kwds):
            return similaritymeasures.dtw(a, b, **kwds)[0]

        dist_metric = dtw
    elif dist_metric == "frechet":
        dist_metric = similaritymeasures.frechet_dist
    else:
        raise ValueError(f"Metric '{dist_metric}' unknown. We only support ['dtw', 'frechet'] for LineStrings")
    X = X.geometry.values
    Y = Y.geometry.values if Y is not None else X

    # the following code is adapted from scikit-learn pairwise_distance
    # https://github.com/scikit-learn/scikit-learn/blob/3f89022fa04d293152f1d32fbc2a5bdaaf2df364/sklearn/metrics/pairwise.py#L1784
    out = np.zeros((len(X), len(Y)), dtype="float")
    if X is Y:
        # Only calculate metric for upper triangle
        iterator = itertools.combinations(range(len(X)), 2)
        for i, j in iterator:
            out[i, j] = dist_metric(X[i].coords, Y[j].coords, **kwds)
        # Make symmetric
        out = out + out.T
    else:
        # Calculate all cells
        iterator = itertools.product(range(len(X)), range(len(Y)))
        for i, j in iterator:
            out[i, j] = dist_metric(X[i].coords, Y[j].coords, **kwds)
    return out


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
            raise ValueError('Method "pfs_mean_speed" requires positionfixes as input.')
        if "tripleg_id" not in positionfixes:
            raise AttributeError('Positionfixes must include column "tripleg_id".')
        # group positionfixes by triplegs and compute average speed for each collection of positionfixes
        grouped_pfs = positionfixes.groupby("tripleg_id").apply(_single_tripleg_mean_speed, include_groups=False)
        # add the speed values to the triplegs column
        tpls = pd.merge(triplegs, grouped_pfs.rename("speed"), how="left", left_index=True, right_index=True)
        tpls.index = tpls.index.astype("int64")
        return tpls

    else:
        raise ValueError(f"Method {method} not known for speed computation.")


def _single_tripleg_mean_speed(positionfixes):
    pfs_sorted = positionfixes.sort_values(by="tracked_at")
    pfs_speed = get_speed_positionfixes(pfs_sorted)
    return np.mean(pfs_speed["speed"].values[1:])
