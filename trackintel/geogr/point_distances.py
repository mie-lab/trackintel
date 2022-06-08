import numpy as np
import math


def haversine_dist(lon_1, lat_1, lon_2, lat_2, r=6371000, float_flag=False):
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
    >>> haversine_dist(8.5, 47.3, 8.7, 47.2)
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
