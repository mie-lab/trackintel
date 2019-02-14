from math import radians, cos, sin, asin, sqrt, pi


def haversine_dist(lon_1, lat_1, lon_2, lat_2):
    """Computes the great circle or haversine distance between two coordinates in WGS84.

    Parameters
    ----------
    lon_1 : float
        The longitude of the first point.
    
    lat_1 : float
        The latitude of the first point.
        
    lon_2 : float
        The longitude of the second point.
    
    lat_2 : float
        The latitude of the second point.

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
    """
    lon_1, lat_1, lon_2, lat_2 = map(radians, [lon_1, lat_1, lon_2, lat_2])
    
    dlon = lon_2 - lon_1 
    dlat = lat_2 - lat_1 

    a = sin(dlat / 2)**2 + cos(lat_1) * cos(lat_2) * sin(dlon / 2)**2
    # The average Earth radius is 6'371'000 m.
    m = 2 * 6371000 * asin(sqrt(a)) 
    return m


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