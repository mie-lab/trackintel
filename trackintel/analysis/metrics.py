import numpy as np
from tqdm import tqdm

from trackintel.geogr import point_haversine_dist, check_gdf_planar


def radius_gyration(sp, method="count", print_progress=False):
    """
    Radius of gyration for individual users.

    Parameters
    ----------
    sp : Staypoints

    method: string, {"count", "duration"}
        Weighting for center of mass and average distance calculation.

        - `count`: assigns each Point the same weight of 1.
        - `duration`: assigns each Point a weight based on duration.
          Additionally needs Timedelta column "duration" in sp.

    print_progress: bool, default False
        Show per-user progress if set to True.

    Returns
    -------
    Series
        Radius of gyration for individual users.

    References
    ----------
    [1] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008).
    Understanding individual human mobility patterns. Nature, 453(7196), 779-782.

    """
    if method not in ["count", "duration"]:
        raise ValueError(f'Method unknown. Should be on of {{"count", "duration"}}. You passed "{method}"')

    if print_progress:
        tqdm.pandas(desc="User radius of gyration calculation")
        s = sp.groupby("user_id").progress_apply(_radius_gyration_user, method=method)
    else:
        s = sp.groupby("user_id").apply(_radius_gyration_user, method=method)

    s = s.rename("radius_gyration")
    return s


def _radius_gyration_user(sp, method):
    """
    User level radius of gyration calculation, see radius_gyration() for more details.

    Parameters
    ----------
    sp : Staypoints
    method: {"count", "duration"}

    Returns
    -------
    float
        The radius of gyration of the user
    """
    x = sp.geometry.x
    y = sp.geometry.y

    if method == "duration":
        w = sp["duration"].dt.total_seconds()
    else:  # method == count (check is done in upper level)
        w = np.ones_like(x)

    x_center = np.average(x, weights=w)
    y_center = np.average(y, weights=w)
    if check_gdf_planar(sp):
        sq_dist = (x - x_center) ** 2 + (y - y_center) ** 2
    else:
        sq_dist = point_haversine_dist(x, y, x_center, y_center) ** 2
    square_rg = np.average(sq_dist, weights=w)
    return np.sqrt(square_rg)
