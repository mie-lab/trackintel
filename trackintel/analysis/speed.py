import trackintel as ti
import numpy as np

from trackintel.geogr.distances import haversine_dist
from trackintel.io.file import read_positionfixes_csv


def compute_speed(positionfixes):
    """
    Compute speed per positionfix

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`speed`]``.
    """
    pfs = positionfixes.copy()
    crs_is_projected = ti.geogr.distances.check_gdf_planar(pfs)

    if crs_is_projected:
        dist_function = lambda x: x.geom.distance(x.next_geom)
    else:
        dist_function = lambda x: ti.geogr.point_distances.haversine_dist(
            x.geom.x, x.geom.y, x.next_geom.x, x.next_geom.y
        )[0]

    # get next location and time
    pfs["next_geom"] = pfs["geom"].shift(-1)
    pfs.loc[pfs.index[-1], "next_geom"] = pfs.loc[pfs.index[-1], "geom"]
    pfs["next_tracked_at"] = pfs["tracked_at"].shift(-1)

    # get distance and time difference
    pfs["dist"] = pfs.apply(dist_function, axis=1)
    pfs["time_diff"] = pfs.apply(lambda x: x.next_tracked_at - x.tracked_at, axis=1).dt.total_seconds()

    # compute speed and convert to kmh
    pfs["speed"] = 3.6 * pfs["dist"] / pfs["time_diff"]

    pfs.drop(columns=["next_geom", "next_tracked_at", "dist", "time_diff"], inplace=True)

    return pfs
