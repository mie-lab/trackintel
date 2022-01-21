from functools import partial, update_wrapper
import trackintel as ti
import numpy as np
import pandas as pd
from trackintel.geogr.distances import calculate_haversine_length


def get_speed_positionfixes(positionfixes):
    """
    Compute speed per positionfix (in m/s)

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`speed`]``. The speed is given in m/s

    Notes
    -----
    The speed at one positionfix is computed from the distance and time since the previous positionfix. For the first
    positionfix, the speed is set to the same value as for the second one.
    """
    pfs = positionfixes.copy()
    if_planar_crs = ti.geogr.distances.check_gdf_planar(pfs)

    # get next location and time
    pfs["prev_geom"] = pfs["geom"].shift(1)
    pfs.loc[pfs.index[0], "prev_geom"] = pfs.loc[pfs.index[0], "geom"]
    pfs["prev_tracked_at"] = pfs["tracked_at"].shift(1)

    # get distance and time difference
    if if_planar_crs:
        dist_function = lambda point: point.geom.distance(point.prev_geom)
    else:
        dist_function = lambda point: ti.geogr.point_distances.haversine_dist(
            point.geom.x, point.geom.y, point.prev_geom.x, point.prev_geom.y
        )[0]
    pfs["dist"] = pfs.apply(dist_function, axis=1)
    pfs["time_diff"] = pfs.apply(lambda x: x.tracked_at - x.prev_tracked_at, axis=1).dt.total_seconds()

    # compute speed (in m/s)
    pfs["speed"] = pfs["dist"] / pfs["time_diff"]
    # The first point speed is imputed
    pfs.loc[pfs.index[0], "speed"] = pfs.iloc[1]["speed"]

    pfs.drop(columns=["prev_geom", "prev_tracked_at", "dist", "time_diff"], inplace=True)

    return pfs


def get_speed_triplegs(triplegs, positionfixes=None, method="tpls_speed"):
    """
    Compute the average speed per positionfix for each tripleg (in m/s)

    Parameters
    ----------
    triplegs: GeoDataFrame (as trackintel triplegs)
        The generated triplegs as returned by ti.preprocessing.positionfixes.generate_triplegs

    positionfixes (Optional): GeoDataFrame (as trackintel positionfixes)
        The positionfixes as returned by ti.preprocessing.positionfixes.generate_triplegs. Only required if the method
        is 'pfs_mean_speed'. In addition the standard columns it must include the column ``[`tripleg_id`]``.

    method: str
        Method how the speed is computed, one of {tpls_speed, pfs_mean_speed}. The 'tpls_speed' method simply divides
        the overall tripleg distance by its duration, while the 'pfs_mean_speed' method is the mean pfs speed.

    Returns
    -------
    tpls: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`speed`]``. The speed is given in m/s.
    """
    # Simple method: Divide overall tripleg distance by overall duration
    if method == "tpls_speed":
        tpls = triplegs.copy()
        # check what distance function we need to compute tripleg distance
        if_planar_crs = ti.geogr.distances.check_gdf_planar(tpls)
        if not if_planar_crs:
            tpls_distance = calculate_haversine_length(triplegs)
        else:
            tpls_distance = triplegs.length
        duration = (tpls["finished_at"] - tpls["started_at"]).apply(lambda x: x.total_seconds())
        # The unit of the speed is m/s
        tpls["speed"] = tpls_distance / duration
        return tpls
    # Pfs-based method: compute speed per positionfix and average then
    elif method == "pfs_mean_speed":
        assert positionfixes is not None, "Method pfs_mean_speed requires positionfixes as input"
        assert "tripleg_id" in positionfixes.columns, "Positionfixes must include column tripleg_id"
        # group positionfixes by triplegs and compute average speed for each collection of positionfixes
        grouped_pfs = positionfixes.groupby("tripleg_id").apply(_single_tripleg_mean_speed)
        # add the speed values to the triplegs column
        tpls = pd.merge(triplegs, grouped_pfs.rename("speed"), left_index=True, right_index=True)
        tpls.index = tpls.index.astype("int64")
        return tpls
    else:
        raise AttributeError(f"Method {method} not known for speed computation.")


def _single_tripleg_mean_speed(positionfixes):
    pfs_sorted = positionfixes.sort_values(by="tracked_at")
    pfs_speed = get_speed_positionfixes(pfs_sorted)
    return np.mean(pfs_speed["speed"].values[1:])


def _copy_docstring(wrapped, assigned=("__doc__",), updated=[]):
    """Thin wrapper for `functools.update_wrapper` to mimic `functools.wraps` but to only copy the docstring."""
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)
