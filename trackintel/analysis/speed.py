import trackintel as ti
import numpy as np
import pandas as pd


def speed_positionfixes(positionfixes):
    """
    Compute speed per positionfix

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`speed`]``. The speed is given in km/h

    Notes
    -----
    The speed at one positionfix is computed from the distance and time since the previous positionfix. For the first
    positionfix, the speed is set to the same value as for the second one.
    """
    pfs = positionfixes.copy()
    crs_is_projected = ti.geogr.distances.check_gdf_planar(pfs)

    if crs_is_projected:
        dist_function = lambda x: x.geom.distance(x.prev_geom)
    else:
        dist_function = lambda x: ti.geogr.point_distances.haversine_dist(
            x.geom.x, x.geom.y, x.prev_geom.x, x.prev_geom.y
        )[0]

    # get next location and time
    pfs["prev_geom"] = pfs["geom"].shift(1)
    pfs.loc[pfs.index[0], "prev_geom"] = pfs.loc[pfs.index[0], "geom"]
    pfs["prev_tracked_at"] = pfs["tracked_at"].shift(1)

    # get distance and time difference
    pfs["dist"] = pfs.apply(dist_function, axis=1)
    pfs["time_diff"] = pfs.apply(lambda x: x.tracked_at - x.prev_tracked_at, axis=1).dt.total_seconds()

    # compute speed and convert to kmh
    pfs["speed"] = 3.6 * pfs["dist"] / pfs["time_diff"]
    pfs.loc[pfs.index[0], "speed"] = pfs.iloc[1]["speed"]

    pfs.drop(columns=["prev_geom", "prev_tracked_at", "dist", "time_diff"], inplace=True)

    return pfs


def mean_speed_triplegs(positionfixes, triplegs):
    """
    Compute the average speed (in km/h) for each tripleg

    Parameters
    ----------
    triplegs: GeoDataFrame (as trackintel triplegs)
        The generated triplegs as returned by ti.preprocessing.positionfixes.generate_triplegs

    positionfixes: GeoDataFrame (as trackintel positionfixes)
        The positionfixes as returned by ti.preprocessing.positionfixes.generate_triplegs. In addition the standard columns
        it must include the column ``[`tripleg_id`]``.

    Returns
    -------
    tpls: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`mean_speed`]``. The speed is given in km/h.
    """
    assert "tripleg_id" in positionfixes.columns, "Positionfixes must include column tripleg_id"
    # group positionfixes by triplegs and compute average speed for each collection of positionfixes
    grouped_pfs = positionfixes.groupby("tripleg_id").apply(_single_tripleg_mean_speed)
    # add the speed values to the triplegs column
    tpls = pd.merge(triplegs, grouped_pfs.rename("mean_speed"), left_index=True, right_index=True)
    tpls.index = tpls.index.astype("int64")
    return tpls


def _single_tripleg_mean_speed(positionfixes):
    pfs_sorted = positionfixes.sort_values(by="tracked_at")
    pfs_speed = speed_positionfixes(pfs_sorted)
    return np.mean(pfs_speed["speed"].values[1:])
