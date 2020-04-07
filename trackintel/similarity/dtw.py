"""
Algorithm adapted from https://github.com/bguillouet/traj-dist

"""


import numpy as np


def e_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1. The timestamps
    in the GeoDataFrames are not (yet) considered.

    Parameters
    ----------
    param t0 : GeoDataFrame with n0 Points
    param t1 : GeoDataFrame with n1 Points

    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = t0.shape[0]
    n1 = t1.shape[0]
    c = np.zeros((n0 + 1, n1 + 1))
    c[1:, 0] = float('inf')
    c[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            c[i, j] = t0.loc[i - 1, 'geom'].distance(t1[i - 1, 'geom']) + min(c[i, j - 1], c[i - 1, j - 1], c[i - 1, j])
    dtw = c[n0, n1]
    return dtw
