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
            c[i, j] = t0.iloc[i - 1]['geom'].distance(t1.iloc[j - 1]['geom']) + min(c[i, j - 1], c[i - 1, j - 1], c[i - 1, j])
    dtw = c[n0, n1]
    return dtw



def e_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.

    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float

    Returns
    -------
    edr : float
           The Edit Distance on real Sequence between trajectory t0 and t1
    """
    n0 = t0.shape[0]
    n1 = t1.shape[0]
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if t0.iloc[i-1].geom.distance(t1.iloc[j-1].geom) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr
























