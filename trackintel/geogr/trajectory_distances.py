from __future__ import division

import numpy as np
from scipy.spatial import distance
from scipy.spatial import minkowski_distance


def _c(ca, i, j, P, Q):
    """
    https://github.com/cjekel/similarity_measures/blob/master/similaritymeasures/similaritymeasures.py
    Recursive caller for discrete Frechet distance
    This is the recursive caller as as defined in [1]_.
    
    Parameters
    ----------
    ca : array_like
        distance like matrix
        
    i : int
        index
        
    j : int
        index
        
    P : array_like
        array containing path P
        
    Q : array_like
        array containing path Q
        
    Returns
    -------
    df : float
        discrete frechet distance
        
    Notes
    -----
    This should work in N-D space. Thanks to MaxBareiss
    https://gist.github.com/MaxBareiss/ba2f9441d9455b56fbc9
    
    """
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = minkowski_distance(P[0], Q[0], p=pnorm)
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, P, Q),
                       minkowski_distance(P[i], Q[0], p=pnorm))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, P, Q),
                       minkowski_distance(P[0], Q[j], p=pnorm))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i-1, j, P, Q), _c(ca, i-1, j-1, P, Q),
                       _c(ca, i, j-1, P, Q)),
                       minkowski_distance(P[i], Q[j], p=pnorm))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


def frechet_dist(exp_data, num_data, p=2):
    """
    Compute the discrete Frechet distance.
    
    Compute the Discrete Frechet Distance between two N-D curves according to
    [1]_. The Frechet distance has been defined as the walking dog problem.
    From Wikipedia: "In mathematics, the Frechet distance is a measure of
    similarity between curves that takes into account the location and
    ordering of the points along the curves.
    
    code from https://github.com/cjekel/similarity_measures/blob/master/similaritymeasures/similaritymeasures.py
    
    Parameters
    ----------
    exp_data : array_like
        Curve from your experimental data. exp_data is of (M, N) shape, where
        M is the number of data points, and N is the number of dimmensions
        
    num_data : array_like
        Curve from your numerical data. num_data is of (P, N) shape, where P
        is the number of data points, and N is the number of dimmensions
        
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use. Default is p=2 (Eculidean).
        The manhattan distance is p=1.
        
    Returns
    -------
    df : float
        discrete Frechet distance
        
    References
    ----------
    .. [1] Thomas Eiter and Heikki Mannila. Computing discrete Frechet
        distance. Technical report, 1994.
        http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.937&rep=rep1&type=pdf
    
    Notes
    -----
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.
    
    Python has a default limit to the amount of recursive calls a single
    function can make. If you have a large dataset, you may need to increase
    this limit. Check out the following resources.
    https://docs.python.org/3/library/sys.html#sys.setrecursionlimit
    https://stackoverflow.com/questions/3323001/what-is-the-maximum-recursion-depth-in-python-and-how-to-increase-it
    Thanks to MaxBareiss
    https://gist.github.com/MaxBareiss/ba2f9441d9455b56fbc9
    This sets a global variable named pnorm, where pnorm = p.
    
    Examples
    --------
    >>> # Generate random experimental data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> exp_data = np.zeros((100, 2))
    >>> exp_data[:, 0] = x
    >>> exp_data[:, 1] = y
    >>> # Generate random numerical data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> num_data = np.zeros((100, 2))
    >>> num_data[:, 0] = x
    >>> num_data[:, 1] = y
    >>> df = frechet_dist(exp_data, num_data)
    """
    # set p as global variable
    global pnorm
    pnorm = p
    # Computes the discrete frechet distance between two polygonal lines
    # Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    # exp_data, num_data are arrays of 2-element arrays (points)
    ca = np.ones((len(exp_data), len(num_data)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(exp_data)-1, len(num_data)-1, exp_data, num_data)

def dtw(exp_data, num_data, metric='euclidean', **kwargs):
    """
    Compute the Dynamic Time Warping distance.
    
    This computes a generic Dynamic Time Warping (DTW) distance and follows
    the algorithm from [2]_. This can use all distance metrics that are
    available in scipy.spatial.distance.cdist.
    
    code from https://github.com/cjekel/similarity_measures/blob/master/similaritymeasures/similaritymeasures.py
    
    Parameters
    ----------
    exp_data : array_like
        Curve from your experimental data. exp_data is of (M, N) shape, where
        M is the number of data points, and N is the number of dimmensions
        
    num_data : array_like
        Curve from your numerical data. num_data is of (P, N) shape, where P
        is the number of data points, and N is the number of dimmensions
        
    metric : str or callable, optional
        The distance metric to use. Default='euclidean'. Refer to the
        documentation for scipy.spatial.distance.cdist. Some examples:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
        'wminkowski', 'yule'.
        
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation in
        scipy.spatial.distance.
            
    Returns
    -------
    r : float
        DTW distance.
        
    d : ndarray (2-D)
        Cumulative distance matrix
        
    Notes
    -----
    The DTW distance is d[-1, -1].
    
    This has O(M, P) computational cost.
    
    The latest scipy.spatial.distance.cdist information can be found at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    
    Your x locations of data points should be exp_data[:, 0], and the y
    locations of the data points should be exp_data[:, 1]. Same for num_data.
    
    This uses the euclidean distance for now. In the future it should be
    possible to support other metrics.
    
    DTW is a non-metric distance, which means DTW doesn't hold the triangle
    inequality.
    https://en.wikipedia.org/wiki/Triangle_inequality
    
    References
    ----------
    .. [2] Senin, P., 2008. Dynamic time warping algorithm review. Information
        and Computer Science Department University of Hawaii at Manoa Honolulu,
        USA, 855, pp.1-23.
        http://seninp.github.io/assets/pubs/senin_dtw_litreview_2008.pdf
        
    Examples
    --------
    >>> # Generate random experimental data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> exp_data = np.zeros((100, 2))
    >>> exp_data[:, 0] = x
    >>> exp_data[:, 1] = y
    >>> # Generate random numerical data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> num_data = np.zeros((100, 2))
    >>> num_data[:, 0] = x
    >>> num_data[:, 1] = y
    >>> r, d = dtw(exp_data, num_data)
    The euclidean distance is used by default. You can use metric and **kwargs
    to specify different types of distance metrics. The following example uses
    the city block or Manhattan distance between points.
    >>> r, d = dtw(exp_data, num_data, metric='cityblock')
    """
    c = distance.cdist(exp_data, num_data, metric=metric, **kwargs)

    d = np.zeros(c.shape)
    d[0, 0] = c[0, 0]
    n, m = c.shape
    for i in range(1, n):
        d[i, 0] = d[i-1, 0] + c[i, 0]
    for j in range(1, m):
        d[0, j] = d[0, j-1] + c[0, j]
    for i in range(1, n):
        for j in range(1, m):
            d[i, j] = c[i, j] + min((d[i-1, j], d[i, j-1], d[i-1, j-1]))
    return d[-1, -1]


def dtw_path(d):
    r"""
    Calculates the optimal DTW path from a given DTW cumulative distance
    matrix.
    This function returns the optimal DTW path using the back propagation
    algorithm that is defined in [2]_. This path details the index from each
    curve that is being compared.
    
    Parameters
    ----------
    d : ndarray (2-D)
        Cumulative distance matrix.
        
    Returns
    -------
    path : ndarray (2-D)
        The optimal DTW path.
        
    Notes
    -----
    Note that path[:, 0] represents the indices from exp_data, while
    path[:, 1] represents the indices from the num_data.

    Examples
    --------
    First calculate the DTW cumulative distance matrix.
    >>> # Generate random experimental data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> exp_data = np.zeros((100, 2))
    >>> exp_data[:, 0] = x
    >>> exp_data[:, 1] = y
    >>> # Generate random numerical data
    >>> x = np.random.random(100)
    >>> y = np.random.random(100)
    >>> num_data = np.zeros((100, 2))
    >>> num_data[:, 0] = x
    >>> num_data[:, 1] = y
    >>> r, d = dtw(exp_data, num_data)
    Now you can calculate the optimal DTW path
    >>> path = dtw_path(d)
    You can visualize the path on the cumulative distance matrix using the
    following code.
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.imshow(d.T, origin='lower')
    >>> plt.plot(path[:, 0], path[:, 1], '-k')
    >>> plt.colorbar()
    >>> plt.show()
    """
    path = []
    i, j = d.shape
    i = i - 1
    j = j - 1
    # back propagation starts from the last point,
    # and ends at d[0, 0]
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            temp_step = min([d[i-1, j], d[i, j-1], d[i-1, j-1]])
            if d[i-1, j] == temp_step:
                i = i - 1
            elif d[i, j-1] == temp_step:
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append((i, j))
    path = np.array(path)
    # reverse the order of path, such that it starts with [0, 0]
    return path[::-1]




