# -*- coding: utf-8 -*-
# @author: Christof Leutenegger

from sklearn.covariance import MinCovDet
from sklearn.utils.validation import check_array
import numpy as np


class MCD(MinCovDet):
    """
    Minimum Covariance Determinant (MCD): robust estimator of covariance.

    The Minimum Covariance Determinant covariance estimator is to be applied
    on Gaussian-distributed data, but could still be relevant on data
    drawn from a unimodal, symmetric distribution. It is not meant to be used
    with multi-modal data (the algorithm used to fit a MinCovDet object is
    likely to fail in such a case).
    One should consider projection pursuit methods to deal with multi-modal
    datasets.
    Read more in scikit-implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html

    Parameters
    ----------
    contamination : float in (0., 0.5], default=0.1
        The proportion of contamination in the dataset.
    store_precision : bool, default=True
        Specify if the estimated precision is stored.
    assume_centered : bool, default=False
        If True, the support of the robust location and the covariance
        estimates is computed, and a covariance estimate is recomputed from
        it, without centering the data.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, the robust location and covariance are directly computed
        with the FastMCD algorithm without additional treatment.
    support_fraction : float (0., 1.), optional
        The proportion of points to be included in the support of the raw
        MCD estimate. Default is None, which implies that the minimum
        value of support_fraction will be used within the algorithm:
        [n_sample + n_features + 1] / 2
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [Rousseeuw] A Fast Algorithm for the Minimum Covariance Determinant
        Estimator, 1999, American Statistical Association and the American
        Society for Quality, TECHNOMETRICS
    [Hardin] Outlier Detection in the Multiple Cluster Setting Using the
        Minimum Covariance Determinant Estimator, 2004, Computational
        Statistics & Data Analysis 44, Nr. 4: 625–38.
        https://doi.org/10.1016/S0167-9473(02)00280-3.

    """

    def __init__(self,
                 contamination=0.1,
                 store_precision=True,
                 assume_centered=False,
                 support_fraction=None,
                 random_state=None):

        self.contamination = contamination
        assert(0 <= contamination <= 0.5)
        super().__init__(store_precision,
                         assume_centered,
                         support_fraction,
                         random_state)

    def fit(self, X, y=None):
        """
        Fit MCD on data with FastMCD algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, only for API consistency

        Returns
        -------
        self : object
            DESCRIPTION.

        """
        super().fit(X)
        # Use mahalanabis distance as the outlier score
        return self

    def outlier_score(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on the
        Mahalanobis distance. Outliers are assigned with larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        X = check_array(X)

        # Computer mahalanobis distance of the samples
        return self.mahalanobis(X)

    def predict(self, X):
        """
        Predict binary decision of X using the fitted detector.

        The decision is based on the given contamination.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        Returns
        -------
        ndarray, shape (n_samples, )

        """
        self.threshold_ = np.percentile(self.dist_,
                                        100 * (1 - self.contamination))
        is_inlier = np.ones(X.shape[0], dtype=int)
        # outlier determined with contamination rate
        is_inlier[self.outlier_score(X) > self.threshold_] = -1
        return is_inlier

    def decision_function(self, X):
        return -self.outlier_score(X) + self.threshold_
