import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from trackintel.geogr.distances import haversine_dist as haversine
from iForest import iForest
from MCD import MCD
from LOF import LOF

AVAILABLE_METHODS = {
    'iForest': iForest,
    'LOF': LOF,
    'MCD': MCD
    }


class AnomalyFeatures():
    """
    Class to create features on which to perform outlier detection algorithms.

    Use the same class for creating the same features and use the same scaler
    for the transformation of the data.

    Parameters
    ----------
    dist_metric : string, optional
        Metric for angles and distances. The default is 'haversine'.
        Two metrices are available 'euclidean' and 'haversine'.

    features : str or iterable, default=None
        Select the features that are generated.
        It is also possible to include the column names of already provided
        features.
        If None selects features that we think of as useful.
        To see what features are available for generation `available_features`.

    log_transf: bool, default=True
        If the data should be transformed into log-space.
        Used on total_len, avg_speed,corner_count, duration, centroid_size.
        Not used on proportional feature like ratio baseline length.

    scaler : bool, ScalerObject, default=True
        If True a MinMaxScaler is applied to the data.
        Fit to the first applied data and transform first and any additional
        data.
        A scaler object can be provided to be used instead of MinMaxScaler.

    copy : bool, default=False
        If the input data should be copied.

    Attributes
    ----------
    corner_count_thresh : float, between (0, 180)
        Threshold from where a turn is counted as a corner. In degrees °.

    scaler : scaler object
        Scaler used to transform the data accordingly.

    available_features : list
        List of the methods this class provides

    Returns
    -------
    Tuple of ndarray generated features and list of feature names.

    Example
    -------
    af = AnomalyFeatures()
    train = af.add_features(data_train)
    test = af.add_features(data_test)
    """

    corner_count_thresh = 15.  # in degrees
    scaler = None

    def __init__(self, dist_metric='haversine',
                 features=None, log_transf=True,
                 scaler=False, copy=True):
        self.dist_metric = dist_metric
        if dist_metric not in ['haversine', 'euclidean']:
            raise ValueError(f'{dist_metric} metric is not available.')
        if features is None:
            self.features = ['total_len', 'ratio_baseline_length',
                             'avg_speed', 'corner_count']
        else:
            # put features in a list
            self.features = [features] if isinstance(features, str) else list(features)
        self._log_transf = log_transf

        if isinstance(scaler, bool):
            self._scaler_bool = scaler
            self.scaler_model = MinMaxScaler
        else:
            self._scaler_bool = True
            self.scaler_model = scaler
        self._copy = copy

    @property
    def _available_features(self):
        return {
            'total_len': self._total_len,
            'duration': self._duration,
            'ratio_baseline_length': self._ratio_baseline_length,
            'avg_speed': self._avg_speed,
            'centroid_size': self._centroid_size,
            'corner_count': self._corner_count
            }

    @property
    def available_features(self):
        """Return a list of features that can be generated with this class."""
        return list(self._available_features.keys())

    def _get_line_array(self, row):
        """Return the Linestring as an array."""
        return np.asarray(row['geometry'].coords)

    def _total_len(self, row):
        line_array = self._get_line_array(row)
        if self.dist_metric == 'haversine':
            return haversine(line_array[:-1, 0],
                             line_array[:-1, 1],
                             line_array[1:, 0],
                             line_array[1:, 1]).sum()
        elif self.dist_metric == 'euclidean':
            return np.linalg.norm(line_array[:-1]-line_array[1:], axis=1).sum()

    def _ratio_baseline_length(self, row):
        """Direct distance between the first and last point of a trajectory."""
        line_array = self._get_line_array(row)
        if self.dist_metric == 'haversine':
            # This [0] is only here to circumvent a bug in the haversine_dist.
            # The function returns an array instead of a float.
            base = haversine(*line_array[0], *line_array[-1])[0]
        elif self.dist_metric == 'euclidean':
            base = np.linalg.norm(line_array[0]-line_array[-1])
        return base / row['total_len']

    def _duration(self, row):
        """Return duration in seconds."""
        return (row['finished_at'] - row['started_at']).total_seconds()

    def _centroid_size(self, row):
        """Return standard deviation from the centroid of the trajectory."""
        centroid = row['geometry'].centroid.coords[0]
        line_array = self._get_line_array(row)
        num_points = len(line_array)
        if self.dist_metric == 'haversine':
            d = haversine(*centroid, line_array[:, 0], line_array[:, 1]).sum()
        elif self.dist_metric == 'euclidean':
            d = np.linalg.norm(line_array - centroid, axis=1).sum()
        # Norm by number of points and dimension (2).
        return d / (2 * num_points)

    def _avg_speed(self, row):
        """Return average speed in [m/s]."""
        # lazy calculation of the avage speed.
        tl = row['total_len'] if ('total_len' in row) else self._total_len(row)
        du = row['duration'] if ('duration' in row) else self._duration(row)
        return tl / du

    def _corner_count(self, row):
        line_array = self._get_line_array(row)
        A = line_array[2:, :]
        B = line_array[:-2, :]
        C = line_array[1:-1, :]
        a = np.linalg.norm(B - C, axis=1)
        b = np.linalg.norm(C - A, axis=1)
        c = np.linalg.norm(A - B, axis=1)
        if self.dist_metric == 'haversine':
            # See formula: https://de.wikipedia.org/wiki/Sphärische_Trigonometrie#Halbwinkelsatz
            a = a * np.pi / 180  # from wgs84 coordinates to rad
            b = b * np.pi / 180
            c = c * np.pi / 180
            s = (a + b + c) / 2
            divider = (np.sin(s) * np.sin(s - c))
            # 1e-32 agaist division 0
            divisor = (np.sin(a) * np.sin(b)) + 1e-32
            angles = 2 * np.arccos(np.sqrt(np.abs(divider / divisor)))

        elif self.dist_metric == 'euclidean':
            divider = a**2 + b**2 - c**2
            divisor = 2*a*b + 1e-32
            angles = np.arccos(divider / divisor)

        angles = angles * 180 / np.pi
        # calculate the outer angle
        angles = 180 - np.nan_to_num(angles)
        return (angles > self.corner_count_thresh).sum()

    def add_features(self, data):
        """Add the selected features inplace to GeoDataFrame."""
        if self._copy:
            data = data.copy()

        for key in self._available_features:
            # create all features and select later
            function = self._available_features[key]
            data[key] = data.apply(function, axis=1)

        if self._log_transf:
            with pd.option_context('mode.use_inf_as_na', True):
                log_features = ['total_len', 'avg_speed', 'corner_count',
                                'duration', 'centroid_size']
                data[log_features] = data[log_features].applymap(lambda x: x if x > 0 else 0.001)
                data[log_features] = data[log_features].applymap(np.log)
                data[log_features] = data[log_features].fillna(0)

        # selection for dropping the keys that are not in the selected features
        data = data.drop(
            columns=(self._available_features.keys()-set(self.features))
            )
        # this way the features matches order of the given features
        features = data[self.features].to_numpy()
        # generate and fit a scaler on first call
        if self.scaler is None:
            self.scaler = self.scaler_model()
            self.scaler.fit(data[self.features])

        if self._scaler_bool:
            features = self.scaler.transform(features)
        return features, self.features


def anomaly_detection(triplegs_train,
                      triplegs_test=None,
                      method='iForest',
                      features='auto',
                      return_features=False,
                      contamination='auto',
                      binary=False,
                      method_kwargs=None,
                      features_kwargs=None):
    """
    Calculate outlier score.

    A negative score means the value is seen by the method as an outlier.

    Parameters
    ----------
    triplegs_train : GeoDataFrame
        A trackintel tripleg object.
        Method trains on this object.
    triplegs_test: GeoDataFrame, optional
        A trackintel tripleg object.
        Method tests on this object, if None takes triplegs_train instead.
    method : str, optional
        Method to calculate the outlier score. The default is 'iForest'.
        The following methods are implemented
        'iForest': Uses the sklearn implementation of IsolationForest
        'LOF' : Uses the sklearn implementation of the LocalOutlierFactor
        'MCD' : Uses the sklearn implementation of MinCovDet and
    contamintation: str or float [0., 1.], optional
        If default 'auto' set it according to method else use percantage.
    binary : bool, optional
        Binary decision. The default is False.
        Return score or binary decision (outlier, non-outlier)
    method_kwargs : dict, optional
        The default is None.
    features_kwargs : dict, optional
        The default is None.

    Returns
    -------
    score : ndarray
    """
    if features_kwargs is None:
        features_kwargs = dict()
    if method_kwargs is None:
        method_kwargs = dict()
    if features != 'auto':
        features_kwargs['features'] = features
    feature_class = AnomalyFeatures(**features_kwargs)
    train_feature, columns = feature_class.add_features(triplegs_train)
    if triplegs_test is None:
        test_feature = train_feature
    else:
        test_feature, _ = feature_class.add_features(triplegs_test)

    # fit and predict the method
    if contamination != 'auto':
        method_kwargs['contamination'] = contamination
    try:
        method = AVAILABLE_METHODS[method]
        method = method(**method_kwargs)
    except KeyError:
        raise ValueError(f"{method} not available use \{{", ".join(available_methods.keys())}\} instead")
    method.fit(train_feature)
    # return either score of decision
    if binary:
        result = method.predict(test_feature)
    else:
        result = method.outlier_score(test_feature)

    if return_features:
        return result, (features, columns)
    return result
