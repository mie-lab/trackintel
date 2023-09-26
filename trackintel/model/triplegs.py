import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)

_required_columns = ["user_id", "started_at", "finished_at"]


@_register_trackintel_accessor("as_triplegs")
class Triplegs(TrackintelBase, TrackintelGeoDataFrame):
    """A pandas accessor to treat a GeoDataFrame as a collections of `Tripleg`.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at']

    Requires valid line geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `triplegs`

    For several usecases, the following additional columns are required:
    ['mode', 'trip_id']

    Notes
    -----
    A `Tripleg` (also called `stage`) is defined as continuous movement without changing the mode of transport.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_triplegs.generate_trips()
    """

    def __init__(self, *args, validate_geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate(self, validate_geometry=validate_geometry)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_triplegs(self):
        return self

    @staticmethod
    def _validate(obj, validate_geometry=True):
        assert obj.shape[0] > 0, f"Geodataframe is empty with shape: {obj.shape}"
        # check columns
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of triplegs, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )

        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

        # check geometry
        if validate_geometry:
            assert (
                obj.geometry.is_valid.all()
            ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"
            if obj.geometry.iloc[0].geom_type != "LineString":
                raise AttributeError("The geometry must be a LineString (only first checked).")

    @staticmethod
    def _check(obj, validate_geometry=True):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if obj.shape[0] <= 0:
            return False
        if not isinstance(obj["started_at"].dtype, pd.DatetimeTZDtype):
            return False
        if not isinstance(obj["finished_at"].dtype, pd.DatetimeTZDtype):
            return False
        if validate_geometry:
            return obj.geometry.is_valid.all() and obj.geometry.iloc[0].geom_type == "LineString"
        return True

    @doc(_shared_docs["write_csv"], first_arg="", long="triplegs", short="tpls")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.file.write_triplegs_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="triplegs", short="tpls")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.postgis.write_triplegs_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def calculate_distance_matrix(self, Y=None, dist_metric="haversine", n_jobs=0, **kwds):
        # if you update this docstring update ti.geogr.calculate_distance_matrix as well.
        """
        Calculate a distance matrix based on a specific distance metric.

        If no Y is given, the pair-wise distances between all elements in self are calculated.
        If Y is given, the distances between all combinations of self and Y are calculated.
        Distances between elements of self and self, and distances between elements of Y and Y are not calculated.

        Parameters
        ----------
        Y : GeoDataFrame (as trackintel triplegs), optional
            Should be of the same type as self

        dist_metric: {'haversine', 'euclidean', 'dtw', 'frechet'}, optional
            The distance metric to be used for calculating the matrix. By default 'haversine.

            For staypoints or positionfixes, a common choice is 'haversine' or 'euclidean'. This function wraps around
            the ``pairwise_distance`` function from scikit-learn if only `X` is given and wraps around the
            ``scipy.spatial.distance.cdist`` function if X and Y are given.
            Therefore the following metrics are also accepted:

            via ``scikit-learn``: `['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']`

            via ``scipy.spatial.distance``: `['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule']`

            For triplegs, common choice is 'dtw' or 'frechet'. This function uses the implementation
            from similaritymeasures.

        n_jobs: int, optional
            Number of cores to use: 'dtw', 'frechet' and all distance metrics from `pairwise_distance` (only available
            if only X is given) are parallelized. By default 1.

        **kwds:
            optional keywords passed to the distance functions.

        Returns
        -------
        D: np.array
            matrix of shape (len(X), len(X)) or of shape (len(X), len(Y)) if Y is provided.

        Examples
        --------
        >>> calculate_distance_matrix(staypoints, dist_metric="haversine")
        >>> calculate_distance_matrix(triplegs_1, triplegs_2, dist_metric="dtw")
        >>> tpls.as_triplegs.calculate_distance_matrix(dist_metric="haversine")
        """
        return ti.geogr.calculate_distance_matrix(self, Y=Y, dist_metric=dist_metric, n_jobs=n_jobs, **kwds)

    def spatial_filter(self, areas, method="within", re_project=False):
        # if you update this docstring update ti.preprocessing.filter.spatial_filter as well.
        """
        Filter Triplegs on a geo extent.

        Parameters
        ----------
        areas : GeoDataFrame
            The areas used to perform the spatial filtering. Note, you can have multiple Polygons
            and it will return all the features intersect with ANY of those geometries.

        method : {'within', 'intersects', 'crosses'}, optional
            The method to filter the 'source' GeoDataFrame, by default 'within'
            - 'within' : return instances in 'source' where no points of these instances lies in the
                exterior of the 'areas' and at least one point of the interior of these instances lies
                in the interior of 'areas'.
            - 'intersects': return instances in 'source' where the boundary or interior of these instances
                intersect in any way with those of the 'areas'
            - 'crosses' : return instances in 'source' where the interior of these instances intersects
                the interior of the 'areas' but does not contain it, and the dimension of the intersection
                is less than the dimension of the one of the 'areas'.

        re_project : bool, default False
            If this is set to True, the 'source' will be projected to the coordinate reference system of 'areas'

        Returns
        -------
        GeoDataFrame (as trackintel triplegs)
            GeoDataFrame containing the features after the spatial filtering.

        Examples
        --------
        >>> tpls.as_triplegs.spatial_filter(areas, method="within", re_project=False)
        """
        return ti.preprocessing.filter.spatial_filter(self, areas, method=method, re_project=re_project)

    def generate_trips(self, staypoints, gap_threshold=15, add_geometry=True):
        # if you update this docstring update ti.preprocessing.generate_triplegs as well
        """
        Generate trips based on staypoints and triplegs.

        Parameters
        ----------
        staypoints : GeoDataFrame (as trackintel staypoints)

        gap_threshold : float, default 15 (minutes)
            Maximum allowed temporal gap size in minutes. If tracking data is missing for more than
            `gap_threshold` minutes, then a new trip begins after the gap.

        add_geometry : bool default True
            If True, the start and end coordinates of each trip are added to the output table in a geometry column "geom"
            of type MultiPoint. Set `add_geometry=False` for better runtime performance (if coordinates are not required).

        print_progress : bool, default False
            If print_progress is True, the progress bar is displayed

        Returns
        -------
        sp: GeoDataFrame (as trackintel staypoints)
            The original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.

        tpls: GeoDataFrame (as trackintel triplegs)
            The original triplegs with a new column ``[`trip_id`]``.

        trips: (Geo)DataFrame (as trackintel trips)
            The generated trips.

        Notes
        -----
        Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
        (e.g., waiting) between two relevant activities.
        The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
        [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
        The following assumptions are implemented

            - If we do not record a person for more than `gap_threshold` minutes,
            we assume that the person performed an activity in the recording gap and split the trip at the gap.
            - Trips that start/end in a recording gap can have an unknown origin/destination
            - There are no trips without a (recorded) tripleg
            - Trips optionally have their start and end point as geometry of type MultiPoint, if `add_geometry==True`
            - If the origin (or destination) staypoint is unknown, and `add_geometry==True`, the origin (and destination)
            geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg),
            respectively. Trips with missing values can still be identified via col `origin_staypoint_id`.


        Examples
        --------
        >>> from trackintel.preprocessing.triplegs import generate_trips
        >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)

        trips can also be directly generated using the tripleg accessor
        >>> staypoints, triplegs, trips = triplegs.as_triplegs.generate_trips(staypoints)
        """
        return ti.preprocessing.triplegs.generate_trips(
            staypoints, self, gap_threshold=gap_threshold, add_geometry=add_geometry
        )

    def predict_transport_mode(self, method="simple-coarse", **kwargs):
        # if you update this docstring update ti.analysis.labelling.predict_transport_mode as well
        """
        Predict the transport mode of triplegs.

        Predict/impute the transport mode that was likely chosen to cover the given
        tripleg, e.g., car, bicycle, or walk.

        Parameters
        ----------
        method: {'simple-coarse'}, default 'simple-coarse'
            The following methods are available for transport mode inference/prediction:
            - 'simple-coarse' : Uses simple heuristics to predict coarse transport classes.

        Returns
        -------
        triplegs : GeoDataFrame (as trackintel triplegs)
            The triplegs with added column mode, containing the predicted transport modes.

        Notes
        -----
        ``simple-coarse`` method includes ``{'slow_mobility', 'motorized_mobility', 'fast_mobility'}``.
        In the default classification, ``slow_mobility`` (<15 km/h) includes transport modes such as
        walking or cycling, ``motorized_mobility`` (<100 km/h) modes such as car or train, and
        ``fast_mobility`` (>100 km/h) modes such as high-speed rail or airplanes.
        These categories are default values and can be overwritten using the keyword argument categories.

        Examples
        --------
        >>> tpls  = tpls.as_triplegs.predict_transport_mode()
        >>> print(tpls["mode"])
        """
        return ti.analysis.labelling.predict_transport_mode(self, method=method, **kwargs)

    def calculate_modal_split(self, freq=None, metric="count", per_user=False, norm=False):
        # if you update this docstring update ti.analysis.modal_split.calculate_modal_split as well.
        """
        Calculate the modal split of triplegs.

        Triplegs require column `mode` on which the modal split is calculated.

        Parameters
        ----------
        freq : str
            frequency string passed on as `freq` keyword to the pandas.Grouper class. If `freq=None` the modal split is
            calculated on all data. A list of possible
            values can be found `here <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset
            -aliases>`_.
        metric : {'count', 'distance', 'duration'}
            Aggregation used to represent the modal split. 'distance' returns in the same unit as the crs. 'duration'
            returns values in seconds.
        per_user : bool, default: False
            If True the modal split is calculated per user
        norm : bool, default: False
            If True every row of the modal split is normalized to 1

        Returns
        -------
        modal_split : DataFrame
            The modal split represented as pandas Dataframe with (optionally) a multi-index. The index can have the
            levels: `('user_id', 'timestamp')` and every mode as a column.

        Notes
        ------
            `freq='W-MON'` is used for a weekly aggregation that starts on mondays.

            If `freq=None` and `per_user=False` are passed the modal split collapses to a single column.

            The modal split can be visualized using :func:`trackintel.visualization.plot_modal_split`

        Examples
        --------
        >>> assert "mode" is in triplegs.columns
        >>> triplegs.calculate_modal_split()
        >>> triplegs.calculate_modal_split(freq='W-MON', metric='distance')
        """
        return ti.analysis.modal_split.calculate_modal_split(
            self, freq=freq, metric=metric, per_user=per_user, norm=norm
        )

    def temporal_tracking_quality(self, granularity="all"):
        # if you update this docstring update ti.analysis.(...).temporal_tracking as well.
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        Parameters
        ----------
        granularity : {"all", "day", "week", "weekday", "hour"}
            The level of which the tracking quality is calculated. The default "all" returns
            the overall tracking quality; "day" the tracking quality by days; "week" the quality
            by weeks; "weekday" the quality by day of the week (e.g, Mondays, Tuesdays, etc.) and
            "hour" the quality by hours.

        Returns
        -------
        quality: DataFrame
            A per-user per-granularity temporal tracking quality dataframe.

        Notes
        -----
        Requires at least the following columns:
        ``['user_id', 'started_at', 'finished_at']``
        which means the function supports trackintel ``staypoints``, ``triplegs``, ``trips`` and ``tours``
        datamodels and their combinations (e.g., staypoints and triplegs sequence).

        The temporal tracking quality is the ratio of tracking time and the total time extent. It is
        calculated and returned per-user in the defined ``granularity``. The time extents
        and the columns for the returned ``quality`` df for different ``granularity`` are:

        - ``all``:
            - time extent: between the latest "finished_at" and the earliest "started_at" for each user.
            - columns: ``['user_id', 'quality']``.

        - ``week``:
            - time extent: the whole week (604800 sec) for each user.
            - columns: ``['user_id', 'week_monday', 'quality']``.

        - ``day``:
            - time extent: the whole day (86400 sec) for each user
            - columns: ``['user_id', 'day', 'quality']``

        - ``weekday``
            - time extent: the whole day (86400 sec) * number of tracked weeks for each user for each user
            - columns: ``['user_id', 'weekday', 'quality']``

        - ``hour``:
            - time extent: the whole hour (3600 sec) * number of tracked days for each user
            - columns: ``['user_id', 'hour', 'quality']``

        Examples
        --------
        >>> # calculate overall tracking quality of staypoints
        >>> temporal_tracking_quality(tpls, granularity="all")
        >>> # calculate per-day tracking quality of sp and tpls sequence
        >>> temporal_tracking_quality(sp_tpls, granularity="day")
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self, granularity=granularity)

    def get_speed(self, positionfixes=None, method="tpls_speed"):
        # if you update this docstring update ti.geogr.get_speed_triplegs as well
        """
        Compute the average speed per positionfix for each tripleg (in m/s)

        Parameters
        ----------
        positionfixes: GeoDataFrame (as trackintel positionfixes), optional
            Only required if the method is 'pfs_mean_speed'.
            In addition to the standard columns positionfixes must include the column ``[`tripleg_id`]``.

        method: {'tpls_speed', 'pfs_mean_speed'}, optional
            Method how of speed calculation, default is "tpls_speed"
            The 'tpls_speed' method divides the tripleg distance by its duration,
            the 'pfs_mean_speed' method calculates the speed via the mean speed of the positionfixes of a tripleg.

        Returns
        -------
        tpls: GeoDataFrame (as trackintel triplegs)
            The original triplegs with a new column ``[`speed`]``. The speed is given in m/s.
        """
        return ti.geogr.get_speed_triplegs(self, positionfixes=positionfixes, method=method)
