import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    doc,
    _shared_docs,
)

_required_columns = ["user_id", "started_at", "finished_at"]


@_register_trackintel_accessor("as_staypoints")
class Staypoints(TrackintelBase, TrackintelGeoDataFrame):
    """A pandas accessor to treat a GeoDataFrame as collections of `Staypoints`.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at']

    Requires valid point geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `Staypoints`.

    For several usecases, the following additional columns are required:
    ['elevation', 'purpose', 'is_activity', 'next_trip_id', 'prev_trip_id', 'trip_id',
    location_id]

    Notes
    -----
    `Staypoints` are defined as location were a person did not move for a while.
    Under consideration of location uncertainty this means that a person stays within
    a close proximity for a certain amount of time.
    The exact definition is use-case dependent.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_staypoints.generate_locations()
    """

    def __init__(self, *args, validate_geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate(self, validate_geometry=validate_geometry)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_staypoints(self):
        return self

    @staticmethod
    def _validate(obj, validate_geometry=True):
        # check columns
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of staypoints, it must have the properties"
                f" {_required_columns}, but it has {', '.join(obj.columns)}."
            )
        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be tz aware datetime64"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be tz aware datetime64"

        if validate_geometry:
            # check geometry
            assert (
                obj.geometry.is_valid.all()
            ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"
            if obj.geometry.iloc[0].geom_type != "Point":
                raise AttributeError("The geometry must be a Point (only first checked).")

    @staticmethod
    def _check(obj, validate_geometry=True):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if not isinstance(obj["started_at"].dtype, pd.DatetimeTZDtype):
            return False
        if not isinstance(obj["finished_at"].dtype, pd.DatetimeTZDtype):
            return False
        if validate_geometry:
            return obj.geometry.is_valid.all() and obj.geometry.iloc[0].geom_type == "Point"
        return True

    @property
    def center(self):
        """Return the center coordinate of this collection of staypoints."""
        lat = self.geometry.y
        lon = self.geometry.x
        return (float(lon.mean()), float(lat.mean()))

    def generate_locations(
        self,
        method="dbscan",
        epsilon=100,
        num_samples=1,
        distance_metric="haversine",
        agg_level="user",
        activities_only=False,
        print_progress=False,
        n_jobs=1,
    ):
        # if you update this docstring update ti.preprocessing.generate_locations as well
        """
        Generate locations from the staypoints.

        Parameters
        ----------
        method : {'dbscan'}
            Method to create locations.
            - 'dbscan' : Uses the DBSCAN algorithm to cluster staypoints.

        epsilon : float, default 100
            The epsilon for the 'dbscan' method. if 'distance_metric' is 'haversine'
            or 'euclidean', the unit is in meters.

        num_samples : int, default 1
            The minimal number of samples in a cluster.

        distance_metric: {'haversine', 'euclidean'}
            The distance metric used by the applied method. Any mentioned below are possible:
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

        agg_level: {'user','dataset'}
            The level of aggregation when generating locations:
            - 'user'      : locations are generated independently per-user.
            - 'dataset'   : shared locations are generated for all users.

        activities_only: bool, default False (requires "activity" column)
            Flag to set if locations should be generated only from staypoints on which the value for "activity" is True.
            Useful if activites represent more significant places.

        print_progress : bool, default False
            If print_progress is True, the progress bar is displayed

        n_jobs: int, default 1
            The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging. See
            https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
            for a detailed description

        Returns
        -------
        sp: GeoDataFrame (as trackintel staypoints)
            The original staypoints with a new column ``[`location_id`]``.

        locs: GeoDataFrame (as trackintel locations)
            The generated locations.

        Examples
        --------
        >>> sp.as_staypoints.generate_locations(method='dbscan', epsilon=100, num_samples=1)
        """
        return ti.preprocessing.staypoints.generate_locations(
            self,
            method=method,
            epsilon=epsilon,
            num_samples=num_samples,
            distance_metric=distance_metric,
            agg_level=agg_level,
            activities_only=activities_only,
            print_progress=print_progress,
            n_jobs=n_jobs,
        )

    def merge_staypoints(self, triplegs, max_time_gap="10min", agg={}):
        # if you update this docstring update ti.preprocessing.staypoints.merge_staypoints as well
        """
        Aggregate staypoints horizontally via time threshold.

        Staypoints must contain a column `location_id` (see `generate_locations` function)

        Parameters
        ----------
        triplegs: GeoDataFrame (as trackintel triplegs)

        max_time_gap : str or pd.Timedelta, default "10min"
            Maximum duration between staypoints to still be merged.
            If str must be parsable by pd.to_timedelta.

        agg: dict, optional
            Dictionary to aggregate the rows after merging staypoints. This dictionary is used as input to the pandas
            aggregate function: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html
            If empty, only the required columns of staypoints (which are ['user_id', 'started_at', 'finished_at']) are
            aggregated and returned. In order to return for example also the geometry column of the merged staypoints, set
            'agg={"geom":"first"}' to return the first geometry of the merged staypoints, or 'agg={"geom":"last"}' to use
            the last one.

        Returns
        -------
        sp: DataFrame
            The new staypoints with the default columns and columns in `agg`, where staypoints at same location and close in
            time are aggregated.

        Notes
        -----
        - Due to the modification of the staypoint index, the relation between the staypoints and the corresponding
        positionfixes **is broken** after execution of this function! In explanation, the staypoint_id column of pfs does
        not necessarily correspond to an id in the new sp table that is returned from this function. The same holds for
        trips (if generated yet) where the staypoints contained in a trip might be merged in this function.
        - If there is a tripleg between two staypoints, the staypoints are not merged. If you for some reason want to merge
        such staypoints, simply pass an empty DataFrame for the tpls argument.

        Examples
        --------
        >>> # direct function call
        >>> ti.preprocessing.staypoints.merge_staypoints(staypoints=sp, triplegs=tpls)
        >>> # or using the trackintel datamodel
        >>> sp.as_staypoints.merge_staypoints(triplegs, max_time_gap="1h", agg={"geom":"first"})
        """
        return ti.preprocessing.staypoints.merge_staypoints(self, triplegs, max_time_gap=max_time_gap, agg=agg)

    def create_activity_flag(self, method="time_threshold", time_threshold=15.0, activity_column_name="is_activity"):
        # if you update this docstring update ti.analysis.labelling.create_activity_flag as well.
        """
        Add a flag whether or not a staypoint is considered an activity based on a time threshold.

        Parameters
        ----------
        staypoints: GeoDataFrame (as trackintel staypoints)

        method: {'time_threshold'}, default = 'time_threshold'
            - 'time_threshold' : All staypoints with a duration greater than the time_threshold are considered an activity.

        time_threshold : float, default = 15 (minutes)
            The time threshold for which a staypoint is considered an activity in minutes. Used by method 'time_threshold'

        activity_column_name : str , default = 'is_activity'
            The name of the newly created column that holds the activity flag.

        Returns
        -------
        staypoints : GeoDataFrame (as trackintel staypoints)
            Original staypoints with the additional activity column

        Examples
        --------
        >>> sp  = sp.as_staypoints.create_activity_flag(method='time_threshold', time_threshold=15)
        >>> print(sp['is_activity'])
        """
        return ti.analysis.labelling.create_activity_flag(
            self, method=method, time_threshold=time_threshold, activity_column_name=activity_column_name
        )

    def spatial_filter(self, areas, method="within", re_project=False):
        # if you update this docstring update ti.preprocessing.filter.spatial_filter as well.
        """
        Filter Staypoints on a geo extent.

        Parameters
        ----------
        areas : GeoDataFrame
            The areas used to perform the spatial filtering. Note, you can have multiple Polygons
            and it will return all the features intersect with ANY of those geometries.

        method : {'within', 'intersects', 'crosses'}, optional
            The method to filter the 'source' GeoDataFrame, by default 'within'
            - 'within'    : return instances in 'source' where no points of these instances lies in the
                exterior of the 'areas' and at least one point of the interior of these instances lies
                in the interior of 'areas'.
            - 'intersects': return instances in 'source' where the boundary or interior of these instances
                intersect in any way with those of the 'areas'
            - 'crosses'   : return instances in 'source' where the interior of these instances intersects
                the interior of the 'areas' but does not contain it, and the dimension of the intersection
                is less than the dimension of the one of the 'areas'.

        re_project : bool, default False
            If this is set to True, the 'source' will be projected to the coordinate reference system of 'areas'

        Returns
        -------
        GeoDataFrame (as trackintel staypoints)
            GeoDataFrame containing the features after the spatial filtering.

        Examples
        --------
        >>> sp.as_staypoints.spatial_filter(areas, method="within", re_project=False)
        """
        return ti.preprocessing.filter.spatial_filter(self, areas, method=method, re_project=re_project)

    @doc(_shared_docs["write_csv"], first_arg="", long="staypoints", short="sp")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.file.write_staypoints_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="staypoints", short="sp")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.postgis.write_staypoints_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def temporal_tracking_quality(self, granularity="all"):
        # if you update this docstring update ti.analysis.tracking_quality as well.
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
        >>> temporal_tracking_quality(sp, granularity="all")
        >>> # calculate per-day tracking quality of sp and tpls sequence
        >>> temporal_tracking_quality(sp_tpls, granularity="day")
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self, granularity=granularity)
