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
    """Trackintel class to treat a GeoDataFrame as collections of `Staypoints`.

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
    >>> staypoints.generate_locations()
    """

    def __init__(self, *args, validate=True, **kwargs):
        super().__init__(*args, **kwargs)
        if validate:
            self.validate(self)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_staypoints(self):
        return self

    @staticmethod
    def validate(obj):
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

        # check geometry
        assert (
            obj.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~ x.geometry.is_valid] where x is you GeoDataFrame"
        if obj.geometry.iloc[0].geom_type != "Point":
            raise AttributeError("The geometry must be a Point (only first checked).")

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
        """
        Generate locations from the staypoints.

        See :func:`trackintel.preprocessing.generate_locations` for full documentation.
        """
        return ti.preprocessing.generate_locations(
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
        """
        Aggregate staypoints horizontally via time threshold.

        See :func:`trackintel.preprocessing.merge_staypoints` for full documentation.
        """
        return ti.preprocessing.merge_staypoints(self, triplegs, max_time_gap=max_time_gap, agg=agg)

    def create_activity_flag(self, method="time_threshold", time_threshold=15.0, activity_column_name="is_activity"):
        """
        Add a flag whether or not a staypoint is considered an activity based on a time threshold.

        See :func:`trackintel.analysis.create_activity_flag` for full documentation.
        """
        return ti.analysis.create_activity_flag(
            self, method=method, time_threshold=time_threshold, activity_column_name=activity_column_name
        )

    def spatial_filter(self, areas, method="within", re_project=False):
        """
        Filter Staypoints on a geo extent.

        See :func:`trackintel.preprocessing.spatial_filter` for full documentation.
        """
        return ti.preprocessing.spatial_filter(self, areas, method=method, re_project=re_project)

    @doc(_shared_docs["write_csv"], first_arg="", long="staypoints", short="sp")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.write_staypoints_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="staypoints", short="sp")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.write_staypoints_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def temporal_tracking_quality(self, granularity="all"):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.temporal_tracking_quality` for full documentation.
        """
        return ti.analysis.temporal_tracking_quality(self, granularity=granularity)

    def generate_trips(self, triplegs, gap_threshold=15, add_geometry=True):
        """
        Generate trips based on staypoints and triplegs.

        See :func:`trackintel.preprocessing.generate_trips` for full documentation.
        """
        return ti.preprocessing.generate_trips(self, triplegs, gap_threshold=gap_threshold, add_geometry=add_geometry)

    def radius_gyration(self, method="count", print_progress=False):
        """
        Calculate radius for gyration for Staypoints

        See :func:`trackintel.analysis.radius_gyration` for full documentation.
        """
        return ti.analysis.radius_gyration(self, method, print_progress)

    def jump_length(self):
        """
        Calculate jump length per user between consecutive staypoints.

        See :func:`trackintel.analysis.jump_length` for full documentation.
        """
        return ti.analysis.jump_length(self)
