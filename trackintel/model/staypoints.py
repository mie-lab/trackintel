import pandas as pd

import trackintel as ti
from trackintel.analysis.labelling import create_activity_flag
from trackintel.analysis.tracking_quality import temporal_tracking_quality
from trackintel.io.file import write_staypoints_csv
from trackintel.io.postgis import write_staypoints_postgis
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _copy_docstring,
    _register_trackintel_accessor,
)
from trackintel.preprocessing.filter import spatial_filter
from trackintel.preprocessing.staypoints import generate_locations, merge_staypoints

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
        assert pd.api.types.is_datetime64tz_dtype(
            obj["started_at"]
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be tz aware datetime64"
        assert pd.api.types.is_datetime64tz_dtype(
            obj["finished_at"]
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
        if not pd.api.types.is_datetime64tz_dtype(obj["started_at"]):
            return False
        if not pd.api.types.is_datetime64tz_dtype(obj["finished_at"]):
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

    @_copy_docstring(generate_locations)
    def generate_locations(self, *args, **kwargs):
        """
        Generate locations from this collection of staypoints.

        See :func:`trackintel.preprocessing.staypoints.generate_locations`.
        """
        return ti.preprocessing.staypoints.generate_locations(self, *args, **kwargs)

    @_copy_docstring(merge_staypoints)
    def merge_staypoints(self, *args, **kwargs):
        """
        Aggregate staypoints horizontally via time threshold.

        See :func:`trackintel.preprocessing.staypoints.merge_staypoints`.
        """
        return ti.preprocessing.staypoints.merge_staypoints(self, *args, **kwargs)

    @_copy_docstring(create_activity_flag)
    def create_activity_flag(self, *args, **kwargs):
        """
        Set a flag if a staypoint is also an activity.

        See :func:`trackintel.analysis.labelling.create_activity_flag`.
        """
        return ti.analysis.labelling.create_activity_flag(self, *args, **kwargs)

    @_copy_docstring(spatial_filter)
    def spatial_filter(self, *args, **kwargs):
        """
        Filter staypoints with a geo extent.

        See :func:`trackintel.preprocessing.filter.spatial_filter`.
        """
        return ti.preprocessing.filter.spatial_filter(self, *args, **kwargs)

    @_copy_docstring(write_staypoints_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of staypoints as a CSV file.

        See :func:`trackintel.io.file.write_staypoints_csv`.
        """
        ti.io.file.write_staypoints_csv(self, filename, *args, **kwargs)

    @_copy_docstring(write_staypoints_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of staypoints to PostGIS.

        See :func:`trackintel.io.postgis.write_staypoints_postgis`.
        """
        ti.io.postgis.write_staypoints_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    @_copy_docstring(temporal_tracking_quality)
    def temporal_tracking_quality(self, *args, **kwargs):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.tracking_quality.temporal_tracking_quality`.
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self, *args, **kwargs)
