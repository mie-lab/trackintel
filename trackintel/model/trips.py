import geopandas as gpd
import pandas as pd

import trackintel as ti
from trackintel.analysis.tracking_quality import temporal_tracking_quality
from trackintel.io.file import write_trips_csv
from trackintel.io.postgis import write_trips_postgis
from trackintel.model.util import (
    _copy_docstring,
    _register_trackintel_accessor,
    TrackintelBase,
    TrackintelDataFrame,
    TrackintelGeoDataFrameWithFallback,
)


@_register_trackintel_accessor("as_trips")
def trips(*args, **kwargs):
    """A pandas accessor to treat (Geo)DataFrames as collections of trips.

    This function will create a TrackintelDataFrame or a TrackintelGeoDataFrame depending if a geometry column is present.
    It can be used like a typical GeoDataFrame/DataFrame constructor.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the (Geo)DataFrame will be treated as unique identifier of the `Trips`

    Trips have an optional geometry of type MultiPoint which describes the start and the end point of the trip

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """
    is_gdf = len(args) > 0 and isinstance(args[0], gpd.GeoDataFrame) or "geometry" in kwargs
    if is_gdf:
        return TripsGeoDataFrame(*args, **kwargs)
    return TripsDataFrame(*args, **kwargs)


_required_columns = ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]


class TripsDataFrame(TrackintelBase, TrackintelDataFrame):
    """Class to treat a DataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the DataFrame will be treated as unique identifier of the `Trips`

    Trips have an optional geometry of type MultiPoint which describes the start and the end point of the trip

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        TripsDataFrame._validate(self)  # static call

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of trips, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(
            obj["started_at"]
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert pd.api.types.is_datetime64tz_dtype(
            obj["finished_at"]
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

    @staticmethod
    def _check(obj):
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if not pd.api.types.is_datetime64tz_dtype(obj["started_at"]):
            return False
        if not pd.api.types.is_datetime64tz_dtype(obj["finished_at"]):
            return False
        return True

    @_copy_docstring(write_trips_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of trips as a CSV file.

        See :func:`trackintel.io.file.write_trips_csv`.
        """
        ti.io.file.write_trips_csv(self, filename, *args, **kwargs)

    @_copy_docstring(write_trips_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of trips to PostGIS.

        See :func:`trackintel.io.postgis.write_trips_postgis`.
        """
        ti.io.postgis.write_trips_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    @_copy_docstring(temporal_tracking_quality)
    def temporal_tracking_quality(self, *args, **kwargs):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.tracking_quality.temporal_tracking_quality`.
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self, *args, **kwargs)

    def generate_tours(self, **kwargs):
        """
        Generate tours based on trips (and optionally staypoint locations).

        See :func:`trackintel.preprocessing.trips.generate_tours`.
        """
        return ti.preprocessing.trips.generate_tours(trips=self, **kwargs)


# added GeoDataFrame and DataFrame manually afterwards such that our methods always come first
class TripsGeoDataFrame(TrackintelGeoDataFrameWithFallback, TripsDataFrame, gpd.GeoDataFrame, pd.DataFrame):
    """Class to treat a GeoDataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the GeoDataFrame will be treated as unique identifier of the `Trips`

    Trips have an optional geometry of type MultiPoint which describes the start and the end point of the trip

    For several usecases, the following additional columns are required:
    ['origin_purpose', 'destination_purpose', 'modes', 'primary_mode', 'tour_id']

    Notes
    -----
    `Trips` are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities. The following assumptions are implemented
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
          activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination staypoint id.
        - If the origin (or destination) staypoint is unknown (and a geometry column exists), the origin/destination
          geometry is set as the first coordinate of the first tripleg (or the last coordinate of the last tripleg)
        - There are no trips without a (recorded) tripleg.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.generate_tours()
    """
    fallback_class = TripsDataFrame

    def __init__(self, *args, validate_geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        TripsGeoDataFrame._validate(self, validate_geometry=validate_geometry)

    @staticmethod
    def _validate(self, validate_geometry=True):
        if not validate_geometry:
            return
        # _validate should not be called from the outside -> we can use fact that it is called only from __init__
        # therefore TrackintelDataFrame validated all the columns and here we only have to validate the geometry
        assert (
            self.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~x.geometry.is_valid] where x is you GeoDataFrame"
        if self.geometry.iloc[0].geom_type != "MultiPoint":
            raise ValueError("The geometry must be a MultiPoint (only first checked).")

    @staticmethod
    def _check(self, validate_geometry=True):
        val = TripsDataFrame._check(self)
        if not validate_geometry:
            return val
        return val and self.geometry.is_valid.all() and self.geometry.iloc[0].geom_type == "MultiPoint"
