import geopandas as gpd
import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelDataFrame,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)


@_register_trackintel_accessor("as_trips")
class Trips:
    """Trackintel class to treat (Geo)DataFrames as collections of trips.

    The class constructor will create a TripsDataFrame or a TripsGeoDataFrame depending if a geometry column is present.

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
    >>> trips.generate_tours()
    """

    def __new__(cls, *args, **kwargs):
        is_gdf = (
            (len(args) > 0 and isinstance(args[0], gpd.GeoDataFrame))
            or "geometry" in kwargs
            or ("data" in kwargs and isinstance(kwargs["data"], gpd.GeoDataFrame))
        )
        if is_gdf:
            return TripsGeoDataFrame(*args, **kwargs)
        return TripsDataFrame(*args, **kwargs)


_required_columns = ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]


class TripsDataFrame(TrackintelBase, TrackintelDataFrame):
    """Class to treat a DataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the DataFrame will be treated as unique identifier of the `Trips`

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
    >>> trips.generate_tours()
    """

    def __init__(self, *args, validate=True, **kwargs):
        super().__init__(*args, **kwargs)
        if validate:
            TripsDataFrame.validate(self)  # static call

    @staticmethod
    def validate(obj):
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of trips, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )

        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

    @doc(_shared_docs["write_csv"], first_arg="", long="trips", short="trips")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.write_trips_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="trips", short="trips")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.write_trips_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    def temporal_tracking_quality(self, granularity="all"):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.temporal_tracking_quality` for full documentation.
        """
        return ti.analysis.temporal_tracking_quality(self, granularity=granularity)

    def generate_tours(self, **kwargs):
        """
        Generate trackintel-tours from trips

        See :func:`trackintel.preprocessing.generate_tours` for full documentation.
        """
        return ti.preprocessing.generate_tours(trips=self, **kwargs)


# added GeoDataFrame manually afterwards such that our methods always come first
class TripsGeoDataFrame(TrackintelGeoDataFrame, TripsDataFrame, gpd.GeoDataFrame):
    """Class to treat a GeoDataFrame as collections of trips.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the GeoDataFrame will be treated as unique identifier of the `Trips`

    TripsGeoDataFrame must have a geometry of type MultiPoint which describes the start and the end point of the trip.

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
    >>> trips.generate_tours()
    """

    fallback_class = TripsDataFrame

    def __init__(self, *args, validate=True, **kwargs):
        super().__init__(*args, validate=validate, **kwargs)
        if validate:
            TripsGeoDataFrame.validate(self)

    @staticmethod
    def validate(self):
        TripsDataFrame.validate(self)
        assert (
            self.geometry.is_valid.all()
        ), "Not all geometries are valid. Try x[~x.geometry.is_valid] where x is you GeoDataFrame"
        if self.geometry.iloc[0].geom_type != "MultiPoint":
            raise ValueError("The geometry must be a MultiPoint (only first checked).")
