from trackintel.analysis.tracking_quality import temporal_tracking_quality
from trackintel.io.postgis import write_trips_postgis
from trackintel.io.file import write_trips_csv
from trackintel.model.util import _copy_docstring
import pandas as pd
import geopandas as gpd

import trackintel as ti


@pd.api.extensions.register_dataframe_accessor("as_trips")
class TripsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of trips.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    The 'index' of the (Geo)DataFrame will be treated as unique identifier of the `Trips`

    Trips have an optional geometry of type MultiPoint which describes the start and the end point of the trip

    For several usecases, the following additional columns are required:
    ['context', 'origin_activity', 'destination_activity', 'modes', 'primary_mode', 'tour_id']

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
    >>> df.as_trips.plot()
    """

    required_columns = ["user_id", "started_at", "finished_at", "origin_staypoint_id", "destination_staypoint_id"]

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in TripsAccessor.required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of trips, "
                + "it must have the properties [%s], but it has [%s]."
                % (", ".join(TripsAccessor.required_columns), ", ".join(obj.columns))
            )

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(
            obj["started_at"]
        ), "dtype of started_at is {} but has to be datetime64 and timezone aware".format(obj["started_at"].dtype)
        assert pd.api.types.is_datetime64tz_dtype(
            obj["finished_at"]
        ), "dtype of finished_at is {} but has to be datetime64 and timezone aware".format(obj["finished_at"].dtype)

        # Check geometry if Trips is a GeoDataFrame
        if isinstance(obj, gpd.GeoDataFrame):
            # check geometry
            assert obj.geometry.is_valid.all(), (
                "Not all geometries are valid. Try x[~ x.geometry.is_valid] " "where x is you GeoDataFrame"
            )
            if obj.geometry.iloc[0].geom_type != "MultiPoint":
                raise AttributeError("The geometry must be a MultiPoint (only first checked).")

    def plot(self, *args, **kwargs):
        """
        Plot this collection of trips.

        See :func:`trackintel.visualization.trips.plot_trips`.
        """
        raise NotImplementedError

    @_copy_docstring(write_trips_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of trips as a CSV file.

        See :func:`trackintel.io.file.write_trips_csv`.
        """
        ti.io.file.write_trips_csv(self._obj, filename, *args, **kwargs)

    @_copy_docstring(write_trips_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of trips to PostGIS.

        See :func:`trackintel.io.postgis.write_trips_postgis`.
        """
        ti.io.postgis.write_trips_postgis(self._obj, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    @_copy_docstring(temporal_tracking_quality)
    def temporal_tracking_quality(self, *args, **kwargs):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.tracking_quality.temporal_tracking_quality`.
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self._obj, *args, **kwargs)

    def generate_tours(self, *args, **kwargs):
        """
        Generate tours based on trips (and optionally staypoint locations).

        See :func:`trackintel.preprocessing.trips.generate_tours`.
        """
        assert len(args) == 0, "When calling 'generate_tours' via the accessor all arguments must be keyword arguments"
        return ti.preprocessing.trips.generate_tours(trips=self._obj, **kwargs)
