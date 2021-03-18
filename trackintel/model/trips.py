import pandas as pd

import trackintel as ti


@pd.api.extensions.register_dataframe_accessor("as_trips")
class TripsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of trips.
    
    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']``

    The ``index`` of the GeoDataFrame will be treated as unique identifier of the `trips`

    For several usecases, the following additional columns are required:
    ``['context', 'origin_activity', 'destination_activity', 'modes', 'primary_mode', 'tour_id']``

    Notes
    -----
    Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities.
    The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
    [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
    The following assumptions are implemented
    
        - All movement before the first and after the last activity is omitted
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an \
            activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination
        - There are no trips without a (recored) tripleg.
        
    ``started_at`` and ``finished_at`` are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_trips.plot()
    """

    # ToDo primary mode of transport

    required_columns = ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in TripsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of trips, "
                                 + "it must have the properties [%s], but it has [%s]."
                                 % (', '.join(TripsAccessor.required_columns), ', '.join(obj.columns)))

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(obj['started_at']), \
            "dtype of started_at is {} but has to be datetime64 and timezone aware".format(obj['started_at'].dtype)
        assert pd.api.types.is_datetime64tz_dtype(obj['finished_at']), \
            "dtype of finished_at is {} but has to be datetime64 and timezone aware".format(obj['finished_at'].dtype)

    def plot(self, *args, **kwargs):
        """Plots this collection of trips. 
        See :func:`trackintel.visualization.trips.plot_trips`."""
        raise NotImplementedError

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of trips as a CSV file.
        See :func:`trackintel.io.file.write_trips_csv`."""
        ti.io.file.write_trips_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name, schema=None,
                   sql_chunksize=None, if_exists='replace'):
        """Stores this collection of trips to PostGIS.
        See :func:`trackintel.io.postgis.write_trips_postgis`."""
        ti.io.postgis.write_trips_postgis(self._obj, conn_string, table_name,
                                          schema, sql_chunksize, if_exists)
