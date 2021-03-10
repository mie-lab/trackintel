import pandas as pd

import trackintel as ti


@pd.api.extensions.register_dataframe_accessor("as_staypoints")
class StaypointsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of staypoints. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at']``

    Requires valid ``point geometries``; the ``index`` of the GeoDataFrame will be treated as unique identifier
    of the `Staypoints`

    For several usecases, the following additional columns are required:
    ``['elevation', 'radius', 'context', 'purpose', 'activity', 'next_trip_id', 'prev_trip_id', 'trip_id',
    location_id]``

    Notes
    -------
    Staypoints are defined as location were a person did not move for a while. Under consideration of location
    uncertainty this means that a person stays within a certain radius for a certain amount of time.
    The exact definition is use-case dependent.

    ``started_at`` and ``finished_at`` are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_staypoints.generate_locations()
    """

    required_columns = ['user_id', 'started_at', 'finished_at']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # check columns
        if any([c not in obj.columns for c in StaypointsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of staypoints, " \
                                 + "it must have the properties [%s], but it has [%s]." \
                                 % (', '.join(StaypointsAccessor.required_columns), ', '.join(obj.columns)))
        # check geometry
        assert obj.geometry.is_valid.all(), "Not all geometries are valid. Try x[~ x.geometry.is_valid] " \
                                            "where x is you GeoDataFrame"
        if obj.geometry.iloc[0].geom_type != 'Point':
            raise AttributeError("The geometry must be a Point (only first checked).")

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(obj['started_at']), \
            "dtype of started_at is {} but has to be tz aware datetime64".format(obj['started_at'].dtype)
        assert pd.api.types.is_datetime64tz_dtype(obj['finished_at']), \
            "dtype of finished_at is {} but has to be tz aware datetime64".format(obj['finished_at'].dtype)

    @property
    def center(self):
        """Returns the center coordinate of this collection of staypoints."""
        lat = self._obj.geometry.y
        lon = self._obj.geometry.x
        return (float(lon.mean()), float(lat.mean()))

    def generate_locations(self, *args, **kwargs):
        """Generate locations from this collection of staypoints.
        See :func:`trackintel.preprocessing.staypoints.generate_locations`."""
        return ti.preprocessing.staypoints.generate_locations(self._obj, *args, **kwargs)

    def create_activity_flag(self, *args, **kwargs):
        """Sets a flag if a staypoint is also an activity.
        See :func:`trackintel.preprocessing.staypoints.create_activity_flag`."""
        return ti.preprocessing.staypoints.create_activity_flag(self._obj, *args, **kwargs)

    def spatial_filter(self, *args, **kwargs):
        """Filter staypoints with a geo extent.
        See :func:`trackintel.preprocessing.filter.spatial_filter`."""
        return ti.preprocessing.filter.spatial_filter(self._obj, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """Plots this collection of staypoints. 
        See :func:`trackintel.visualization.staypoints.plot_staypoints`."""
        ti.visualization.staypoints.plot_staypoints(self._obj, *args, **kwargs)

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of staypoints as a CSV file.
        See :func:`trackintel.io.file.write_staypoints_csv`."""
        ti.io.file.write_staypoints_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name):
        """Stores this collection of staypoints to PostGIS.
        See :func:`trackintel.io.postgis.write_staypoints_postgis`."""
        ti.io.postgis.write_staypoints_postgis(self._obj, conn_string, table_name)
