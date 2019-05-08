import pandas as pd
import trackintel as ti

import trackintel.visualization.staypoints
import trackintel.preprocessing.staypoints
import trackintel.io.postgis
import trackintel.io.file


@pd.api.extensions.register_dataframe_accessor("as_staypoints")
class StaypointsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of staypoints. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at', 'elevation', 'geom']``

    Examples
    --------
    >>> df.as_staypoints.extract_places()
    """

    required_columns = ['user_id', 'started_at', 'finished_at', 'elevation', 'geom']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in StaypointsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of staypoints, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(StaypointsAccessor.required_columns), ', '.join(obj.columns)))
        if not (obj.shape[0] > 0 and any([row.geom_type[0] is not 'Point' for _, row in obj['geom'].iteritems()])):
            print(obj.shape[0] > 0, any( [row.geom_type[0] is not 'Point' for _, row in obj['geom'].iteritems()]))
            raise AttributeError("The geometry must be a Point (only first checked).")

    @property
    def center(self):
        """Returns the center coordinate of this collection of staypoints."""
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

    def extract_places(self, *args, **kwargs):
        """Extracts places from this collection of staypoints.
        See :func:`trackintel.preprocessing.staypoints.cluster_staypoints`."""
        return ti.preprocessing.staypoints.cluster_staypoints(self._obj, *args, **kwargs)

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