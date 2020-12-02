import pandas as pd

import trackintel as ti
import trackintel.preprocessing.filter
import trackintel.visualization.locations
import trackintel.visualization.staypoints


@pd.api.extensions.register_dataframe_accessor("as_locations")
class LocationsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of locations. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'center']``

    For several usecases, the following additional columns are required:
    ``['elevation', 'context', 'extent']``

    Examples
    --------
    >>> df.as_locations.plot()
    """

    required_columns = ['user_id', 'center']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in LocationsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of locations, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(LocationsAccessor.required_columns), ', '.join(obj.columns)))

        if not (obj.shape[0] > 0 and obj['center'].iloc[0].geom_type == 'Point'):
            # todo: We could think about allowing both geometry types for locations (point and polygon)
            # One for extend and one for the center
            raise AttributeError("The center geometry must be a Point (only first checked).")

    def plot(self, *args, **kwargs):
        """Plots this collection of locations. 
        See :func:`trackintel.visualization.locations.plot_locations`."""
        ti.visualization.locations.plot_center_of_locations(self._obj, *args, **kwargs)

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of locations as a CSV file.
        See :func:`trackintel.io.file.write_locations_csv`."""
        ti.io.file.write_locations_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name, schema=None,
            sql_chunksize=None, if_exists='replace'):
        """Stores this collection of locations to PostGIS.
        See :func:`trackintel.io.postgis.write_locations_postgis`."""
        ti.io.postgis.write_locations_postgis(self._obj, conn_string, table_name, 
            schema, sql_chunksize, if_exists)
        
    def spatial_filter(self, *args, **kwargs):
        """Filter locations with a geo extent.
        See :func:`trackintel.preprocessing.filter.spatial_filter`."""
        return ti.preprocessing.filter.spatial_filter(self._obj, *args, **kwargs)