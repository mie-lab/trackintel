import pandas as pd
import trackintel as ti

import trackintel.visualization.staypoints
import trackintel.visualization.places


@pd.api.extensions.register_dataframe_accessor("as_places")
class PlacesAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of places. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'center']``

    For several usecases, the following additional columns are required:
    ``['elevation', 'context', 'extent']``

    Examples
    --------
    >>> df.as_places.plot()
    """

    required_columns = ['user_id', 'center']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in PlacesAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of staypoints, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(PlacesAccessor.required_columns), ', '.join(obj.columns)))

        if not (obj.shape[0] > 0 and obj['center'].iloc[0].geom_type is 'Point'):
            # todo: We could think about allowing both geometry types for places (point and polygon)
            # One for extend and one for the center
            raise AttributeError("The center geometry must be a Point (only first checked).")

    def plot(self, *args, **kwargs):
        """Plots this collection of places. 
        See :func:`trackintel.visualization.places.plot_places`."""
        ti.visualization.places.plot_center_of_places(self._obj, *args, **kwargs)

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of places as a CSV file.
        See :func:`trackintel.io.file.write_places_csv`."""
        ti.io.file.write_places_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name, schema=None,
            sql_chunksize=None, if_exists='replace'):
        """Stores this collection of places to PostGIS.
        See :func:`trackintel.io.postgis.write_places_postgis`."""
        ti.io.postgis.write_places_postgis(self._obj, conn_string, table_name, 
            schema, sql_chunksize, if_exists)
