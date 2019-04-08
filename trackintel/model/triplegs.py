import pandas as pd
import trackintel as ti
import shapely

import trackintel.visualization.triplegs
import trackintel.io.postgis
import trackintel.io.file


@pd.api.extensions.register_dataframe_accessor("as_triplegs")
class TriplegsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of triplegs. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at', 'geom']``

    Examples
    --------
    >>> df.as_triplegs.plot()
    """

    required_columns = ['user_id', 'started_at', 'finished_at', 'geom']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in TriplegsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of triplegs, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(TriplegsAccessor.required_columns), ', '.join(obj.columns)))
        if obj.shape[0] > 0 and obj['geom'].geom_type[0] is not 'LineString':
            raise AttributeError("The geometry must be a LineString (only first checked).")

    def plot(self, *args, **kwargs):
        """Plots this collection of triplegs. 
        See :func:`trackintel.visualization.triplegs.plot_triplegs`."""
        ti.visualization.triplegs.plot_triplegs(self._obj, *args, **kwargs)

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of triplegs as a CSV file.
        See :func:`trackintel.io.file.write_triplegs_csv`."""
        ti.io.file.write_triplegs_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name):
        """Stores this collection of triplegs to PostGIS.
        See :func:`trackintel.io.postgis.store_positionfixes_postgis`."""
        ti.io.postgis.write_triplegs_postgis(self._obj, conn_string, table_name)