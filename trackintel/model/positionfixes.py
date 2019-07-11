import pandas as pd
import trackintel as ti

import trackintel.preprocessing.positionfixes
import trackintel.visualization.positionfixes
import trackintel.io.postgis
import trackintel.io.file


@pd.api.extensions.register_dataframe_accessor("as_positionfixes")
class PositionfixesAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of positionfixes. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'tracked_at', 'elevation', 'accuracy', 'geom']``

    Examples
    --------
    >>> df.as_positionfixes.extract_staypoints()
    """

    required_columns = ['user_id', 'tracked_at', 'elevation', 'accuracy', 'geom']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in PositionfixesAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of positionfixes, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(PositionfixesAccessor.required_columns), ', '.join(obj.columns)))
        if obj.shape[0] > 0 and obj.geometry.iloc[0].geom_type is not 'Point':
            raise AttributeError("The geometry must be a Point (only first checked).")

    @property
    def center(self):
        """Returns the center coordinate of this collection of positionfixes."""
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

    def extract_staypoints(self, *args, **kwargs):
        """Extracts staypoints from this collection of positionfixes. 
        See :func:`trackintel.preprocessing.positionfixes.extract_staypoints`."""
        return ti.preprocessing.positionfixes.extract_staypoints(self._obj, *args, **kwargs)

    def extract_triplegs(self, *args, **kwargs):
        """Extracts triplegs from this collection of positionfixes."""
        raise NotImplementedError

    def extract_staypoints_and_triplegs(self, *args, **kwargs):
        """Extracts staypoints, uses them to build triplegs, and builds all associations 
        with the original positionfixes (i.e., returning everything in accordance with the trackintel
        :doc:`/content/data_model_sql`).

        Returns
        -------
        tuple
            A tuple consisting of (positionfixes, staypoints, triplegs).
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        """Plots this collection of positionfixes. 
        See :func:`trackintel.visualization.positionfixes.plot_positionfixes`."""
        ti.visualization.positionfixes.plot_positionfixes(self._obj, *args, **kwargs)

    def to_csv(self, filename, *args, **kwargs):
        """Stores this collection of trackpoints as a CSV file.
        See :func:`trackintel.io.file.write_positionfixes_csv`."""
        ti.io.file.write_positionfixes_csv(self._obj, filename, *args, **kwargs)

    def to_postgis(self, conn_string, table_name):
        """Stores this collection of positionfixes to PostGIS.
        See :func:`trackintel.io.postgis.write_positionfixes_postgis`."""
        ti.io.postgis.write_positionfixes_postgis(self._obj, conn_string, table_name)