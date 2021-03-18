import pandas as pd

import trackintel as ti
import trackintel.preprocessing.filter
import trackintel.visualization.triplegs


@pd.api.extensions.register_dataframe_accessor("as_triplegs")
class TriplegsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of triplegs. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at']``

    Requires valid ``line geometries``; the ``index`` of the GeoDataFrame will be treated as unique identifier
    of the `triplegs`

    For several usecases, the following additional columns are required:
    ``['mode', 'trip_id']``

    Notes
    -------
    A `tripleg` (also called `stage`) is defined as continuous movement without changing the mode of transport.

    ``started_at`` and ``finished_at`` are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_triplegs.plot()
    """

    required_columns = ['user_id', 'started_at', 'finished_at']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert obj.shape[0] > 0, "Geodataframe is empty with shape: {}".format(obj.shape)
        # check columns
        if any([c not in obj.columns for c in TriplegsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of triplegs, " \
                                 + "it must have the properties [%s], but it has [%s]." \
                                 % (', '.join(TriplegsAccessor.required_columns), ', '.join(obj.columns)))
        # check geometry
        assert obj.geometry.is_valid.all(), "Not all geometries are valid. Try x[~ x.geometry.is_valid] " \
                                            "where x is you GeoDataFrame"
        if obj.geometry.iloc[0].geom_type != 'LineString':
            raise AttributeError("The geometry must be a LineString (only first checked).")

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(obj['started_at']), \
            "dtype of started_at is {} but has to be datetime64 and timezone aware".format(obj['started_at'].dtype)
        assert pd.api.types.is_datetime64tz_dtype(obj['finished_at']), \
            "dtype of finished_at is {} but has to be datetime64 and timezone aware".format(obj['finished_at'].dtype)

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

    def similarity(self, *args, **kwargs):
        """Calculate pair-wise distance among triplegs (x) or to other triplegs (y)
        See :func:`trackintel.geogr.distances.calculate_distance_matrix`.
        """
        return ti.geogr.distances.calculate_distance_matrix(self._obj, *args, **kwargs)

    def spatial_filter(self, *args, **kwargs):
        """Filter triplegs with a geo extent.
        See :func:`trackintel.preprocessing.filter.spatial_filter`."""
        return ti.preprocessing.filter.spatial_filter(self._obj, *args, **kwargs)

    def predict_transport_mode(self, *args, **kwargs):
        """Predict/impute the transport mode with which each tripleg was likely covered.
        See :func:`trackintel.analysis.transport_mode_identification.predict_transport_mode`.
        """
        return ti.analysis.transport_mode_identification.predict_transport_mode(self._obj, *args, **kwargs)

    def calculate_modal_split(self, *args, **kwargs):
        """Calculates the modal split of the triplegs.
        See :func:`trackintel.analysis.modal_split.calculate_modal_split`.
        """
        return ti.analysis.modal_split.calculate_modal_split(self._obj, *args, **kwargs)
