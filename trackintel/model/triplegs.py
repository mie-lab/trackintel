import pandas as pd
import trackintel as ti

from trackintel.analysis.labelling import predict_transport_mode
from trackintel.analysis.modal_split import calculate_modal_split
from trackintel.analysis.tracking_quality import temporal_tracking_quality
from trackintel.geogr.distances import calculate_distance_matrix
from trackintel.io.file import write_triplegs_csv
from trackintel.io.postgis import write_triplegs_postgis
from trackintel.model.util import copy_docstring
from trackintel.preprocessing.filter import spatial_filter
from trackintel.preprocessing.triplegs import generate_trips
from trackintel.visualization.triplegs import plot_triplegs


@pd.api.extensions.register_dataframe_accessor("as_triplegs")
class TriplegsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of `Tripleg`.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at']

    Requires valid line geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `triplegs`

    For several usecases, the following additional columns are required:
    ['mode', 'trip_id']

    Notes
    -----
    A `Tripleg` (also called `stage`) is defined as continuous movement without changing the mode of transport.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_triplegs.plot()
    """

    required_columns = ["user_id", "started_at", "finished_at"]

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert obj.shape[0] > 0, "Geodataframe is empty with shape: {}".format(obj.shape)
        # check columns
        if any([c not in obj.columns for c in TriplegsAccessor.required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of triplegs, "
                + "it must have the properties [%s], but it has [%s]."
                % (", ".join(TriplegsAccessor.required_columns), ", ".join(obj.columns))
            )
        # check geometry
        assert obj.geometry.is_valid.all(), (
            "Not all geometries are valid. Try x[~ x.geometry.is_valid] " "where x is you GeoDataFrame"
        )
        if obj.geometry.iloc[0].geom_type != "LineString":
            raise AttributeError("The geometry must be a LineString (only first checked).")

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(
            obj["started_at"]
        ), "dtype of started_at is {} but has to be datetime64 and timezone aware".format(obj["started_at"].dtype)
        assert pd.api.types.is_datetime64tz_dtype(
            obj["finished_at"]
        ), "dtype of finished_at is {} but has to be datetime64 and timezone aware".format(obj["finished_at"].dtype)

    @copy_docstring(plot_triplegs)
    def plot(self, *args, **kwargs):
        """
        Plot this collection of triplegs.

        See :func:`trackintel.visualization.triplegs.plot_triplegs`.
        """
        ti.visualization.triplegs.plot_triplegs(self._obj, *args, **kwargs)

    @copy_docstring(write_triplegs_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of triplegs as a CSV file.

        See :func:`trackintel.io.file.write_triplegs_csv`.
        """
        ti.io.file.write_triplegs_csv(self._obj, filename, *args, **kwargs)

    @copy_docstring(write_triplegs_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of triplegs to PostGIS.

        See :func:`trackintel.io.postgis.store_positionfixes_postgis`.
        """
        ti.io.postgis.write_triplegs_postgis(
            self._obj, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

    @copy_docstring(calculate_distance_matrix)
    def calculate_distance_matrix(self, *args, **kwargs):
        """
        Calculate pair-wise distance among triplegs or to other triplegs.

        See :func:`trackintel.geogr.distances.calculate_distance_matrix`.
        """
        return ti.geogr.distances.calculate_distance_matrix(self._obj, *args, **kwargs)

    @copy_docstring(spatial_filter)
    def spatial_filter(self, *args, **kwargs):
        """
        Filter triplegs with a geo extent.

        See :func:`trackintel.preprocessing.filter.spatial_filter`.
        """
        return ti.preprocessing.filter.spatial_filter(self._obj, *args, **kwargs)

    @copy_docstring(generate_trips)
    def generate_trips(self, *args, **kwargs):
        """
        Generate trips based on staypoints and triplegs.

        See :func:`trackintel.preprocessing.triplegs.generate_trips`.
        """
        # if spts in kwargs: 'spts' can not be in args as it would be the first argument
        if "spts" in kwargs:
            return ti.preprocessing.triplegs.generate_trips(tpls=self._obj, **kwargs)
        # if 'spts' no in kwargs it has to be the first argument in 'args'
        else:
            assert len(args) <= 1, (
                "All arguments except 'stps_input' have to be given as keyword arguments. You gave"
                f" {args[1:]} as positional arguments."
            )
            return ti.preprocessing.triplegs.generate_trips(spts=args[0], tpls=self._obj, **kwargs)

    @copy_docstring(predict_transport_mode)
    def predict_transport_mode(self, *args, **kwargs):
        """
        Predict/impute the transport mode with which each tripleg was likely covered.

        See :func:`trackintel.analysis.labelling.predict_transport_mode`.
        """
        return ti.analysis.labelling.predict_transport_mode(self._obj, *args, **kwargs)

    @copy_docstring(calculate_modal_split)
    def calculate_modal_split(self, *args, **kwargs):
        """
        Calculate the modal split of the triplegs.

        See :func:`trackintel.analysis.modal_split.calculate_modal_split`.
        """
        return ti.analysis.modal_split.calculate_modal_split(self._obj, *args, **kwargs)

    @copy_docstring(temporal_tracking_quality)
    def temporal_tracking_quality(self, *args, **kwargs):
        """
        Calculate per-user temporal tracking quality (temporal coverage).

        See :func:`trackintel.analysis.tracking_quality.temporal_tracking_quality`.
        """
        return ti.analysis.tracking_quality.temporal_tracking_quality(self._obj, *args, **kwargs)
