import pandas as pd
import geopandas as gpd
import trackintel as ti
from trackintel.geogr.distances import calculate_distance_matrix
from trackintel.io.file import write_positionfixes_csv
from trackintel.io.postgis import write_positionfixes_postgis
from trackintel.model.util import _copy_docstring
from trackintel.preprocessing.positionfixes import generate_staypoints, generate_triplegs
from trackintel.visualization.positionfixes import plot_positionfixes
from trackintel.model.util import (
    get_speed_positionfixes,
    TrackintelBase,
    TrackintelGeoDataFrame,
    _register_trackintel_accessor,
)

_required_columns = ["user_id", "tracked_at"]


@_register_trackintel_accessor("as_positionfixes")
class Positionfixes(TrackintelBase, TrackintelGeoDataFrame, gpd.GeoDataFrame):
    """A pandas accessor to treat (Geo)DataFrames as collections of `Positionfixes`.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'tracked_at']

    Requires valid point geometries; the 'index' of the GeoDataFrame will be treated as unique identifier
    of the `Positionfixes`.

    For several usecases, the following additional columns are required:
    ['elevation', 'accuracy' 'staypoint_id', 'tripleg_id']

    Notes
    -----
    In GPS based movement data analysis `Positionfixes` are the smallest unit of tracking and
    represent timestamped locations.

    'tracked_at' is a timezone aware pandas datetime object.

    Examples
    --------
    >>> df.as_positionfixes.generate_staypoints()
    """

    def __init__(self, *args, validate_geometry=True, **kwargs):
        # could be moved to super class
        # validate kwarg is necessary as the object is not fully initialised if we call it from _constructor
        # (geometry-link is missing). thus we need a way to stop validating too early.
        super().__init__(*args, **kwargs)
        self._validate(self, validate_geometry=validate_geometry)

    # create circular reference directly -> avoid second call of init via accessor
    @property
    def as_positionfixes(self):
        return self

    @staticmethod
    def _validate(obj, validate_geometry=True):
        assert obj.shape[0] > 0, f"Geodataframe is empty with shape: {obj.shape}"
        # check columns
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of positionfixes, "
                f"it must have the columns {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )
        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(
            obj["tracked_at"]
        ), f"dtype of tracked_at is {obj['tracked_at'].dtype} but has to be datetime64 and timezone aware"

        # check geometry
        if validate_geometry:
            assert obj.geometry.is_valid.all(), (
                "Not all geometries are valid. Try x[~ x.geometry.is_valid] " "where x is you GeoDataFrame"
            )

            if obj.geometry.iloc[0].geom_type != "Point":
                raise AttributeError("The geometry must be a Point (only first checked).")

    @staticmethod
    def _check(obj, validate_geometry=True):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if obj.shape[0] <= 0:
            return False
        if not pd.api.types.is_datetime64tz_dtype(obj["tracked_at"]):
            return False
        if validate_geometry:
            return obj.geometry.is_valid.all() and obj.geometry.iloc[0].geom_type == "Point"
        return True

    @property
    def center(self):
        """Return the center coordinate of this collection of positionfixes."""
        lat = self.geometry.y
        lon = self.geometry.x
        return (float(lon.mean()), float(lat.mean()))

    @_copy_docstring(generate_staypoints)
    def generate_staypoints(self, *args, **kwargs):
        """
        Generate staypoints from this collection of positionfixes.

        See :func:`trackintel.preprocessing.positionfixes.generate_staypoints`.
        """
        return ti.preprocessing.positionfixes.generate_staypoints(self, *args, **kwargs)

    @_copy_docstring(generate_triplegs)
    def generate_triplegs(self, staypoints=None, *args, **kwargs):
        """
        Generate triplegs from this collection of positionfixes.

        See :func:`trackintel.preprocessing.positionfixes.generate_triplegs`.
        """
        return ti.preprocessing.positionfixes.generate_triplegs(self, staypoints, *args, **kwargs)

    @_copy_docstring(plot_positionfixes)
    def plot(self, *args, **kwargs):
        """
        Plot this collection of positionfixes.

        See :func:`trackintel.visualization.positionfixes.plot_positionfixes`.
        """
        ti.visualization.positionfixes.plot_positionfixes(self, *args, **kwargs)

    @_copy_docstring(write_positionfixes_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of trackpoints as a CSV file.

        See :func:`trackintel.io.file.write_positionfixes_csv`.
        """
        ti.io.file.write_positionfixes_csv(self, filename, *args, **kwargs)

    @_copy_docstring(write_positionfixes_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of positionfixes to PostGIS.

        See :func:`trackintel.io.postgis.write_positionfixes_postgis`.
        """
        ti.io.postgis.write_positionfixes_postgis(
            self, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

    @_copy_docstring(calculate_distance_matrix)
    def calculate_distance_matrix(self, *args, **kwargs):
        """
        Calculate pair-wise distance among positionfixes or to other positionfixes.

        See :func:`trackintel.geogr.distances.calculate_distance_matrix`.
        """
        return ti.geogr.distances.calculate_distance_matrix(self, *args, **kwargs)

    @_copy_docstring(get_speed_positionfixes)
    def get_speed(self, *args, **kwargs):
        """
        Compute speed per positionfix (in m/s)

        See :func:`trackintel.model.util.get_speed_positionfixes`.
        """
        return ti.model.util.get_speed_positionfixes(self, *args, **kwargs)
