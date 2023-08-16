import trackintel as ti
from trackintel.io.file import write_locations_csv
from trackintel.io.postgis import write_locations_postgis
from trackintel.model.util import (
    TrackintelBase,
    TrackintelGeoDataFrame,
    _copy_docstring,
    _register_trackintel_accessor,
)
from trackintel.preprocessing.filter import spatial_filter
from trackintel.visualization.locations import plot_locations

_required_columns = ["user_id", "center"]


@_register_trackintel_accessor("as_locations")
class Locations(TrackintelBase, TrackintelGeoDataFrame):
    """A pandas accessor to treat a GeoDataFrames as a collections of locations.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'center']

    For several usecases, the following additional columns are required:
    ['elevation', 'extent']

    Notes
    -----
    `Locations` are spatially aggregated `Staypoints` where a user frequently visits.

    Examples
    --------
    >>> df.as_locations.plot()
    """

    def __init__(self, *args, validate_geometry=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate(self, validate_geometry=validate_geometry)

    @property
    def as_locations(self):
        return self

    @staticmethod
    def _validate(obj, validate_geometry):
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of locations, it must have the properties"
                f" {_required_columns}, but it has [{', '.join(obj.columns)}]."
            )
        if obj.shape[0] <= 0:
            raise ValueError(f"GeoDataFrame is empty with shape: {obj.shape}")

        if validate_geometry and obj["center"].iloc[0].geom_type != "Point":
            # todo: We could think about allowing both geometry types for locations (point and polygon)
            # One for extend and one for the center
            raise ValueError("The center geometry must be a Point (only first checked).")

    @staticmethod
    def _check(obj, validate_geometry=True):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if obj.shape[0] <= 0:
            return False
        if validate_geometry:
            return obj.geometry.iloc[0].geom_type == "Point"
        return True

    @_copy_docstring(plot_locations)
    def plot(self, *args, **kwargs):
        """
        Plot this collection of locations.

        See :func:`trackintel.visualization.locations.plot_locations`.
        """
        ti.visualization.locations.plot_locations(self, *args, **kwargs)

    @_copy_docstring(write_locations_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of locations as a CSV file.

        See :func:`trackintel.io.file.write_locations_csv`.
        """
        ti.io.file.write_locations_csv(self, filename, *args, **kwargs)

    @_copy_docstring(write_locations_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of locations to PostGIS.

        See :func:`trackintel.io.postgis.write_locations_postgis`.
        """
        ti.io.postgis.write_locations_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)

    @_copy_docstring(spatial_filter)
    def spatial_filter(self, *args, **kwargs):
        """
        Filter locations with a geo extent.

        See :func:`trackintel.preprocessing.filter.spatial_filter`.
        """
        return ti.preprocessing.filter.spatial_filter(self, *args, **kwargs)
