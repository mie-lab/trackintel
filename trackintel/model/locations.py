import pandas as pd
import trackintel as ti
import trackintel.io
from trackintel.io.file import write_locations_csv
from trackintel.io.postgis import write_locations_postgis
from trackintel.model.util import _copy_docstring
from trackintel.preprocessing.filter import spatial_filter
from trackintel.visualization.locations import plot_locations


@pd.api.extensions.register_dataframe_accessor("as_locations")
class LocationsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of locations.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['user_id', 'center']

    For several usecases, the following additional columns are required:
    ['elevation', 'context', 'extent']

    Notes
    -----
    `Locations` are spatially aggregated `Staypoints` where a user frequently visits.

    Examples
    --------
    >>> df.as_locations.plot()
    """

    required_columns = ["user_id", "center"]

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in LocationsAccessor.required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of locations, "
                + "it must have the properties [%s], but it has [%s]."
                % (", ".join(LocationsAccessor.required_columns), ", ".join(obj.columns))
            )

        if not (obj.shape[0] > 0 and obj["center"].iloc[0].geom_type == "Point"):
            # todo: We could think about allowing both geometry types for locations (point and polygon)
            # One for extend and one for the center
            raise AttributeError("The center geometry must be a Point (only first checked).")

    @_copy_docstring(plot_locations)
    def plot(self, *args, **kwargs):
        """
        Plot this collection of locations.

        See :func:`trackintel.visualization.locations.plot_locations`.
        """
        ti.visualization.locations.plot_locations(self._obj, *args, **kwargs)

    @_copy_docstring(write_locations_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of locations as a CSV file.

        See :func:`trackintel.io.file.write_locations_csv`.
        """
        ti.io.file.write_locations_csv(self._obj, filename, *args, **kwargs)

    @_copy_docstring(write_locations_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of locations to PostGIS.

        See :func:`trackintel.io.postgis.write_locations_postgis`.
        """
        ti.io.postgis.write_locations_postgis(
            self._obj, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

    @_copy_docstring(spatial_filter)
    def spatial_filter(self, *args, **kwargs):
        """
        Filter locations with a geo extent.

        See :func:`trackintel.preprocessing.filter.spatial_filter`.
        """
        return ti.preprocessing.filter.spatial_filter(self._obj, *args, **kwargs)
