import pandas as pd

import trackintel as ti
from trackintel.io.file import write_tours_csv
from trackintel.io.postgis import write_tours_postgis
from trackintel.model.util import _register_trackintel_accessor, TrackintelBase, TrackintelDataFrame, _copy_docstring

_required_columns = ["user_id", "started_at", "finished_at"]


@_register_trackintel_accessor("as_tours")
class Tours(TrackintelBase, TrackintelDataFrame):
    """A pandas accessor to treat DataFrames as collections of `Tours`.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at']

    The 'index' of the DataFrame will be treated as unique identifier of the `Tours`

    For several usecases, the following additional columns are required:
    ['location_id', 'journey', 'origin_staypoint_id', 'destination_staypoint_id']

    Notes
    -----
    Tours are an aggregation level in transport planning that summarize all trips until a person returns to the
    same location. Tours starting and ending at home (=journey) are especially important.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_tours.to_csv("filename.csv")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate(self)

    # createte circular reference directly -> avoid second call of init via accessor
    @property
    def as_tours(self):
        return self

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in _required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of tours, it must have the properties"
                f" {_required_columns}, but it has {', '.join(obj.columns)}."
            )

        # check timestamp dtypes
        assert isinstance(
            obj["started_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of started_at is {obj['started_at'].dtype} but has to be datetime64 and timezone aware"
        assert isinstance(
            obj["finished_at"].dtype, pd.DatetimeTZDtype
        ), f"dtype of finished_at is {obj['finished_at'].dtype} but has to be datetime64 and timezone aware"

    @staticmethod
    def _check(obj):
        """Check does the same as _validate but returns bool instead of potentially raising an error."""
        if any([c not in obj.columns for c in _required_columns]):
            return False
        if not isinstance(obj["started_at"].dtype, pd.DatetimeTZDtype):
            return False
        if not isinstance(obj["finished_at"].dtype, pd.DatetimeTZDtype):
            return False
        return True

    @_copy_docstring(write_tours_csv)
    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of tours as a CSV file.

        See :func:`trackintel.io.file.write_tours_csv`.
        """
        ti.io.file.write_tours_csv(self, filename, *args, **kwargs)

    @_copy_docstring(write_tours_postgis)
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        """
        Store this collection of tours to PostGIS.

        See :func:`trackintel.io.postgis.write_tours_postgis`.
        """
        ti.io.postgis.write_tours_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)
