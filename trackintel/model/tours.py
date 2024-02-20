import pandas as pd

import trackintel as ti
from trackintel.model.util import (
    TrackintelBase,
    TrackintelDataFrame,
    _register_trackintel_accessor,
    _shared_docs,
    doc,
)

_required_columns = ["user_id", "started_at", "finished_at"]


@_register_trackintel_accessor("as_tours")
class Tours(TrackintelBase, TrackintelDataFrame):
    """Trackintel class to treat DataFrames as collections of `Tours`.

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
    >>> tours.to_csv("filename.csv")
    """

    def __init__(self, *args, validate=True, **kwargs):
        super().__init__(*args, **kwargs)
        if validate:
            self.validate(self)

    # createte circular reference directly -> avoid second call of init via accessor
    @property
    def as_tours(self):
        return self

    @staticmethod
    def validate(obj):
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

    @doc(_shared_docs["write_csv"], first_arg="", long="tours", short="tours")
    def to_csv(self, filename, *args, **kwargs):
        ti.io.write_tours_csv(self, filename, *args, **kwargs)

    @doc(_shared_docs["write_postgis"], first_arg="", long="tours", short="tours")
    def to_postgis(
        self, name, con, schema=None, if_exists="fail", index=True, index_label=None, chunksize=None, dtype=None
    ):
        ti.io.write_tours_postgis(self, name, con, schema, if_exists, index, index_label, chunksize, dtype)
