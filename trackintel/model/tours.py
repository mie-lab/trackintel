import pandas as pd


@pd.api.extensions.register_dataframe_accessor("as_tours")
class ToursAccessor(object):
    """A pandas accessor to treat DataFrames as collections of `Tours`.

    Requires at least the following columns:
    ['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'journey']

    The 'index' of the GeoDataFrame will be treated as unique identifier of the `Tours`

    For several usecases, the following additional columns are required:
    ['context']

    Notes
    -----
    Tours are an aggregation level in transport planning that summarize all trips until a person returns to the
    same location. Tours starting and ending at home (=journey) are especially important.

    'started_at' and 'finished_at' are timezone aware pandas datetime objects.

    Examples
    --------
    >>> df.as_tours.plot()
    """

    required_columns = ["user_id", "started_at", "finished_at", "origin_destination_location_id", "journey"]

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in ToursAccessor.required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of tours, "
                + "it must have the properties [%s], but it has [%s]."
                % (", ".join(ToursAccessor.required_columns), ", ".join(obj.columns))
            )

        # check timestamp dtypes
        assert pd.api.types.is_datetime64tz_dtype(
            obj["started_at"]
        ), "dtype of started_at is {} but has to be datetime64 and timezone aware".format(obj["started_at"].dtype)
        assert pd.api.types.is_datetime64tz_dtype(
            obj["finished_at"]
        ), "dtype of finished_at is {} but has to be datetime64 and timezone aware".format(obj["finished_at"].dtype)

    def to_csv(self, filename, *args, **kwargs):
        """
        Store this collection of tours as a CSV file.

        See :func:`trackintel.io.file.write_tours_csv`.
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        """
        Plot this collection of tours.

        See :func:`trackintel.visualization.tours.plot_tours`.
        """
        raise NotImplementedError
