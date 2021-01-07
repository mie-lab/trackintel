import pandas as pd


@pd.api.extensions.register_dataframe_accessor("as_tours")
class ToursAccessor(object):
    """A pandas accessor to treat DataFrames as collections of tours. 

    Requires at least the following columns: 
    ``['user_id', 'started_at', 'finished_at', 'origin_destination_location_id', 'journey']``

    The ``index`` of the GeoDataFrame will be treated as unique identifier of the `trips`

    For several usecases, the following additional columns are required:
    ``['context']``

    Notes
    --------
    Tours are an aggregation level in transport planning that summarize all trips until a person returns to the
    same location. Tours starting and ending at home (=journey) are especially important.

    Examples
    --------
    >>> df.as_tours.plot()
    """

    required_columns = ['user_id', 'started_at', 'finished_at', 'origin_destination_location_id', 'journey']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in ToursAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of staypoints, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(ToursAccessor.required_columns), ', '.join(obj.columns)))

    def plot(self, *args, **kwargs):
        """Plots this collection of tours. 
        See :func:`trackintel.visualization.tours.plot_tours`."""
        raise NotImplementedError
