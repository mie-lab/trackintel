import pandas as pd
import trackintel as ti
import shapely


@pd.api.extensions.register_dataframe_accessor("as_triplegs")
class TriplegsAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of triplegs. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements."""

    required_columns = ['user_id', 'started_at', 'finished_at', 'geometry']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in TriplegsAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of triplegs, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(TriplegsAccessor.required_columns), ', '.join(obj.columns)))
        if obj.shape[0] > 0 and obj['geometry'].geom_type[0] is not 'LineString':
            raise AttributeError("The geometry must be a LineString (only first checked).")

    def plot(self, *args, **kwargs):
        """Plots this collection of triplegs."""
        raise NotImplementedError
