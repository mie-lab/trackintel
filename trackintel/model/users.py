import pandas as pd
import trackintel as ti


@pd.api.extensions.register_dataframe_accessor("as_users")
class UsersAccessor(object):
    """A pandas accessor to treat DataFrames as collections of users.

    This will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements.

    Requires at least the following columns:
    ['id']

    For several usecases, the following additional columns are required:
    ['attributes', 'geom_home', 'geom_work']

    Examples
    --------
    >>> df.as_users.plot_home_and_work()
    """

    required_columns = ["id"]

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in UsersAccessor.required_columns]):
            raise AttributeError(
                "To process a DataFrame as a collection of users, "
                + "it must have the properties [%s], but it has [%s]."
                % (", ".join(UsersAccessor.required_columns), ", ".join(obj.columns))
            )

    def plot_home_and_work(self):
        """
        Plot home and work locations of users.

        See :func:`trackintel.visualization.users.plot_home_and_work`.
        """
        raise NotImplementedError
