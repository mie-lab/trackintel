import pandas as pd
import trackintel as ti

import trackintel.visualization.staypoints


@pd.api.extensions.register_dataframe_accessor("as_staypoints")
class PositionfixesAccessor(object):
    """A pandas accessor to treat (Geo)DataFrames as collections of staypoints. This
    will define certain methods and accessors, as well as make sure that the DataFrame
    adheres to some requirements."""

    required_columns = ['user_id', 'started_at', 'finished_at', 
                        'longitude', 'latitude', 'elevation']

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if any([c not in obj.columns for c in PositionfixesAccessor.required_columns]):
            raise AttributeError("To process a DataFrame as a collection of staypoints, " \
                + "it must have the properties [%s], but it has [%s]." \
                % (', '.join(PositionfixesAccessor.required_columns), ', '.join(obj.columns)))

    @property
    def center(self):
        """Returns the center coordinate of this collection of staypoints."""
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

    def plot(self, *args, **kwargs):
        """Plots this collection of staypoints. 
        See :func:`trackintel.visualization.staypoints.plot_staypoints`."""
        ti.visualization.staypoints.plot_staypoints(self._obj, *args, **kwargs)