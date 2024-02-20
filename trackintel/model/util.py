import warnings
from functools import wraps, partial
from textwrap import dedent

import pandas as pd
from geopandas import GeoDataFrame


def _wrapped_gdf_method(func):
    """Decorator function that downcast types to trackintel class if is (Geo)DataFrame and has the required columns."""

    @wraps(func)  # copy all metadata
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, pd.DataFrame):
            if isinstance(result, GeoDataFrame):
                result.__class__ = self.__class__
            elif self.fallback_class is not None:
                result.__class__ = self.fallback_class
        return result

    return wrapper


class TrackintelGeoDataFrame(GeoDataFrame):
    """
    Helper class to subtype GeoDataFrame correctly and fallback to custom class.

    Three possible outcomes for the _constructor
    - If GeoDataFrame, keep it a TrackintelGeoDataFrame
    - If DataFrame without fallback class, just return the DataFrame
    - If DataFrame with fallback class, fall back to it.
    """

    # set fallback_class for missing geometry, if None default to pd.DataFrame
    fallback_class = None

    @property
    def _constructor(self):
        """Interface to subtype pandas properly"""
        super_cons = super()._constructor

        def _constructor_with_fallback(*args, **kwargs):
            result = super_cons(*args, **kwargs)
            if isinstance(result, GeoDataFrame):
                return self.__class__(result, validate=False)
            # uses DataFrame constructor -> must be DataFrame
            if self.fallback_class is not None:
                return self.fallback_class(result, validate=False)
            return result

        return _constructor_with_fallback

    # Following methods manually set self.__class__ fix to GeoDataFrame.
    # Thus to properly subtype, we need to downcast them with the _wrapped_gdf_method decorator.
    @_wrapped_gdf_method
    def __getitem__(self, key):
        return super().__getitem__(key)

    @_wrapped_gdf_method
    def copy(self, deep=True):
        return super().copy(deep=deep)

    @_wrapped_gdf_method
    def merge(self, *args, **kwargs):
        return super().merge(*args, **kwargs)


class TrackintelDataFrame(pd.DataFrame):
    """Helper class to subtype DataFrame and handle fallback"""

    # subclassing DataFrames is significant simpler.
    @property
    def _constructor(self):
        """Interface to subtype pandas properly"""
        return partial(self.__class__, validate=False)


class TrackintelBase(object):
    """Class for supplying basic functionality to all Trackintel classes."""

    # so far we don't have a lot of methods here
    # but a lot of IO code can be moved here.
    pass


class NonCachedAccessor:
    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        # copied code from pandas accessor, minus the caching
        return self._accessor(obj)


def _register_trackintel_accessor(name: str):
    from pandas import DataFrame

    def decorator(accessor):
        if hasattr(DataFrame, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(DataFrame)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
            )
        setattr(DataFrame, name, NonCachedAccessor(name, accessor))
        DataFrame._accessors.add(name)
        return accessor

    return decorator


# doc is derived from pandas.util._decorators (2.1.0)
# module https://github.com/pandas-dev/pandas/blob/main/LICENSE


def doc(*docstrings, **params):
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated):
        # collecting docstring and docstring templates
        components = []
        if decorated.__doc__:
            components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                components.extend(docstring._docstring_components)
            elif isinstance(docstring, str) or docstring.__doc__:
                components.append(docstring)

        decorated._docstring_components = components
        params_applied = (c.format(**params) if (isinstance(c, str) and params) else c for c in components)
        decorated.__doc__ = "".join(c if isinstance(c, str) else dedent(c.__doc__ or "") for c in params_applied)
        return decorated

    return decorator


_shared_docs = {}

# in _shared_docs as all write_postgis_xyz functions use this docstring
_shared_docs[
    "write_postgis"
] = """
Stores {long} to PostGIS. Usually, this is directly called on a {long}
DataFrame (see example below).

Parameters
----------{first_arg}
name : str
    The name of the table to write to.

con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
    active connection to PostGIS database.

schema : str, optional
    The schema (if the database supports this) where the table resides.

if_exists : str, {{'fail', 'replace', 'append'}}, default 'fail'
    How to behave if the table already exists.

    - fail: Raise a ValueError.
    - replace: Drop the table before inserting new values.
    - append: Insert new values to the existing table.

index : bool, default True
    Write DataFrame index as a column. Uses index_label as the column name in the table.

index_label : str or sequence, default None
    Column label for index column(s). If None is given (default) and index is True, then the index names are used.

chunksize : int, optional
    How many entries should be written at the same time.

dtype: dict of column name to SQL type, default None
    Specifying the datatype for columns.
    The keys should be the column names and the values should be the SQLAlchemy types.

Examples
--------
>>> {short}.to_postgis(conn_string, table_name)
>>> ti.io.write_{long}_postgis({short}, conn_string, table_name)
"""

_shared_docs[
    "write_csv"
] = """
Write {long} to csv file.

Wraps the pandas to_csv function.
Geometry get transformed to WKT before writing.

Parameters
----------{first_arg}
filename : str
    The file to write to.

args
    Additional arguments passed to pd.DataFrame.to_csv().

kwargs
    Additional keyword arguments passed to pd.DataFrame.to_csv().

Examples
--------
>>> {short}.to_csv("export_{long}.csv")
"""
