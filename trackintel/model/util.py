from functools import partial, update_wrapper, wraps
import pandas as pd
import warnings
from geopandas import GeoDataFrame


def _copy_docstring(wrapped, assigned=("__doc__",), updated=[]):
    """Thin wrapper for `functools.update_wrapper` to mimic `functools.wraps` but to only copy the docstring."""
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)


def _wrapped_gdf_method(func):
    """Decorator function that downcast types to trackintel class if is (Geo)DataFrame and has the required columns."""

    @wraps(func)  # copy all metadata
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, pd.DataFrame) and self._check(result, validate_geometry=False):
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
    - If check fails or is DataFrame without fallback_class set, return GeoDataFrame/DataFrame instance w/o changes
    - If check succeeds and has geometry, change class to current class and return
    - If check succeeds and has no geometry, change class to fallback_class and return
    """

    # set fallback_class for succeeding _check and missing geometry, if None default to pd.DataFrame
    fallback_class = None

    @staticmethod
    def _check(self, validate_geometry=True):
        raise NotImplementedError

    @property
    def _constructor(self):
        """Interface to subtype pandas properly"""
        super_cons = super()._constructor

        def _constructor_with_fallback(*args, **kwargs):
            result = super_cons(*args, **kwargs)
            if not self._check(result, validate_geometry=False):
                # check fails -> not one of our classes
                return result
            if isinstance(result, GeoDataFrame):
                # cannot validate geometry as geometry column is maybe not set
                return self.__class__(result, validate_geometry=False)
            if self.fallback_class is not None:
                return self.fallback_class(result)
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

    @staticmethod
    def _check(self):  # has no geometry to check
        raise NotImplementedError

    @property
    def _constructor(self):
        """Interface to subtype pandas properly"""
        super_cons = super()._constructor

        def _constructor_with_fallback(*args, **kwargs):
            result = super_cons(*args, **kwargs)
            if not self._check(result):
                return result
            return self.__class__(result)

        return _constructor_with_fallback


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
