from functools import partial, update_wrapper, wraps
import numpy as np
import pandas as pd
import warnings
from geopandas import GeoDataFrame

import trackintel as ti
from trackintel.geogr.distances import calculate_haversine_length, check_gdf_planar
from trackintel.geogr.point_distances import haversine_dist


def get_speed_positionfixes(positionfixes):
    """
    Compute speed per positionfix (in m/s)

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes have to follow the standard definition for positionfixes DataFrames.

    Returns
    -------
    pfs: GeoDataFrame (as trackintel positionfixes)
        The original positionfixes with a new column ``[`speed`]``. The speed is given in m/s

    Notes
    -----
    The speed at one positionfix is computed from the distance and time since the previous positionfix. For the first
    positionfix, the speed is set to the same value as for the second one.
    """
    pfs = positionfixes.copy()
    is_planar_crs = ti.geogr.distances.check_gdf_planar(pfs)

    g = pfs.geometry
    # get distance and time difference
    if is_planar_crs:
        dist = g.distance(g.shift(1)).to_numpy()
    else:
        x = g.x.to_numpy()
        y = g.y.to_numpy()
        dist = np.zeros(len(pfs), dtype=np.float64)
        dist[1:] = haversine_dist(x[:-1], y[:-1], x[1:], y[1:])

    time_delta = (pfs["tracked_at"] - pfs["tracked_at"].shift(1)).dt.total_seconds().to_numpy()
    # compute speed (in m/s)
    speed = dist / time_delta
    speed[0] = speed[1]  # The first point speed is imputed
    pfs["speed"] = speed
    return pfs


def get_speed_triplegs(triplegs, positionfixes=None, method="tpls_speed"):
    """
    Compute the average speed per positionfix for each tripleg (in m/s)

    Parameters
    ----------
    triplegs: GeoDataFrame (as trackintel triplegs)
        The generated triplegs as returned by ti.preprocessing.positionfixes.generate_triplegs

    positionfixes (Optional): GeoDataFrame (as trackintel positionfixes)
        The positionfixes as returned by ti.preprocessing.positionfixes.generate_triplegs. Only required if the method
        is 'pfs_mean_speed'. In addition the standard columns it must include the column ``[`tripleg_id`]``.

    method: str
        Method how the speed is computed, one of {tpls_speed, pfs_mean_speed}. The 'tpls_speed' method simply divides
        the overall tripleg distance by its duration, while the 'pfs_mean_speed' method is the mean pfs speed.

    Returns
    -------
    tpls: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`speed`]``. The speed is given in m/s.
    """
    # Simple method: Divide overall tripleg distance by overall duration
    if method == "tpls_speed":
        if check_gdf_planar(triplegs):
            distance = triplegs.length
        else:
            distance = calculate_haversine_length(triplegs)
        duration = (triplegs["finished_at"] - triplegs["started_at"]).dt.total_seconds()
        # The unit of the speed is m/s
        tpls = triplegs.copy()
        tpls["speed"] = distance / duration
        return tpls

    # Pfs-based method: compute speed per positionfix and average then
    elif method == "pfs_mean_speed":
        if positionfixes is None:
            raise AttributeError('Method "pfs_mean_speed" requires positionfixes as input.')
        if "tripleg_id" not in positionfixes:
            raise AttributeError('Positionfixes must include column "tripleg_id".')
        # group positionfixes by triplegs and compute average speed for each collection of positionfixes
        grouped_pfs = positionfixes.groupby("tripleg_id").apply(_single_tripleg_mean_speed)
        # add the speed values to the triplegs column
        tpls = pd.merge(triplegs, grouped_pfs.rename("speed"), how="left", left_index=True, right_index=True)
        tpls.index = tpls.index.astype("int64")
        return tpls

    else:
        raise AttributeError(f"Method {method} not known for speed computation.")


def _single_tripleg_mean_speed(positionfixes):
    pfs_sorted = positionfixes.sort_values(by="tracked_at")
    pfs_speed = get_speed_positionfixes(pfs_sorted)
    return np.mean(pfs_speed["speed"].values[1:])


def _copy_docstring(wrapped, assigned=("__doc__",), updated=[]):
    """Thin wrapper for `functools.update_wrapper` to mimic `functools.wraps` but to only copy the docstring."""
    return partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)


def _wrapped_gdf_method(func):
    """Decorator function that downcast types to trackintel class if is GeoDataFrame and has the required columns."""

    @wraps(func)  # copy all metadata
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if not isinstance(result, GeoDataFrame) or not self._check(result, validate_geometry=False):
            return result
        # as geopandas only change the __class__ attribute, we can just change it back
        result.__class__ = self.__class__
        return result

    return wrapper


class TrackintelGeoDataFrame(GeoDataFrame):
    """Helper class to subtype GeoDataFrame correctly."""

    @staticmethod
    def _check(self, validate_geometry=True):
        raise NotImplementedError

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

    @property
    def _constructor(self):
        """Interface to subtype pandas properly"""
        # we loose access to self in inner function -> write objects to local variables
        super_cons = super()._constructor

        def _constructor_with_fallback(*args, **kwargs):
            result = super_cons(*args, **kwargs)
            # cannot validate_geometry as geometry column is maybe not set
            if isinstance(result, GeoDataFrame) and self._check(result, validate_geometry=False):
                return self.__class__(result, validate_geometry=False)
            return result

        return _constructor_with_fallback


def _wrapped_gdf_method_fallback(func):
    """Decorator function that downcast types to trackintel class if is (Geo)DataFrame and has the required columns."""

    @wraps(func)  # copy all metadata
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if isinstance(result, pd.DataFrame) and self._check(result, validate_geometry=False):
            if isinstance(result, GeoDataFrame):
                result.__class__ = self.__class__
            else:
                result.__class__ = self.fallback_class
        return result

    return wrapper


# short and memorizable name
class TrackintelGeoDataFrameWithFallback(GeoDataFrame):
    """
    Helper class to subtype GeoDataFrame correctly and fallback to custom class.

    Fallback to fallback_class if looses geometry but _check does still succeed.
    Fallback to GeoDataFrame if still has geometry but _check fails.
    Fallback to DataFrame if looses geometry and _check fails.
    """

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
            return self.fallback_class(result)

        return _constructor_with_fallback

    # Following methods manually set self.__class__ fix to GeoDataFrame.
    # Thus to properly subtype, we need to downcast them with the _wrapped_gdf_method_fallback decorator.
    @_wrapped_gdf_method_fallback
    def __getitem__(self, key):
        return super().__getitem__(key)

    @_wrapped_gdf_method_fallback
    def copy(self, deep=True):
        return super().copy(deep=deep)

    @_wrapped_gdf_method_fallback
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
