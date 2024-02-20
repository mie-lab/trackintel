import datetime

import numpy as np
import pandas as pd

from trackintel.geogr import get_speed_triplegs


def create_activity_flag(staypoints, method="time_threshold", time_threshold=15.0, activity_column_name="is_activity"):
    """
    Add a flag whether or not a staypoint is considered an activity based on a time threshold.

    Parameters
    ----------
    staypoints: Staypoints

    method: {'time_threshold'}, default = 'time_threshold'
        - 'time_threshold' : All staypoints with a duration greater than the time_threshold are considered an activity.

    time_threshold : float, default = 15 (minutes)
        The time threshold for which a staypoint is considered an activity in minutes. Used by method 'time_threshold'

    activity_column_name : str , default = 'is_activity'
        The name of the newly created column that holds the activity flag.

    Returns
    -------
    staypoints : Staypoints
        Original staypoints with the additional activity column

    Examples
    --------
    >>> sp  = sp.create_activity_flag(method='time_threshold', time_threshold=15)
    >>> print(sp['is_activity'])
    """
    if method == "time_threshold":
        staypoints[activity_column_name] = staypoints["finished_at"] - staypoints["started_at"] > datetime.timedelta(
            minutes=time_threshold
        )
    else:
        raise ValueError(f"Method {method} not known for creating activity flag.")

    return staypoints


def predict_transport_mode(triplegs, method="simple-coarse", **kwargs):
    """
    Predict the transport mode of triplegs.

    Predict/impute the transport mode that was likely chosen to cover the given
    tripleg, e.g., car, bicycle, or walk.

    Parameters
    ----------
    triplegs: Triplegs

    method: {'simple-coarse'}, default 'simple-coarse'
        The following methods are available for transport mode inference/prediction:

        - 'simple-coarse' : Uses simple heuristics to predict coarse transport classes.

    Returns
    -------
    triplegs : Triplegs
        The triplegs with added column mode, containing the predicted transport modes.

    Notes
    -----
    ``simple-coarse`` method includes ``{'slow_mobility', 'motorized_mobility', 'fast_mobility'}``.
    In the default classification, ``slow_mobility`` (<15 km/h) includes transport modes such as
    walking or cycling, ``motorized_mobility`` (<100 km/h) modes such as car or train, and
    ``fast_mobility`` (>100 km/h) modes such as high-speed rail or airplanes.
    These categories are default values and can be overwritten using the keyword argument categories.

    Examples
    --------
    >>> tpls  = tpls.predict_transport_mode()
    >>> print(tpls["mode"])
    """
    if method == "simple-coarse":
        # implemented as keyword argument if later other methods that don't use categories are added
        categories = kwargs.pop(
            "categories", {15 / 3.6: "slow_mobility", 100 / 3.6: "motorized_mobility", np.inf: "fast_mobility"}
        )
        triplegs = triplegs.copy()
        triplegs["mode"] = _predict_transport_mode_simple_coarse(triplegs, categories)
        return triplegs
    else:
        raise ValueError(f"Method {method} not known for predicting tripleg transport modes.")


def _predict_transport_mode_simple_coarse(triplegs, categories):
    """
    Predict a transport mode based on provided categories.

    Implements a simple speed based heuristic (over the whole tripleg).
    As such, it is very fast, but also very simple and coarse.

    Parameters
    ----------
    triplegs : Triplegs
        The triplegs for the transport mode prediction.

    categories : dict, optional
        The categories for the speed classification {upper_boundary: 'category_name'}.
        The unit for the upper boundary is m/s.

    Returns
    -------
    cuts : pd.Series
        Column containing the predicted transport modes.

    For additional documentation, see
    :func:`trackintel.analysis.transport_mode_identification.predict_transport_mode`.
    """
    categories = dict(sorted(categories.items(), key=lambda item: item[0]))
    intervals = pd.IntervalIndex.from_breaks([-np.inf] + list(categories.keys()), closed="left")
    speed = get_speed_triplegs(triplegs)["speed"]
    cuts = pd.cut(speed, intervals)
    return cuts.cat.rename_categories(categories.values())
