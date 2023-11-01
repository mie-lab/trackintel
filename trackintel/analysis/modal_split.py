import pandas as pd

from trackintel.geogr import check_gdf_planar, calculate_haversine_length


def calculate_modal_split(tpls, freq=None, metric="count", per_user=False, norm=False):
    """
    Calculate the modal split of triplegs

    Parameters
    ----------
    tpls : Triplegs
        triplegs require the column `mode`.
    freq : str
        frequency string passed on as `freq` keyword to the pandas.Grouper class. If `freq=None` the modal split is
        calculated on all data. A list of possible
        values can be found `here <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset
        -aliases>`_.
    metric : {'count', 'distance', 'duration'}
        Aggregation used to represent the modal split. 'distance' returns in the same unit as the crs. 'duration'
        returns values in seconds.
    per_user : bool, default: False
        If True the modal split is calculated per user
    norm : bool, default: False
        If True every row of the modal split is normalized to 1

    Returns
    -------
    modal_split : DataFrame
        The modal split represented as pandas Dataframe with (optionally) a multi-index. The index can have the
        levels: `('user_id', 'timestamp')` and every mode as a column.

    Notes
    ------
        `freq='W-MON'` is used for a weekly aggregation that starts on mondays.
        If `freq=None` and `per_user=False` are passed the modal split collapses to a single column.
        The modal split can be visualized using :func:`trackintel.plot_modal_split`

    Examples
    --------
    >>> triplegs.calculate_modal_split()
    >>> tripleg.calculate_modal_split(freq='W-MON', metric='distance')
    """
    tpls = tpls.copy()  # copy as we add additional columns on tpls

    # count on mode, sum on length and duration
    agg = "sum"
    # calculate distance and duration if required
    if metric == "distance":
        tpls[metric] = _calculate_length(tpls)
    elif metric == "duration":
        tpls[metric] = (tpls["finished_at"] - tpls["started_at"]).dt.total_seconds()
    elif metric == "count":
        agg = "count"
        metric = "mode"  # count on mode
    else:
        error_msg = f"Metric {metric} unknown, only metrics {{'count', 'distance', 'duration'}} are supported."
        raise AttributeError(error_msg)

    group = []
    if per_user:
        group = ["user_id"]

    if freq is not None:
        tpls.set_index("started_at", inplace=True)
        tpls.index.name = "timestamp"
        group.append(pd.Grouper(freq=freq))

    modal_split = pd.pivot_table(tpls, index=group, columns=["mode"], aggfunc={metric: agg}, fill_value=0)
    if group:  # non-empty group creates MultiIndex that we need to handle
        modal_split.columns = modal_split.columns.droplevel(0)

    if norm:  # norm rows to 1
        return modal_split.div(modal_split.sum(axis=1), axis=0)
    return modal_split


def _calculate_length(tpls):
    """Help function to calculate length of tripleg.

    Checks if crs is planar or if not. If not uses ``calculate_haversine_length``.

    Parameters
    ----------
    tpls : Triplegs
    """
    if check_gdf_planar(tpls):
        return tpls.length  # if planar use geopandas function
    return pd.Series(calculate_haversine_length(tpls), index=tpls.index)
