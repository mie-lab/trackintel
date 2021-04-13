import numpy as np
import pandas as pd

from trackintel.geogr.distances import check_gdf_crs, calculate_haversine_length


def calculate_modal_split(tpls_in, freq=None, metric="count", per_user=False, norm=False):
    """Calculate the modal split of triplegs

    Parameters
    ----------
    tpls_in : GeoDataFrame (as trackintel triplegs)
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

        The modal split can be visualized using :func:`trackintel.visualization.modal_split.plot_modal_split`

    Examples
    --------
    >>> triplegs.calculate_modal_split()
    >>> tripleg.calculate_modal_split(freq='W-MON', metric='distance')

    """
    tpls = tpls_in.copy()

    # precalculate distance and duration if required
    if metric == "distance":
        if_planer_crs = check_gdf_crs(tpls)
        if not if_planer_crs:
            tpls["distance"] = calculate_haversine_length(tpls)
        else:
            tpls["distance"] = tpls.length
    elif metric == "duration":
        tpls["duration"] = tpls["finished_at"] - tpls["started_at"]

    # create grouper
    if freq is None:
        if per_user:
            tpls_grouper = tpls.groupby(["user_id", "mode"])
        else:
            tpls_grouper = tpls.groupby(["mode"])
    else:
        tpls.set_index("started_at", inplace=True)
        tpls.index.name = "timestamp"
        if per_user:
            tpls_grouper = tpls.groupby(["user_id", "mode", pd.Grouper(freq=freq)])
        else:
            tpls_grouper = tpls.groupby(["mode", pd.Grouper(freq=freq)])

    # aggregate
    if metric == "count":
        modal_split = tpls_grouper["mode"].count()

    elif metric == "distance":
        modal_split = tpls_grouper["distance"].sum()

    elif metric == "duration":
        modal_split = tpls_grouper["duration"].sum()
        modal_split = modal_split.dt.total_seconds()

    # move 'mode' to columns
    modal_split.name = "modal_split"
    modal_split = pd.DataFrame(modal_split)

    # if mode is the only index, we replace it with zeros so that everything gets aggregated into a single row
    modal_split["mode"] = modal_split.index.get_level_values("mode")
    if modal_split.index.nlevels == 1:
        modal_split.index = 0 * np.arange(0, modal_split.shape[0])
    else:
        modal_split = modal_split.droplevel("mode")

    # transform Dataframe such that:
    # - unique mode names are column names
    # - time/user_id are the indices
    modal_split = modal_split.pivot_table(index=modal_split.index, columns="mode", values="modal_split")
    modal_split.fillna(0, inplace=True)

    if norm:
        # norm rows to 1
        modal_split = modal_split.div(modal_split.sum(axis=1), axis=0)

    return modal_split
