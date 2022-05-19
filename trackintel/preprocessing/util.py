from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def calc_temp_overlap(start_1, end_1, start_2, end_2):
    """
    Calculate the portion of the first time span that overlaps with the second.

    Parameters
    ----------
    start_1: datetime
        start of first time span
    end_1: datetime
        end of first time span
    start_2: datetime
        start of second time span
    end_2: datetime
        end of second time span

    Returns
    -------
    float:
        The ratio by which the first timespan overlaps with the second.

    Examples
    --------
    >>> ti.preprocessing.calc_temp_overlap(start_1, end_1, start_2, end_2)

    """
    start = max(start_1, start_2)
    end = min(end_1, end_2)
    temp_overlap = max(timedelta(0), end - start)

    dur = end_1 - start_1
    if dur <= timedelta(0):
        return 0  # either invalid or division 0
    return temp_overlap / dur


def applyParallel(dfGrouped, func, n_jobs, print_progress, **kwargs):
    """
    Funtion warpper to parallelize funtions after .groupby().

    Parameters
    ----------
    dfGrouped: pd.DataFrameGroupBy
        The groupby object after calling df.groupby(COLUMN).

    func: function
        Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).

    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description

    print_progress: boolean
        If set to True print the progress of apply.

    **kwargs:
        Other arguments passed to func.

    Returns
    -------
    pd.DataFrame:
        The result of dfGrouped.apply(func)

    Examples
    --------
    >>> from trackintel.preprocessing.util import applyParallel
    >>> applyParallel(tpfs.groupby("user_id", as_index=False), func, n_jobs=2)
    """
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)


def _explode_agg(column, agg, orig_df, agg_df):
    """
    Assign new aggrated information back to the original dataframe.

    Parameters
    ----------
    column : IndexLabel
        Column(s) to explode. Should be index column of orig_df.
    agg : IndexLabel
        Aggregate column to join back to original df.
    orig_df : pd.DataFrame
        Original Dataframe without the aggregate column.
    agg_df : pd.DataFrame
        Dataframe with the aggregate column.

    Returns
    -------
    pd.DataFrame
        Original Dataframe with additional colum from aggregated DataFrame.
    """
    temp = agg_df.explode(column)
    temp = temp[temp[column].notna()]
    temp.index = temp[column]
    return orig_df.join(temp[agg], how="left")
