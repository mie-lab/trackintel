import numpy as np

from tqdm import tqdm


def random_entropy(stps, print_progress=False):
    """Random entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    s: pd.Series
        the random entropy calculated at the individual level.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User random entropy calculation")
        s = stps.groupby("user_id").progress_apply(lambda x: _random_entropy_user(x))
    else:
        s = stps.groupby("user_id").apply(lambda x: _random_entropy_user(x))

    s.rename("randomEntropy", inplace=True)
    return s


def uncorrelated_entropy(stps, print_progress=False):
    """
    Uncorrelated entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User uncorrelated entropy calculation")
        s = stps.groupby("user_id").progress_apply(lambda x: _uncorrelated_entropy_user(x))
    else:
        s = stps.groupby("user_id").apply(lambda x: _uncorrelated_entropy_user(x))

    s.rename("uncorrelatedEntropy", inplace=True)
    return s


def real_entropy(stps, print_progress=False):
    """
    Real entropy of individual visited locations.

    Parameters
    ----------
    stps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".

    print_progress: boolen, default False
        Show per-user progress if set to True.

    Returns
    -------
    pandas DataFrame
        the real entropy of the individuals.

    References
    ----------
    [1] Song, C., Qu, Z., Blumm, N. and Barabási, A.L., 2010. Limits of predictability in human mobility. Science, 327(5968), pp.1018-1021.

    """
    if print_progress:
        tqdm.pandas(desc="User uncorrelated entropy calculation")
        s = stps.sort_values(by="started_at").groupby("user_id").progress_apply(lambda x: _real_entropy_user(x))
    else:
        s = stps.sort_values(by="started_at").groupby("user_id").apply(lambda x: _real_entropy_user(x))

    s.rename("realEntropy", inplace=True)
    return s


def _random_entropy_user(stps_user):
    """
    User level random entropy calculation, see random_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the random entropy of the individual
    """
    locs_numb = len(stps_user["location_id"].unique())
    return np.log2(locs_numb)


def _uncorrelated_entropy_user(stps_user):
    """
    User level uncorrelated entropy calculation, see uncorrelated_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    locs_prob = stps_user["location_id"].value_counts(normalize=True, sort=False).values
    return -(locs_prob * np.log2(locs_prob)).sum()


def _real_entropy_user(stps_user):
    """
    User level real entropy calculation, see real_entropy() for details.

    Parameters
    ----------
    stps_user : Geodataframe (as trackintel staypoints)
        The staypoints from an individual, should contain column "location_id".

    Returns
    -------
    float
        the real entropy of the individual
    """
    locs_series = stps_user["location_id"].values

    n = len(locs_series)

    sum_lambda = 0

    # 1 to ensure to consider the first situation from where
    # locs_series[i:j] = [] and locs_series[i:j] = locs_series[0:1]
    # TODO: sckitmobility also add 2 here to account for the "last elements", I am not sure why
    sum_lambda = sum_lambda + 1

    for i in range(1, n - 1):
        j = i + 1

        while True:
            # if the locs_series[i:j] is longer than locs_series[:i],
            # we can no longer find it locs_series[i:j] in locs_series[:i]
            if j - i > i:
                # TODO: I did not check whether this is correct
                j = n
                break

            # if locs_series[i:j] exist in locs_series[:i], we increase j by 1
            # sliding_window_view creates sublist of length len(locs_series[i:j]) from locs_series[:i]
            ls = np.lib.stride_tricks.sliding_window_view(locs_series[:i], j - i).tolist()
            if [locs_series[i:j]] in ls:
                j += 1
            else:
                # we find the "shortest substring" that does not exist in locs_series[:i]
                break

        # length of the substring
        sum_lambda += j - i

    # TODO: this is the function S5 from the suppl. material, sckitmobility uses np.log2
    return 1.0 / (sum_lambda * 1 / n) * np.log(n)
