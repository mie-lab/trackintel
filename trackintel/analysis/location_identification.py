_RECEIPTS = {
    "FREQ": freq_receipt,
}


def location_identifier(sps, pre_filter=True, receipt="FREQ"):
    """Finde home and work/location for each user with different receipts.

    Parameters
    ----------
    sps : Geodataframe (as trackintel staypoints)
        Staypoints with column "location_id".
    pre_filter : bool, default True
        Prefiltering the staypoints to exclude locations with not enough data.
        The filter function can also be acessed via `pre_filter`.
    receipt : {'FREQ'}, default "FREQ"
        Choose which receipt to use.
        - FREQ: Select the most visited location
        - PASS: just for decoration.

    Returns
    -------
    Geodataframe (as trackintel staypoints)
        With additional column assigning one of three location identies {'home', 'work', None}.

    Note
    ----
    The receipt are adapted from [1]. The original algorithms count the distinct hours at a
    location as the home location is derived from geo-tagged tweets. We directly sum the time
    spent at a location as our data model includes that.

    References
    ----------
    Chen, Qingqing, and Ate Poorthuis. 2021.
    ‘Identifying Home Locations in Human Mobility Data: An Open-Source R Package for Comparison and Reproducibility’.
    International Journal of Geographical Information Science 0 (0): 1–24.
    https://doi.org/10.1080/13658816.2021.1887489.

    Examples
    --------
    >>> ti.analysis.location_identification.location_idenifier(sps, pre_filter=True, receipt="FREQ")
    """
    # what do we do here?
    # we take the gdf and assert two things 1. is staypoint 2. has location_id column
    assert sps.as_staypoints
    if "location_id" not in sps.columns:
        raise KeyError((
            "To derive location activities the GeoDataFrame (as trackintel staypoints)must have a column "
            f"named 'location_id' but it has [{', '.join(sps.columns)}]"))
    # then hand it to to the filter function if necessary.
    if pre_filter:
        sps = pre_filter_locations()

    m = _RECEIPTS[receipt]()  # das müssen wir mal schöner machen.

    return m


def pre_filter_locations(sps, tresh_sp, thresh_min_locs, tresh_sp_at_locs, thresh_time, thresh_period):
    """Filter locations and user out that have not enough data to do a proper analysis.

    Parameters
    ----------
    sps : GeoDataFrame (as trackintel staypoints)
        Staypoints with the column "location_id".
    thresh_sp : int
        Minimum staypoints a user must have to be included.
    thresh_min_locs : int
        Minimum locations a user must have to be included.
    tresh_sp_at_locs : int
        Minimum number of staypoints at a location to have to include location.
    thresh_time : int
        Minimum timespan that a user must spend at location.
    thresh_period : pd.TimePeriod
        Minimum number of time a user have spent at location.

    Returns
    -------
    GeoDataFrame (as trackintel staypoints)
        A boolean index arrray with True everywhere the Bedingungen are erfüllt.

    Examples
    --------
    >> do something
    """
    return sps.copy()


def freq_receipt(sps):
    """Docstring here :D
    """

    pass


def staypoints_with_location_assertion(funct):
    