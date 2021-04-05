import datetime


def create_activity_flag(staypoints, method="time_threshold", time_threshold=5.0, activity_column_name="activity"):
    """
    Add a flag whether or not a staypoint is considered an activity.

    Parameters
    ----------
    staypoints: GeoDataFrame (as trackintel staypoints)
        The original input staypoints

    method: {'time_threshold'}, default = 'time_threshold'

        - 'time_threshold' : All staypoints with a duration greater than the time_threshold are considered an activity.

    time_threshold : float, default = 5 (minutes)
        The time threshold for which a staypoint is considered an activity in minutes. Used by method 'time_threshold'

    activity_column_name : str , default = 'activity'
        The name of the newly created column that holds the activity flag.

    Returns
    -------
    staypoints : GeoDataFrame (as trackintel staypoints)
        Original staypoints with the additional activity column

    Examples
    --------
    >>> stps  = stps.as_staypoints.create_activity_flag(method='time_threshold', time_threshold=5)
    >>> print(stps['activity'])
    """
    if method == "time_threshold":
        staypoints[activity_column_name] = staypoints["finished_at"] - staypoints["started_at"] > datetime.timedelta(
            minutes=time_threshold
        )
    else:
        raise NameError("Method {} is not known".format(method))

    return staypoints
