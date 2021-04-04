def calc_temp_overlap(start_1, end_1, start_2, end_2):
    """
    Calculate the portion of the first time span that overlaps with the second

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
        The ratio by which the

    """

    # case 1: no overlap - 1 was before 2
    if end_1 < start_2:
        return 0

    # case 2: no overlap - 1 comes after 2
    elif end_2 < start_1:
        return 0

    # case 3: 2 fully in 1
    if (start_1 <= start_2) and (end_1 >= end_2):
        temp_overlap = end_2 - start_2

    # case 4: 1 fully in 2
    elif (start_2 <= start_1) and (end_2 >= end_1):
        temp_overlap = end_1 - start_1

    # case 5: 1 overlaps 2 from right
    elif (start_2 <= start_1) and (end_2 <= end_1):
        temp_overlap = end_2 - start_1

    # case 6: 1 overlaps 2 from left
    elif (start_1 <= start_2) and (end_1 <= end_2):
        temp_overlap = end_1 - start_2

    else:
        raise Exception("wrong case")

    temp_overlap = temp_overlap.total_seconds()

    # no overlap at all
    assert temp_overlap >= 0, "the overlap can not be lower than 0"

    dur = end_1 - start_1
    if dur.total_seconds() == 0:
        return 0
    else:
        overlap_ratio = temp_overlap / dur.total_seconds()

    return overlap_ratio
