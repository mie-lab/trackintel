from datetime import timedelta
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

import trackintel as ti


def get_trips_grouped(trips, tours):
    """Helper function to get grouped trips by tour id

    Parameters
    ----------
    trips: GeoDataFrame (as trackintel trips)
        Trips dataframe

    tours: GeoDataFrame (as trackintel tours)
        Output of generate_tours function, must contain column "trips" with list of trip ids on tour

    Returns
    -------
    trips_grouped_by_tour: DataFrameGroupBy object
        Trips grouped by tour id

    Examples
    --------
    >>> get_trips_grouped(trips, tours)

    Notes
    -------
    This function is necessary because when running generate_tours, one trip only gets the tour ID of the smallest
    tour it belongs to assigned. Here, we return all trips for each tour, which might contain a nested tour.
    """
    trips_inp = trips.copy()
    if "tour_id" in trips_inp.columns:
        trips_inp.drop(columns=["tour_id"], inplace=True)
    # make smaller version of tours
    tours_to_trips = tours.reset_index()[["id", "trips"]]
    # switch to trips id as index
    tours_to_trips.rename(columns={"id": "tour_id", "trips": "trip_id"}, inplace=True)
    # expand this small version so that each trip id is one row
    tours_expanded = tours_to_trips.explode("trip_id").reset_index(drop=True)

    # join with trips table by id
    tours_with_trips = tours_expanded.merge(trips_inp, left_on="trip_id", right_on="id", how="left")
    # group
    trips_grouped_by_tour = tours_with_trips.groupby("tour_id")
    return trips_grouped_by_tour


def generate_tours(
    trips,
    staypoints=None,
    max_dist=100,
    max_time="1d",
    max_nr_gaps=0,
    print_progress=False,
):
    """
    Generate trackintel-tours from trips

    Parameters
    ----------
    trips : GeoDataFrame (as trackintel trips)
        The trips have to follow the standard definition for trips DataFrames

    staypoints : GeoDataFrame (as trackintel staypoints, preprocessed to contain location IDs), default None
        The staypoints have to follow the standard definition for staypoints DataFrames. The location ID column
        is necessary to connect trips via locations to a tour. If None, trips will be connected based only on a
        distance threshold `max_dist`.

    max_dist: float, default 100 (meters)
        Maximum distance between the end point of one trip and the start point of the next trip on a tour.
        This is parameter is only used if staypoints is None!
        Also, if `max_nr_gaps > 0`, a tour can contain larger spatial gaps (see Notes below)

    max_time: str or pd.Timedelta, default "1d" (1 day)
        Maximum time that a tour is allowed to take

    max_nr_gaps: int, default 0
        Maximum number of spatial gaps on the tour. Use with caution - see notes below.

    print_progress : bool, default False
        If print_progress is True, the progress bar is displayed


    Returns
    -------
    trips_with_tours: GeoDataFrame (as trackintel trips)
        Same as `trips`, but with column `tour_id`, containing a list of the tours that the trip is part of (see notes).

    tours: GeoDataFrame (as trackintel tours)
        The generated tours

    Examples
    --------
    >>> trips.as_trips.generate_tours(staypoints)

    Notes
    -------
    - Tours are defined as a collection of trips in a certain time frame that start and end at the same point
    - Tours and trips have an N:N relationship: One tour consists of multiple trips, but also one trip can be part of
      multiple tours, due to nested tours or overlapping tours.
    - This function implements two possibilities to generate tours of trips: Via the location ID in the `staypoints`
      df, or via a maximum distance. Thus, note that only one of the parameters `staypoints` or `max_dist` is used!
    - Nested tours are possible and will be regarded as 2 (or more tours).
    - It is possible to allow spatial gaps to occur on the tour, which might be useful to deal with missing data.
      Example: The two trips home-work, supermarket-home would still be detected as a tour when max_nr_gaps >= 1,
      although the work-supermarket trip is missing.
      Warning: This only counts the number of gaps, but neither temporal or spatial distance of gaps, nor the number
      of missing trips in a gap are bounded. Thus, this parameter should be set with caution, because trips that are
      hours apart might still be connected to a tour if `max_nr_gaps > 0`.
    """
    # Two options: either the location IDs for staypoints on the trips are provided, or a maximum distance threshold
    # between end and start of trips is used
    if staypoints is not None:
        assert (
            "location_id" in staypoints.columns
        ), "Staypoints with location ID is required, otherwise tours are generated without location using max_dist"
        geom_col = None  # not used
        crs_is_projected = False  # not used
    else:
        # if no location is given, we need the trips table to have a geometry column
        assert isinstance(trips, gpd.geodataframe.GeoDataFrame), "Trips table must be a GeoDataFrame"
        geom_col = trips.geometry.name
        # get crs
        crs_is_projected = ti.geogr.distances.check_gdf_planar(trips)

    # convert max_time to timedelta
    if isinstance(max_time, str):
        max_time = pd.to_timedelta(max_time)
    # otherwise check if it's a Timedelta already, and raise error if not
    elif not isinstance(max_time, pd.Timedelta):
        raise TypeError("Parameter max_time must be either of type String or pd.Timedelta!")

    trips_input = trips.copy()
    # If the trips already have a column "tour_id", we drop it
    if "tour_id" in trips_input:
        trips_input.drop(columns="tour_id", inplace=True)
        warnings.warn("Deleted existing column 'tour_id' from trips.")

    kwargs = {
        "max_dist": max_dist,
        "max_nr_gaps": max_nr_gaps,
        "max_time": max_time,
        "staypoints": staypoints,
        "geom_col": geom_col,
        "crs_is_projected": crs_is_projected,
    }
    if print_progress:
        tqdm.pandas(desc="User trip generation")
        tours = (
            trips_input.groupby(["user_id"], group_keys=False, as_index=False)
            .progress_apply(_generate_tours_user, **kwargs)
            .reset_index(drop=True)
        )
    else:
        tours = (
            trips_input.groupby(["user_id"], group_keys=False, as_index=False)
            .apply(_generate_tours_user, **kwargs)
            .reset_index(drop=True)
        )

    # No tours found
    if len(tours) == 0:
        warnings.warn("No tours can be generated, return empty tours")
        return trips_input, tours

    # index management
    tours["id"] = np.arange(len(tours))
    tours.set_index("id", inplace=True)

    # assign tour id to trips
    tour2trip_map = tours.reset_index().explode("trips").rename(columns={"id": "tour_id"})
    # Each trip is only assigned to one tour. If a trip belongs to multiple tours, we can find its smallest subtour
    # by using the first one it is assigned to (nested tours are always found before big tours - have smaller tour_id)
    temp = tour2trip_map.groupby("trips").agg({"tour_id": list})

    trips_with_tours = trips_input.join(temp, how="left")

    # trips id (generated by this function) should be int64
    tours.index = tours.index.astype("int64")

    return trips_with_tours, tours


def _generate_tours_user(
    user_trip_df,
    staypoints=None,
    max_dist=100,
    max_nr_gaps=0,
    max_time=timedelta(days=1),
    geom_col="geom",
    crs_is_projected=False,
):
    """
    Compute tours from trips for one user

    Parameters
    ----------
    user_trip_df : GeoDataFrame (as trackintel trips)
        The trips have to follow the standard definition for trips DataFrames

    staypoints : GeoDataFrame (as trackintel staypoints, preprocessed to contain location IDs), default None
        The staypoints have to follow the standard definition for staypoints DataFrames. The location ID column
        is necessary to connect trips via locations to a tour. If None, trips will be connected based only on a
        distance threshold `max_dist`.

    max_dist: float, default 100 (meters)
        Maximum distance between the end point of one trip and the start point of the next trip on a tour.
        However, if `max_nr_gaps > 0`, a tour can contain larger spatial gaps (see notes in `generate_tours`)

    max_time: Timedelta, default 1 day
        Maximum time that a tour is allowed to take

    max_nr_gaps: int, default 0
        Maximum number of spatial gaps on the tour. Use with caution - see notes in `generate_tours`.

    geom_col : str, optional
        Name of geometry column of user_trip_df, by default "geom"

    crs_is_projected : bool, optional
        Whether the crs of user_trip_df is projected, by default False

    Returns
    -------
    tours_df: DataFrame
        Tours for one user
    """
    user_id = user_trip_df["user_id"].unique()
    assert len(user_id) == 1
    user_id = user_id[0]

    # sort by time
    user_trip_df = user_trip_df.sort_values(by=["started_at"])

    # save only the trip id (row.name) in the start candidates
    start_candidates = []

    # collect tours
    tours = []
    # Iterate over trips
    for _, row in user_trip_df.iterrows():
        end_time = row["finished_at"]

        if len(start_candidates) > 0:
            # Check if there is a spatial gap between the previous and current trip:
            # If staypoints with locations are available, check whether they share the same location
            if staypoints is not None:
                end_start_at_same_loc = _check_same_loc(
                    user_trip_df.loc[start_candidates[-1], "destination_staypoint_id"],  # dest. stp of previous trip
                    row["origin_staypoint_id"],  # start stp of current trip
                    staypoints,
                )
            else:
                # If no locations are available, check whether the distance is smaller than max_dist
                end_start_at_same_loc = _check_max_dist(
                    user_trip_df.loc[start_candidates[-1], geom_col][1],  # destination point of previous trip
                    row[geom_col][0],  # start point of current trip
                    max_dist,
                    crs_is_projected,
                )

            # if the current trip does not start at the end of the previous trip, there is a gap
            if not end_start_at_same_loc:
                # option 1: no gaps allowed - start search again
                if max_nr_gaps == 0:
                    start_candidates = [row.name]
                    continue
                # option 2: gaps allowed - search further
                else:
                    start_candidates.append(np.nan)

        # Add this point as a candidate
        start_candidates.append(row.name)

        # Check whether endpoint would be an unknown activity
        if pd.isna(row["destination_staypoint_id"]):
            continue

        # keep a list of which candidates to remove (because of time frame)
        new_list_start = 0

        # keep track of how many gaps we encountered, if greater than max_nr_gaps then stop
        gap_counter = 0

        # check for all candidates whether they form a tour with the current trip
        for j, cand in enumerate(start_candidates[::-1]):
            # gap
            if np.isnan(cand):
                gap_counter += 1
                if gap_counter > max_nr_gaps:
                    # these gaps won't vanish, so we can crop the candidate list here
                    new_list_start = j + 1
                    break
                else:
                    continue

            # check time difference - if time too long, we can remove the candidate
            cand_start_time = user_trip_df.loc[cand, "started_at"]
            if end_time - cand_start_time > max_time:
                new_list_start = len(start_candidates) - j - 1
                break

            # check whether the start-end candidate of a tour is an unknown activity
            if pd.isna(user_trip_df.loc[cand, "origin_staypoint_id"]):
                continue

            # check if endpoint of trip = start location of cand
            if staypoints is not None:
                end_start_at_same_loc = _check_same_loc(
                    user_trip_df.loc[cand, "origin_staypoint_id"],  # start stp of first trip
                    row["destination_staypoint_id"],  # destination stp of current trip
                    staypoints,
                )
            else:
                # if no locations are available, check whether the distance is smaller than max_dist
                end_start_at_same_loc = _check_max_dist(
                    user_trip_df.loc[cand, geom_col][0],  # start point of first trip
                    row[geom_col][1],  # destination point of current trip
                    max_dist,
                    crs_is_projected=crs_is_projected,
                )

            if end_start_at_same_loc:
                # Tour found!
                # collect the trips on the tour in a list
                non_gap_trip_idxs = [c for c in start_candidates[-j - 1 :] if ~np.isnan(c)]
                tour_candidate = user_trip_df[user_trip_df.index.isin(non_gap_trip_idxs)]
                tours.append(_create_tour_from_stack(tour_candidate, staypoints, max_time))

                # do not consider the other trips - one trip cannot close two tours at a time
                break

        # remove points because they are out of the time window
        start_candidates = start_candidates[new_list_start:]

    if len(tours) == 0:
        return pd.DataFrame(
            tours,
            columns=[
                "user_id",
                "started_at",
                "finished_at",
                "origin_staypoint_id",
                "destination_staypoint_id",
                "trips",
                "location_id",
            ],
        )
    tours_df = pd.DataFrame(tours)
    return tours_df


def _check_same_loc(stp1, stp2, staypoints):
    """Check whether two staypoints are at the same location

    Parameters
    ----------
    stp1 : int
        First staypoint id
    stp2 : int
        Second staypoint id
    staypoints : Trackintel staypoints
        GeoDataFrame with staypoints and also location ids

    Returns
    -------
    share_location, bool
        If True, stp1 and stp2 are at the same location
    """
    if pd.isna(stp1) or pd.isna(stp2):
        return False
    share_location = staypoints.loc[stp1, "location_id"] == staypoints.loc[stp2, "location_id"]
    return share_location


def _check_max_dist(p1, p2, max_dist, crs_is_projected=False):
    """
    Check whether two points p1, p2 are less or equal than max_dist apart

    Parameters
    --------
    p1, p2: shapely Point objects
    max_dist: int

    Returns
    ------
    dist_below_thresh: bool
        indicating whether p1 and p2 are less than max_dist apart
    """
    if crs_is_projected:
        dist = p1.distance(p2)
    else:
        dist = ti.geogr.point_distances.haversine_dist(p1.x, p1.y, p2.x, p2.y)
    dist_below_thresh = dist <= max_dist
    return dist_below_thresh


def _create_tour_from_stack(temp_tour_stack, staypoints, max_time):
    """
    Aggregate information of tour elements in a structured dictionary.

    Parameters
    ----------
    temp_tour_stack : list
        list of dictionary like elements (either pandas series or python dictionary).
        Contains all trips that will be aggregated into a tour

    Returns
    -------
    tour_dict_entry: dictionary

    """
    # this function return and empty dict if no tripleg is in the stack
    first_trip = temp_tour_stack.iloc[0]
    last_trip = temp_tour_stack.iloc[-1]

    # get location ID if available:
    if staypoints is not None:
        start_loc = staypoints.loc[first_trip["origin_staypoint_id"], "location_id"]
        # double check whether start and end location are the same
        end_loc = staypoints.loc[last_trip["destination_staypoint_id"], "location_id"]
        assert start_loc == end_loc
    else:
        # set location to NaN since not available
        start_loc = pd.NA

    # all data has to be from the same user
    assert len(temp_tour_stack["user_id"].unique()) == 1

    # double check if tour requirements are fulfilled
    assert last_trip["finished_at"] - first_trip["started_at"] <= max_time

    tour_dict_entry = {
        "user_id": first_trip["user_id"],
        "started_at": first_trip["started_at"],
        "finished_at": last_trip["finished_at"],
        "origin_staypoint_id": first_trip["origin_staypoint_id"],
        "destination_staypoint_id": last_trip["destination_staypoint_id"],
        "trips": list(temp_tour_stack.index),
        "location_id": start_loc,
    }

    return tour_dict_entry
