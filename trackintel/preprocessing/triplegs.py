import ast
import copy
import datetime
import pandas as pd
from shapely.geometry import LineString
from simplification.cutil import simplify_coords  # , simplify_coordsvw

def smoothen_triplegs(triplegs, method='douglas-peucker', epsilon = 1.0):
    """reduces number of points while retaining structure of tripleg
    Parameters
    ----------
    triplegs: shapely file
        triplegs to be reduced
    method: method used to smoothen
        only one method available so far
    epsilon: float
        slack parameter, higher epsilon means removing more points
    """
    input_copy = copy.deepcopy(triplegs)
    input_copy.geom = [LineString(ast.literal_eval(str(simplify_coords(input_copy.geom[i].coords, epsilon))))
                       for i in range(len(input_copy.geom))]
    return input_copy

def _temp_trip_stack_has_tripleg(temp_trip_stack):
    """
    Check if a trip has at least 1 tripleg
    Parameters
    ----------
        temp_trip_stack : list
                    list of dictionary like elements (either pandas series or python dictionary). Contains all elements
                    that will be aggregated into a trip

    Returns
    -------
    Bool
    """

    has_tripleg = False
    for row in temp_trip_stack:
        if row['type'] == 'tripleg':
            has_tripleg = True
            break

    return has_tripleg


def _create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity, trip_id_counter):
    """
    Aggregate information of trip elements in a structured dictionary

    Parameters
    ----------
    temp_trip_stack : list
                    list of dictionary like elements (either pandas series or python dictionary). Contains all elements
                    that will be aggregated into a trip
    origin_activity : dictionary like
                    Either dictionary or pandas series
    destination_activity : dictionary like
                    Either dictionary or pandas series
    trip_id_counter : int
            current trip id

    Returns
    -------
    dictionary

    """

    # this function return and empty dict if no tripleg is in the stack
    first_trip_element = temp_trip_stack[0]
    last_trip_element = temp_trip_stack[-1]

    # all data has to be from the same user
    assert origin_activity['user_id'] == last_trip_element['user_id']

    # double check if trip requirements are fulfilled
    assert origin_activity['activity'] == True
    assert destination_activity['activity'] == True
    assert first_trip_element['activity'] == False

    trip_dict_entry = {'id': trip_id_counter,
                       'user_id': origin_activity['user_id'],
                       'started_at': first_trip_element['started_at'],
                       'finished_at': last_trip_element['finished_at'],
                       'origin': origin_activity['id'],
                       'destination': destination_activity['id']}

    return trip_dict_entry


def _return_ids_to_df(temp_trip_stack, origin_activity, destination_activity, spts, tpls, trip_id_counter):
    """
    Write trip ids into the staypoint and tripleg GeoDataFrames.

    Parameters
    ----------
    temp_trip_stack : list
                    list of dictionary like elements (either pandas series or python dictionary). Contains all elements
                    that will be aggregated into a trip
    origin_activity : dictionary like
                    Either dictionary or pandas series
    destination_activity : dictionary like
                    Either dictionary or pandas series
    spts : GeoDataFrame
            Staypoints
    tpls :
            Triplegs
    trip_id_counter : int
            current trip id

    Returns
    -------
    None
        Function alters the staypoint and tripleg GeoDataFrames inplace
    """

    spts.loc[spts['id'] == origin_activity['id'], ['next_trip_id']] = trip_id_counter
    spts.loc[spts['id'] == destination_activity['id'], ['prev_trip_id']] = trip_id_counter

    for row in temp_trip_stack:
        if row['type'] == 'tripleg':
            tpls.loc[tpls['id'] == row['id'], ['trip_id']] = trip_id_counter
        elif row['type'] == 'staypoint':
            spts.loc[spts['id'] == row['id'], ['trip_id']] = trip_id_counter


def generate_trips(stps_input, tpls_input, gap_threshold=15, id_offset=0, print_progress=False):
    """ Generate trips based on staypoints and triplegs

    `generate_trips` aggregates the staypoints `stps_input` and `tpls_input` into `trips` which are returned
      in a new DataFrame. The function returns new versions of `stps_input` and `tpls_input` that are identically except
    for additional id's that allow the matching between staypoints, triplegs and trips.


    Parameters
    ----------
    stps_input : GeoDataFrame
                Staypoints that are used for the trip generation
    tpls_input : GeoDataFrame
                Triplegs that are used for the trip generation
    gap_threshold : float
                Maximum allowed temporal gap size in minutes. If tracking data is misisng for more than `gap_threshold`
                minutes, then a new trip begins after the gap.
    id_offset : int
                IDs for trips are incremented starting from this value.

    Returns
    -------
    (GeoDataFrame, GeoDataFrame, GeoDataFrame)
        the tuple contains (staypoints, triplegs, trips)

    Notes
    -----
    Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities.
    The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
    [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
    The following assumptions are implemented
        - All movement before the first and after the last activity is omitted
        - If we do not record a person for more than `gap_threshold` minutes, we assume that the person performed an
            activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination
        - There are no trips without a (recored) tripleg


    """
    assert 'activity' in stps_input.columns, "staypoints need the column 'activities' \
                                         to be able to generate trips"

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    spts = stps_input.copy()

    trip_id_counter = id_offset
    tpls['type'] = 'tripleg'
    spts['type'] = 'staypoint'
    tpls['prev_trip_id'] = None
    spts['next_trip_id'] = None
    spts['trip_id'] = None

    trips_of_user_list = []
    dont_print_list = []

    # create table with relevant information from triplegs and staypoints.
    spts_tpls = spts[['started_at', 'finished_at', 'user_id', 'id', 'type', 'activity']].append(
        tpls[['started_at', 'finished_at', 'user_id', 'id', 'type']])

    # transform nan to bool
    spts_tpls['activity'] = spts_tpls['activity'] == True

    spts_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)
    spts_tpls['started_at_next'] = spts_tpls['started_at'].shift(-1)
    spts_tpls['activity_next'] = spts_tpls['activity'].shift(-1)

    for user_id_this in spts_tpls['user_id'].unique():

        spts_tpls_this = spts_tpls[spts_tpls['user_id'] == user_id_this]
        # assert (spts_tpls_this['started_at'].is_monotonic)  # this is expensive and should be replaced

        origin_activity = None
        temp_trip_stack = []
        before_first_trip = True
        in_tripleg = False

        for _, row in spts_tpls_this.iterrows():
            if print_progress:
                if trip_id_counter % 100 == 0:
                    if not trip_id_counter in dont_print_list:
                        print("trip number: {}".format(trip_id_counter))
                        dont_print_list.append(trip_id_counter)

            # skip all non-activities before the first trip
            if before_first_trip:
                if row['activity'] == False:
                    continue
                else:
                    before_first_trip = False

            # check if we can start a new trip
            if in_tripleg is False:
                # If there are several activities in a row, we skip until the last one
                if row['activity'] is True and row['activity_next'] is True:
                    continue

                # if we did not start a tripleg yet and the origin is set
                # encountering another activity means that we have to defer the
                # start of the trip by 1
                elif row['activity'] is True and origin_activity is not None:
                    origin_activity = row
                    continue

                # If the current row is an activity and `origin_activity` is not set
                # then we start a new trip. This is for example the case for the first trip
                elif row['activity'] is True and origin_activity is None:
                    origin_activity = row
                    in_tripleg = True
                    continue

                # this is the standard case after we regularly finished a tripleg
                elif row['activity'] is False and origin_activity is not None:
                    in_tripleg = True

            if in_tripleg is True:
                # during trip generation/recording

                # check if gap
                if row['started_at_next'] - row['finished_at'] > datetime.timedelta(minutes=gap_threshold):
                    # in case of a (temporal) gap we split the trip. This means we save the current trip and start
                    # in case of a gap, the destination of the current trip and the origin of the next trip
                    # are unknown. For the next trip, we set the `in_tripleg` flag to true to add everything after
                    # the gap to the new trip (e.g., we don't skip until the next activity).

                    # if the trip stack is empty, we do not generate the current trip.
                    if len(temp_trip_stack) == 0:
                        origin_activity = {'user_id': row['user_id'], 'activity': True, 'id': None}
                        in_tripleg = True
                        temp_trip_stack = list()

                    # if the trip has no recored tripleg, we do not generate the current trip.
                    elif (not _temp_trip_stack_has_tripleg(temp_trip_stack)):
                        origin_activity = {'user_id': row['user_id'], 'activity': True, 'id': None}
                        in_tripleg = True
                        temp_trip_stack = list()

                    else:
                        # generate trip, start new trip
                        destination_activity = {'user_id': row['user_id'], 'activity': True, 'id': None}

                        trips_of_user_list.append(_create_trip_from_stack(temp_trip_stack, origin_activity,
                                                                          destination_activity, trip_id_counter))
                        _return_ids_to_df(temp_trip_stack, origin_activity, destination_activity,
                                          spts, tpls, trip_id_counter)

                        trip_id_counter += 1
                        origin_activity = destination_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_tripleg = True

                # check if trip ends regularly
                elif row['activity'] is True:
                    # if there are no triplegs in the trip, set the current activity as origin and start over
                    if not _temp_trip_stack_has_tripleg(temp_trip_stack):
                        origin_activity = row
                        temp_trip_stack = list()
                        in_tripleg = True

                    # record trip and reset flags
                    else:
                        destination_activity = row
                        trips_of_user_list.append(_create_trip_from_stack(temp_trip_stack, origin_activity,
                                                                          destination_activity, trip_id_counter))
                        _return_ids_to_df(temp_trip_stack, origin_activity, destination_activity,
                                          spts, tpls, trip_id_counter)
                        trip_id_counter += 1
                        origin_activity = destination_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_tripleg = False

                else:
                    temp_trip_stack.append(row)

    trips = pd.DataFrame(trips_of_user_list)
    tpls.drop(['type'], axis=1, inplace=True)
    spts.drop(['type'], axis=1, inplace=True)

    return spts, tpls, trips