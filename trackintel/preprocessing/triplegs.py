import ast
import copy
import datetime

import numpy as np
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


def _check_if_temp_trip_stack_has_tripleg(temp_trip_stack):
    has_tripleg = False
    for row in temp_trip_stack:
        if row['type'] == 'tripleg':
            has_tripleg = True
            break

    return has_tripleg


def _create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity, trip_id_counter):
    # this function return and empty dict if no tripleg is in the stack

    first_trip_element = temp_trip_stack[0]
    last_trip_element = temp_trip_stack[-1]

    # every trip needs at least 1 tripleg
    if not _check_if_temp_trip_stack_has_tripleg(temp_trip_stack):
        return {}

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


def generate_trips(tpls_input, stps_input, gap_threshold=15, id_offset=0):
    """

    Returns
    -------

    """

    # if there are several chained activities without a tripleg in between, all activities are added to the trip.

    # Definition of trips: All movement and waiting between two activities

    # assert staypoints have activities
    assert 'activity' in stps_input.columns, "staypoints need the column 'activities' \
                                         to be able to generate trips"

    # trip id counter
    trip_id_counter = id_offset

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    stps = stps_input.copy()
    tpls['type'] = 'tripleg'
    stps['type'] = 'staypoint'

    # create table with relevant information from triplegs and staypoints.
    spts_tpls = stps[['started_at', 'finished_at', 'user_id', 'id', 'type', 'activity']].append(
        tpls[['started_at', 'finished_at', 'user_id', 'id', 'type']])

    # transform nan to bool
    spts_tpls['activity'] = (spts_tpls['activity'] == True)

    spts_tpls.sort_values(by='started_at',
                          inplace=True)  # TODO: make sure that this stays sorted after filtering for users
    spts_tpls = spts_tpls.sort_values(by=['user_id', 'started_at'])
    spts_tpls['started_at_next'] = spts_tpls['started_at'].shift(-1)
    spts_tpls['activity_next'] = spts_tpls['activity'].shift(-1)

    spts_tpls.to_csv('spts_tpls.csv')
    trips_of_user_list = []
    for user_id_this in spts_tpls['user_id'].unique():

        spts_tpls_this = spts_tpls[spts_tpls['user_id'] == user_id_this]
        spts_tpls_this_debug = pd.DataFrame(spts_tpls_this)

        trip_started = False
        origin_activity = None
        destination_activity = None
        assert (spts_tpls_this['started_at'].is_monotonic)  # this is expensive and should be replaced
        temp_trip_stack = []
        before_first_trip = True

        for _, row in spts_tpls_this.iterrows():

            # skip all non-activities before the first trip
            # if not before_first_trip:
            #     pass
            # elif  before_first_trip and not row['activity']:  # check if this works with nan
            #     continue
            # else:
            #     before_first_trip = False

            if before_first_trip:
                if not row['activity']:
                    continue



            # if we are currently not recording a new trip, then skip everything until the next entry is not an activity
            if origin_activity is None and not row['activity']:
                continue
            elif row['activity'] and row['activity_next']:
                if not trip_started:
                    origin_activity = None
                    continue

            # check if activity
            if origin_activity is not None and row['activity'] == True:
                destination_activity = row
                trips_of_user_list.append(_create_trip_from_stack(temp_trip_stack, origin_activity,
                                                                  destination_activity, trip_id_counter))
                trip_id_counter += 1
                origin_activity = destination_activity
                destination_activity = None
                temp_trip_stack = list()
                trip_started = False

            elif row['activity']:
                origin_activity = row

            # check if gap
            elif row['started_at_next'] - row['finished_at'] > datetime.timedelta(minutes=gap_threshold):
                # generate trip, start new trip
                destination_activity = {'user_id': row['user_id'], 'activity': True, 'id': np.nan}

                trips_of_user_list.append(_create_trip_from_stack(temp_trip_stack, origin_activity,
                                                                  destination_activity, trip_id_counter))
                trip_id_counter += 1
                origin_activity = destination_activity
                destination_activity = None
                temp_trip_stack = list()
                trip_started = False

            else:
                temp_trip_stack.append(row)
                trip_started = True
                # add to stack

    # Todo write IDs back to tripleg and staypoints
    # Todo create good test data set for comparison

    trips = pd.DataFrame(trips_of_user_list)
    return trips
    # fields for trips = [id, user_id, started_at, finished_at, origin, destination]

    # Todo: What about first and last trip?

    # sort triplegs and staypoints
    #
    # sort by index again
    # return
