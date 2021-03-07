import copy
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

def smoothen_triplegs(triplegs, tolerance=1.0, preserve_topology=True):
    """
    Reduce number of points while retaining structure of tripleg.
    
    A wrapper function using shapely.simplify():
    https://shapely.readthedocs.io/en/stable/manual.html#object.simplify
    
    Parameters
    ----------
    triplegs: GeoDataFrame (as trackintel triplegs)
        triplegs to be simplified
        
    tolerance: float, default 1.0
        a higher tolerance removes more points; the units of tolerance are the same as the 
        projection of the input geometry
    
    preserve_topology: bool, default True
        whether to preserve topology. If set to False the Douglas-Peucker algorithm is used.
    
    Returns
    -------
    ret_tpls: GeoDataFrame (as trackintel triplegs)
        The simplified triplegs GeoDataFrame
    """
    ret_tpls = triplegs.copy()
    origin_geom = ret_tpls.geom
    simplified_geom = origin_geom.simplify(tolerance, preserve_topology=preserve_topology)
    ret_tpls.geom = simplified_geom

    return ret_tpls


def generate_trips(stps_input, tpls_input, gap_threshold=15, id_offset=0, print_progress=False):
    """Generate trips based on staypoints and triplegs.

    Parameters
    ---------- 
    stps_input : GeoDataFrame (as trackintel staypoints)
        The staypoints have to follow the standard definition for staypoints DataFrames.
                
    tpls_input : GeoDataFrame (as trackintel triplegs)
        The triplegs have to follow the standard definition for triplegs DataFrames.
                
    gap_threshold : float, default 15 (minutes)
        Maximum allowed temporal gap size in minutes. If tracking data is missing for more than 
        `gap_threshold` minutes, then a new trip begins after the gap.
                
    id_offset : int, default 0
        IDs for trips are incremented starting from this value.

    Returns
    -------
    staypoints: GeoDataFrame (as trackintel staypoints)
        the original staypoints with new columns ``[`trip_id`, `prev_trip_id`, `next_trip_id`]``.
        
    triplegs: GeoDataFrame (as trackintel triplegs)
        The original triplegs with a new column ``[`trip_id`]``.
    
    trips: GeoDataFrame (as trackintel trips)
        The generated trips. 

    Notes
    -----
    Trips are an aggregation level in transport planning that summarize all movement and all non-essential actions
    (e.g., waiting) between two relevant activities.
    The function returns altered versions of the input staypoints and triplegs. Staypoints receive the fields
    [`trip_id` `prev_trip_id` and `next_trip_id`], triplegs receive the field [`trip_id`].
    The following assumptions are implemented
    
        - All movement before the first and after the last activity is omitted
        - If we do not record a person for more than `gap_threshold` minutes, \
            we assume that the person performed an activity in the recording gap and split the trip at the gap.
        - Trips that start/end in a recording gap can have an unknown origin/destination
        - There are no trips without a (recored) tripleg

    Examples
    --------
    >>> staypoints, triplegs, trips = generate_trips(staypoints, triplegs)
    """
    assert 'activity' in stps_input.columns, "staypoints need the column 'activities' \
                                         to be able to generate trips"

    # we copy the input because we need to add a temporary column
    tpls = tpls_input.copy()
    spts = stps_input.copy()

    tpls['type'] = 'tripleg'
    spts['type'] = 'staypoint'

    # create table with relevant information from triplegs and staypoints.
    spts_tpls = spts[['started_at', 'finished_at', 'user_id', 'type', 'activity']].append(
        tpls[['started_at', 'finished_at', 'user_id', 'type']])

    # create ID field from index
    spts_tpls['id'] = spts_tpls.index

    # transform nan to bool
    spts_tpls['activity'] = spts_tpls['activity'] == True

    spts_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)
    spts_tpls['started_at_next'] = spts_tpls['started_at'].shift(-1)
    spts_tpls['activity_next'] = spts_tpls['activity'].shift(-1)
    
    if print_progress:
        tqdm.pandas(desc='User trip generation')
        trips = spts_tpls.groupby(['user_id'], 
                                  group_keys=False, 
                                  as_index=False).progress_apply(_generate_trips_user, gap_threshold=gap_threshold).reset_index(drop=True)
    else:
        trips = spts_tpls.groupby(['user_id'], 
                                  group_keys=False, 
                                  as_index=False).apply(_generate_trips_user, gap_threshold=gap_threshold).reset_index(drop=True)
    
    # index management
    trips['id'] = np.arange(len(trips)) + id_offset
    trips.set_index('id', inplace=True)
    
    # assign trip_id to tpls
    trip2tpl_map = trips[['tpls']].to_dict()['tpls']
    ls = []
    for key, values in trip2tpl_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[tpls.index.name, 'trip_id']).set_index(tpls.index.name)
    tpls = tpls.join(temp, how='left')
    
    # assign trip_id to spts, for non-activity spts
    trip2spt_map = trips[['spts']].to_dict()['spts']
    ls = []
    for key, values in trip2spt_map.items():
        for value in values:
            ls.append([value, key])
    temp = pd.DataFrame(ls, columns=[spts.index.name, 'trip_id']).set_index(spts.index.name)
    spts = spts.join(temp, how='left')
    
    # assign prev_trip_id to spts
    temp = trips[['destination_staypoint_id']].copy()
    temp.rename(columns={"destination_staypoint_id":spts.index.name}, inplace=True)
    temp.index.name = "prev_trip_id"
    temp = temp.reset_index().set_index(spts.index.name)
    spts = spts.join(temp, how ='left')
    
    # assign next_trip_id to spts
    temp = trips[['origin_staypoint_id']].copy()
    temp.rename(columns={"origin_staypoint_id":spts.index.name}, inplace=True)
    temp.index.name = "next_trip_id"
    temp = temp.reset_index().set_index(spts.index.name)
    spts = spts.join(temp, how ='left')
    
    # final cleaning
    tpls.drop(columns=['type'], inplace=True)
    spts.drop(columns=['type'], inplace=True)
    trips.drop(columns = ['tpls', 'spts'], inplace=True)
    
    ## dtype consistency 
    # trips id (generated by this function) should be int64
    trips.index = trips.index.astype('int64')
    # trip id of spts and tpls can only be in Int64 (missing values)
    spts['trip_id'] = spts['trip_id'].astype('Int64')
    spts['prev_trip_id'] = spts['prev_trip_id'].astype('Int64')
    spts['next_trip_id'] = spts['next_trip_id'].astype('Int64')
    tpls['trip_id'] = tpls['trip_id'].astype('Int64')
    
    # user_id of trips should be the same as tpls
    trips['user_id'] = trips['user_id'].astype(tpls['user_id'].dtype)
    
    return spts, tpls, trips


def _generate_trips_user(df, gap_threshold):
    # function called after groupby: should only contain records of one user
    user_id = df['user_id'].unique()
    assert len(user_id) == 1
    user_id = user_id[0]

    unknown_activity = {'user_id': user_id, 'activity': True, 'id': np.nan}
    origin_activity = unknown_activity
    temp_trip_stack = []
    in_trip = False
    trip_ls = []

    for _, row in df.iterrows():
        
        
        # check if we can start a new trip
        # (we make sure that we start the trip with the most recent activity)
        if in_trip is False:
            # If there are several activities in a row, we skip until the last one
            if row['activity'] and row['activity_next']:
                continue

            # if this is the last activity before the trip starts, reset the origin
            elif row['activity']:
                origin_activity = row
                in_trip = True
                continue

            # if for non-activities we simply start the trip
            else:
                in_trip = True
                
        if in_trip is True:
            # during trip generation/recording

            # check if trip ends regularly
            if row['activity'] is True:

                # if there are no triplegs in the trip, set the current activity as origin and start over
                if not _check_trip_stack_has_tripleg(temp_trip_stack):
                    origin_activity = row
                    temp_trip_stack = list()
                    in_trip = True

                else:
                    # record trip
                    destination_activity = row
                    trip_ls.append(_create_trip_from_stack(temp_trip_stack, origin_activity,destination_activity))

                    # set values for next trip
                    if row['started_at_next'] - row['finished_at'] > datetime.timedelta(minutes=gap_threshold):
                        # if there is a gap after this trip the origin of the next trip is unknown
                        origin_activity = unknown_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_trip = False

                    else:
                        # if there is no gap after this trip the origin of the next trip is the destination of the
                        # current trip
                        origin_activity = destination_activity
                        destination_activity = None
                        temp_trip_stack = list()
                        in_trip = False

            # check if gap during the trip
            elif row['started_at_next'] - row['finished_at'] > datetime.timedelta(minutes=gap_threshold):
                # in case of a gap, the destination of the current trip and the origin of the next trip
                # are unknown.

                # add current item to trip
                temp_trip_stack.append(row)

                # if the trip has no recored triplegs, we do not generate the current trip.
                if not _check_trip_stack_has_tripleg(temp_trip_stack):
                    origin_activity = unknown_activity
                    in_trip = True
                    temp_trip_stack = list()

                else:
                    # add tripleg to trip, generate trip, start new trip with unknown origin
                    destination_activity = unknown_activity

                    trip_ls.append(_create_trip_from_stack(temp_trip_stack, origin_activity,destination_activity))
                    origin_activity = unknown_activity
                    destination_activity = None
                    temp_trip_stack = list()
                    in_trip = True

            else:
                temp_trip_stack.append(row)
    
    # if user ends generate last trip with unknown destination
    if (len(temp_trip_stack) > 0) and (_check_trip_stack_has_tripleg(temp_trip_stack)):
        destination_activity = unknown_activity
        trip_ls.append(_create_trip_from_stack(temp_trip_stack, origin_activity,destination_activity,))
    
    # print(trip_ls)
    trips = pd.DataFrame(trip_ls)
    return trips


def _check_trip_stack_has_tripleg(temp_trip_stack):
    """
    Check if a trip has at least 1 tripleg.
    
    Parameters
    ----------
    temp_trip_stack : list
        list of dictionary like elements (either pandas series or python dictionary). 
        Contains all elements that will be aggregated into a trip

    Returns
    -------
    has_tripleg: Bool
    """
    has_tripleg = False
    for row in temp_trip_stack:
        if row['type'] == 'tripleg':
            has_tripleg = True
            break

    return has_tripleg


def _create_trip_from_stack(temp_trip_stack, origin_activity, destination_activity):
    """
    Aggregate information of trip elements in a structured dictionary.

    Parameters
    ----------
    temp_trip_stack : list
        list of dictionary like elements (either pandas series or python dictionary). 
        Contains all elements that will be aggregated into a trip
        
    origin_activity : dictionary like
        Either dictionary or pandas series
        
    destination_activity : dictionary like
        Either dictionary or pandas series

    Returns
    -------
    trip_dict_entry: dictionary

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

    trip_dict_entry = {'user_id': origin_activity['user_id'],
                       'started_at': first_trip_element['started_at'],
                       'finished_at': last_trip_element['finished_at'],
                       'origin_staypoint_id': origin_activity['id'],
                       'destination_staypoint_id': destination_activity['id'],
                       'tpls': [tripleg['id'] for tripleg in temp_trip_stack if tripleg['type'] == 'tripleg'],
                       'spts': [tripleg['id'] for tripleg in temp_trip_stack if tripleg['type'] == 'staypoint']}
    
    return trip_dict_entry
