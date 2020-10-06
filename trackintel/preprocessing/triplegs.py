from shapely.geometry import LineString
from simplification.cutil import simplify_coords#, simplify_coordsvw
import copy
import ast

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

def generate_trips(trpls, stps):
    """

    Returns
    -------

    """

    # Definition of trips: All movement and waiting between two activities
    # assert staypoints have activities
    # trip id counter

    # fields for trips = [id, user_id, started_at, finished_at, origin, destination]

    # create table with relevant information from triplegs and staypoints.
    # sort table by start time

    # loop user_id

        # get next staypoint ID

        # get all triplegs between old and new ID

        # check temporal gaps





    # Todo: What about first and last trip?


    # sort triplegs and staypoints
    #
    # sort by index again
    # return

