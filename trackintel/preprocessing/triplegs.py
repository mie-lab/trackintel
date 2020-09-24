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

def generate_trips():
    """

    Returns
    -------

    """

    # Todo: What about first and last trip?

    # sort triplegs and staypoints
    #


