from shapely.geometry import LineString
from simplification.cutil import simplify_coords#, simplify_coordsvw
import copy
import ast

def smoothen_triplegs(triplegs, method='douglas-peucker', epsilon = 1.0):
    """Not implemented.
    """
    input_copy = copy.deepcopy(triplegs)
    input_copy.geom = [LineString(ast.literal_eval(str(simplify_coords(input_copy.geom[i].coords, epsilon))))
                       for i in range(len(input_copy.geom))]
    return input_copy