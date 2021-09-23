import logging

import osmnx as ox
from matplotlib.collections import LineCollection
from networkx.exception import NetworkXPointlessConcept


def plot_osm_streets(north, south, east, west, ax):
    """Plots with osmnx OpenStreetMap streets onto an axis.

    Parameters
    ----------
    north : float
        The northernmost coordinate (to retrieve OSM data for).

    south : float
        The southernmost coordinate.

    east : float
        The easternmost coordinate.

    west : float
        The westernmost coordinate.

    ax : matplotlib.pyplot.Artist, optional
        Axis on which to draw the plot.

    Examples
    --------
    >>> ti.visualization.osm.plot_osm_street(47.392, 47.364, 8.557, 8.509, ax)
    """
    try:
        G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        lines = []
        for u, v, data in G.edges(keys=False, data=True):
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                lines.append(list(zip(xs, ys)))
            else:
                x1 = G.nodes[u]["x"]
                y1 = G.nodes[u]["y"]
                x2 = G.nodes[v]["x"]
                y2 = G.nodes[v]["y"]
                line = [(x1, y1), (x2, y2)]
                lines.append(line)
        lc = LineCollection(lines, colors="#999999", linewidths=0.5, alpha=1, zorder=0)
        ax.add_collection(lc)
    except NetworkXPointlessConcept as e:
        logging.warn("Plotting of OSM graph failed: %s", e)
