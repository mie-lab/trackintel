import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox

from matplotlib.collections import LineCollection

from trackintel.visualization.util import regular_figure, save_fig


def plot_positionfixes(positionfixes, out_filename):
    """Plots positionfixes to a file.

    Parameters
    ----------
    positionfixes : float
        The positionfixes to plot.
    
    out_filename : float
        The file to plot to.
    """
    _, ax = regular_figure()
    positionfixes.plot(ax=ax)
    save_fig(out_filename, formats=['png'])


def plot_staypoints(staypoints, out_filename, radius=None, positionfixes=None, plot_osm=False):
    """Plots staypoints to a file.

    Parameters
    ----------
    staypoints : float
        The positionfixes to plot.
    
    out_filename : float
        The file to plot to.

    radius : float
        The radius with which circles around staypoints should be drawn.

    positionfixes : df
        If available, some positionfixes that can additionally be plotted.

    plot_osm : bool
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.
    """
    _, ax = regular_figure()

    if plot_osm:
        if positionfixes is not None:
            west = positionfixes['longitude'].min()
            east = positionfixes['longitude'].max()
            north = positionfixes['latitude'].max()
            south = positionfixes['latitude'].min()
        else:
            west = staypoints['longitude'].min() - 0.03
            east = staypoints['longitude'].max() + 0.03
            north = staypoints['latitude'].max() + 0.03
            south = staypoints['latitude'].min() - 0.03

        G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
        lines = []
        for u, v, data in G.edges(keys=False, data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                x1 = G.nodes[u]['x']
                y1 = G.nodes[u]['y']
                x2 = G.nodes[v]['x']
                y2 = G.nodes[v]['y']
                line = [(x1, y1), (x2, y2)]
                lines.append(line)
        lc = LineCollection(lines, colors='#999999', linewidths=0.5, alpha=1, zorder=1)
        ax.add_collection(lc)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if radius is None:
        radius = 5
    for pt in staypoints.to_dict('records'):
        circle = mpatches.Circle((pt['longitude'], pt['latitude']), radius, 
                                  facecolor='none', edgecolor='g', zorder=3)
        ax.add_artist(circle)
    save_fig(out_filename, formats=['png'])
