import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trackintel.visualization.util import regular_figure, save_fig
from trackintel.visualization.osm import plot_osm_streets


def plot_staypoints(staypoints, out_filename=None, radius=None, positionfixes=None, plot_osm=False):
    """Plots staypoints (optionally to a file).

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to plot.
    
    out_filename : str
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float
        The radius with which circles around staypoints should be drawn.

    positionfixes : GeoDataFrame
        If available, some positionfixes that can additionally be plotted.

    plot_osm : bool
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.
    """
    _, ax = regular_figure()

    if positionfixes is not None:
        west = positionfixes['geom'].x.min() - 0.01
        east = positionfixes['geom'].x.max() + 0.01
        north = positionfixes['geom'].y.max() + 0.01
        south = positionfixes['geom'].y.min() - 0.01
    else:
        west = staypoints['geom'].x.min() - 0.03
        east = staypoints['geom'].x.max() + 0.03
        north = staypoints['geom'].y.max() + 0.03
        south = staypoints['geom'].y.min() - 0.03

    if plot_osm:
        plot_osm_streets(north, south, east, west, ax)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if radius is None:
        radius = 5
    for pt in staypoints.to_dict('records'):
        circle = mpatches.Circle((pt['geom'].x, pt['geom'].y), radius, 
                                  facecolor='none', edgecolor='g', zorder=3)
        ax.add_artist(circle)

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])

    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()