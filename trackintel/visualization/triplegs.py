import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trackintel.visualization.util import regular_figure, save_fig
from trackintel.visualization.osm import plot_osm_streets


def plot_triplegs(triplegs, out_filename=None, positionfixes=None, staypoints=None, 
                  staypoints_radius=None, plot_osm=False):
    """Plots triplegs (optionally to a file).

    Parameters
    ----------
    triplegs : GeoDataFrame
        The triplegs to plot.
    
    out_filename : str
        The file to plot to, if this is not set, the plot will simply be shown.

    positionfixes : GeoDataFrame
        If available, some positionfixes that can additionally be plotted.

    plot_osm : bool
        If this is set to True, it will download an OSM street network and plot 
        below the triplegs.
    """
    _, ax = regular_figure()

    if plot_osm:
        if positionfixes is not None:
            west = positionfixes['geom'].x.min()
            east = positionfixes['geom'].x.max()
            north = positionfixes['geom'].y.max()
            south = positionfixes['geom'].y.min()
        else:
            west = triplegs['geom'].x.min() - 0.03
            east = triplegs['geom'].x.max() + 0.03
            north = triplegs['geom'].y.max() + 0.03
            south = triplegs['geom'].y.min() - 0.03
        plot_osm_streets(north, south, east, west, ax)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if staypoints is not None:
        if staypoints_radius is None:
            staypoints_radius = 3
        for pt in staypoints.to_dict('records'):
            circle = mpatches.Circle((pt['geom'].x, pt['geom'].y), staypoints_radius, 
                                      facecolor='none', edgecolor='c', zorder=3)
            ax.add_artist(circle)

    triplegs.plot(ax=ax, cmap='viridis')

    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()
