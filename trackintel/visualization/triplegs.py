import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig


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
            west = positionfixes.geometry.x.min()
            east = positionfixes.geometry.x.max()
            north = positionfixes.geometry.y.max()
            south = positionfixes.geometry.y.min()
        else:
            triplegs_bounds = triplegs.bounds 
            west = min(triplegs_bounds.minx) - 0.03 #TODO: maybe a relative value instead of 0.03
            east = max(triplegs_bounds.maxx) + 0.03
            north = max(triplegs_bounds.maxy) + 0.03
            south = min(triplegs_bounds.miny) - 0.03
        plot_osm_streets(north, south, east, west, ax)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if staypoints is not None:
        if staypoints_radius is None:
            staypoints_radius = 3
        for pt in staypoints.to_dict('records'):
            circle = mpatches.Circle((pt.geometry.x, pt.geometry.y), staypoints_radius,
                                     facecolor='none', edgecolor='c', zorder=3)
            ax.add_artist(circle)

    triplegs.plot(ax=ax, cmap='viridis')

    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()
