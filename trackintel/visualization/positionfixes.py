import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trackintel.visualization.util import regular_figure, save_fig
from trackintel.visualization.osm import plot_osm_streets


def plot_positionfixes(positionfixes, out_filename=None, plot_osm=False):
    """Plots positionfixes (optionally to a file).

    Parameters
    ----------
    positionfixes : GeoDataFrame
        The positionfixes to plot.
    
    out_filename : str
        The file to plot to, if this is not set, the plot will simply be shown.

    plot_osm : bool
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.
    """
    _, ax = regular_figure()

    if plot_osm:
        west = positionfixes['longitude'].min()
        east = positionfixes['longitude'].max()
        north = positionfixes['latitude'].max()
        south = positionfixes['latitude'].min()
        plot_osm_streets(north, south, east, west, ax)

    positionfixes.plot(ax=ax, markersize=0.5)
    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()
