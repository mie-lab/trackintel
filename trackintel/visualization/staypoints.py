import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig


def plot_staypoints(staypoints, out_filename=None, radius=None, positionfixes=None, plot_osm=False):
    """Plots staypoints (optionally to a file). You can specify the radius with which 
    each staypoint should be drawn, as well as if underlying positionfixes and OSM streets
    should be drawn.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to plot.
    
    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float, optional
        The radius with which circles around staypoints should be drawn.

    positionfixes : GeoDataFrame, optional
        If available, some positionfixes that can additionally be plotted.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.

    Examples
    --------
    >>> df.as_staypoints.plot('output.png', radius=10, positionfixes=pdf, plot_osm=True)
    """
    _, ax = regular_figure()
    name_geocol = staypoints.geometry.name

    if positionfixes is not None:
        west = positionfixes.geometry.x.min() - 0.01
        east = positionfixes.geometry.x.max() + 0.01
        north = positionfixes.geometry.y.max() + 0.01
        south = positionfixes.geometry.y.min() - 0.01
    else:
        west = staypoints.geometry.x.min() - 0.03
        east = staypoints.geometry.x.max() + 0.03
        north = staypoints.geometry.y.max() + 0.03
        south = staypoints.geometry.y.min() - 0.03

    if plot_osm:
        plot_osm_streets(north, south, east, west, ax)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if radius is None:
        radius = 5
    for pt in staypoints.to_dict('records'):
        circle = mpatches.Circle((pt[name_geocol].x, pt[name_geocol].y), radius,
                                 facecolor='none', edgecolor='g', zorder=3)
        ax.add_artist(circle)

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])

    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()