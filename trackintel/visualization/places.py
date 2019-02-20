import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trackintel.visualization.util import regular_figure, save_fig
from trackintel.visualization.osm import plot_osm_streets


def plot_places(places, out_filename=None, radius=None, positionfixes=None, 
                staypoints=None, staypoints_radius=None, plot_osm=False):
    """Plots places (optionally to a file).

    Parameters
    ----------
    places : GeoDataFrame
        The places to plot.
    
    out_filename : str
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float
        The radius with which circles around places should be drawn.

    positionfixes : GeoDataFrame
        If available, some positionfixes that can additionally be plotted.

    staypoints : GeoDataFrame
        If available, some staypoints that can additionally be plotted.

    plot_osm : bool
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.
    """
    _, ax = regular_figure()

    if plot_osm:
        if positionfixes is not None:
            west = positionfixes['geometry'].x.min()
            east = positionfixes['geometry'].x.max()
            north = positionfixes['geometry'].y.max()
            south = positionfixes['geometry'].y.min()
        else:
            west = places['geometry'].x.min() - 0.03
            east = places['geometry'].x.max() + 0.03
            north = places['geometry'].y.max() + 0.03
            south = places['geometry'].y.min() - 0.03
        plot_osm_streets(north, south, east, west, ax)

    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5, zorder=2)

    if staypoints is not None:
        if staypoints_radius is None:
            staypoints_radius = 3
        for pt in staypoints.to_dict('records'):
            circle = mpatches.Circle((pt['geometry'].x, pt['geometry'].y), staypoints_radius, 
                                      facecolor='none', edgecolor='c', zorder=3)
            ax.add_artist(circle)

    if radius is None:
        radius = 5
    for pt in places.to_dict('records'):
        circle = mpatches.Circle((pt['geometry'].x, pt['geometry'].y), radius, 
                                  facecolor='none', edgecolor='r', zorder=4)
        ax.add_artist(circle)
    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()