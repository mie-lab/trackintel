import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig


def plot_center_of_locations(locations, out_filename=None, radius=None, positionfixes=None, 
                          staypoints=None, staypoints_radius=None, plot_osm=False):
    """Plots locations (optionally to a file). Optionally, you can specify several other
    datasets to be plotted beneath the locations.

    Parameters
    ----------
    locations : GeoDataFrame
        The locations to plot.
    
    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float, optional
        The radius with which circles around locations should be drawn.

    positionfixes : GeoDataFrame, optional
        If available, some positionfixes that can additionally be plotted.

    staypoints : GeoDataFrame, optional
        If available, some staypoints that can additionally be plotted.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.

    Examples
    --------
    >>> df.as_locations.plot('output.png', radius=10, positionfixes=pdf, 
    >>>                   staypoints=spf, staypoints_radius=8, plot_osm=True)
    """
    _, ax = regular_figure()

    if plot_osm:
        if positionfixes is not None:
            west = positionfixes.geometry.x.min()
            east = positionfixes.geometry.x.max()
            north = positionfixes.geometry.y.max()
            south = positionfixes.geometry.y.min()
        else:
            west = locations['center'].x.min() - 0.03
            east = locations['center'].x.max() + 0.03
            north = locations['center'].y.max() + 0.03
            south = locations['center'].y.min() - 0.03
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

    if radius is None:
        radius = 5
    for pt in locations.to_dict('records'):
        circle = mpatches.Circle((pt['center'].x, pt['center'].y), radius, 
                                  facecolor='none', edgecolor='r', zorder=4)
        ax.add_artist(circle)
    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    else:
        plt.show()