import matplotlib.pyplot as plt
import warnings
from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig
from trackintel.geogr.distances import check_gdf_planar


def plot_positionfixes(positionfixes, out_filename=None, plot_osm=False, axis=None):
    """Plots positionfixes (optionally to a file).

    If you specify ``plot_osm=True`` this will use ``osmnx`` to plot streets
    below the positionfixes. Depending on the extent of your data, this might
    take a long time. The data gets transformed to wgs84 for the plotting.

    Parameters
    ----------
    positionfixes : GeoDataFrame (as trackintel positionfixes)
        The positionfixes to plot.

    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot
        below the staypoints.

    axis : matplotlib.pyplot.Artist, optional
        axis on which to draw the plot

    Examples
    --------
    >>> pfs.as_positionfixes.plot('output.png', plot_osm=True)
    """
    if axis is None:
        _, ax = regular_figure()
    else:
        ax = axis
    _, positionfixes = check_gdf_planar(positionfixes, transform=True)

    if plot_osm:
        west = positionfixes.geometry.x.min()
        east = positionfixes.geometry.x.max()
        north = positionfixes.geometry.y.max()
        south = positionfixes.geometry.y.min()
        plot_osm_streets(north, south, east, west, ax)

    positionfixes.plot(ax=ax, markersize=0.5, zorder=2)
    ax.set_aspect("equal", adjustable="box")

    if out_filename is not None:
        save_fig(out_filename, formats=["png"])
    elif axis is None:
        plt.show()
