import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import warnings

from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig
from trackintel.geogr.distances import check_gdf_crs


def plot_triplegs(
    triplegs, out_filename=None, positionfixes=None, staypoints=None, staypoints_radius=100, plot_osm=False, axis=None
):
    """Plot triplegs (optionally to a file).

    You can specify several other datasets to be plotted beneath the triplegs,
    as well as if the OSM streets should be drawn. The data gets transformed to
    wgs84 for the plotting.

    Parameters
    ----------
    triplegs : GeoDataFrame (as trackintel triplegs)
        The triplegs to plot.

    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    positionfixes : GeoDataFrame (as trackintel positionfixes), optional
        If available, some positionfixes that can additionally be plotted.

    staypoints : GeoDataFrame (as trackintel staypoints), optional
        If available, some staypoints that can additionally be plotted.

    staypoints_radius : float, default 100 (meter)
        The radius in meter with which circles around staypoints should be drawn.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot
        below the triplegs.

    axis : matplotlib.pyplot.Artist, optional
        axis on which to draw the plot

    Example
    -------
    >>> tpls.as_triplegs.plot('output.png', positionfixes=pfs, staypoints=stps, plot_osm=True)
    """
    if axis is None:
        _, ax = regular_figure()
    else:
        ax = axis
    _, triplegs = check_gdf_crs(triplegs, transform=True)

    if staypoints is not None:
        staypoints.as_staypoints.plot(radius=staypoints_radius, positionfixes=positionfixes, plot_osm=plot_osm, axis=ax)
    elif positionfixes is not None:
        positionfixes.as_positionfixes.plot(plot_osm=plot_osm, axis=ax)
    elif plot_osm:
        triplegs_bounds = triplegs.bounds
        west = min(triplegs_bounds.minx) - 0.03  # TODO: maybe a relative value instead of 0.03
        east = max(triplegs_bounds.maxx) + 0.03
        north = max(triplegs_bounds.maxy) + 0.03
        south = min(triplegs_bounds.miny) - 0.03
        plot_osm_streets(north, south, east, west, ax)

    triplegs.plot(ax=ax, cmap="viridis")
    ax.set_aspect("equal", adjustable="box")

    if out_filename is not None:
        save_fig(out_filename, formats=["png"])
    else:
        plt.show()
