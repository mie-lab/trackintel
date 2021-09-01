import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import warnings

from trackintel.geogr.distances import meters_to_decimal_degrees
from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig
from trackintel.geogr.distances import check_gdf_planar


def plot_locations(
    locations,
    out_filename=None,
    radius=150,
    positionfixes=None,
    staypoints=None,
    staypoints_radius=100,
    plot_osm=False,
    axis=None,
):
    """Plot locations (optionally to a file).

    Optionally, you can specify several other datasets to be plotted beneath the locations.

    Parameters
    ----------
    locations : GeoDataFrame (as trackintel locations)
        The locations to plot.

    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float, default 150 (meter)
        The radius in meter with which circles around locations should be drawn.

    positionfixes : GeoDataFrame (as trackintel positionfixes), optional
        If available, some positionfixes that can additionally be plotted.

    staypoints : GeoDataFrame (as trackintel staypoints), optional
        If available, some staypoints that can additionally be plotted.

    staypoints_radius : float, default 100 (meter)
        The radius in meter with which circles around staypoints should be drawn.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot
        below the staypoints.

    axis : matplotlib.pyplot.Artist, optional
        axis on which to draw the plot

    Examples
    --------
    >>> locs.as_locations.plot('output.png', radius=200, positionfixes=pfs, staypoints=sp, plot_osm=True)
    """
    if axis is None:
        _, ax = regular_figure()
    else:
        ax = axis
    _, locations = check_gdf_planar(locations, transform=True)

    if staypoints is not None:
        staypoints.as_staypoints.plot(radius=staypoints_radius, positionfixes=positionfixes, plot_osm=plot_osm, axis=ax)
    elif positionfixes is not None:
        positionfixes.as_positionfixes.plot(plot_osm=plot_osm, axis=ax)
    elif plot_osm:
        west = locations["center"].x.min() - 0.03
        east = locations["center"].x.max() + 0.03
        north = locations["center"].y.max() + 0.03
        south = locations["center"].y.min() - 0.03
        plot_osm_streets(north, south, east, west, ax)

    center_latitude = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    radius = meters_to_decimal_degrees(radius, center_latitude)
    for pt in locations.to_dict("records"):
        circle = mpatches.Circle((pt["center"].x, pt["center"].y), radius, facecolor="none", edgecolor="r", zorder=4)
        ax.add_artist(circle)
    ax.set_aspect("equal", adjustable="box")

    if out_filename is not None:
        save_fig(out_filename, formats=["png"])
    else:
        plt.show()
