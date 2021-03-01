import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from trackintel.geogr.distances import meters_to_decimal_degrees
from trackintel.visualization.osm import plot_osm_streets
from trackintel.visualization.util import regular_figure, save_fig


def plot_staypoints(staypoints, out_filename=None, radius=None, positionfixes=None, plot_osm=False, axis=None):
    """Plots staypoints (optionally to a file). You can specify the radius with which 
    each staypoint should be drawn, as well as if underlying positionfixes and OSM streets
    should be drawn. The data gets transformed to wgs84 for the plotting.

    Parameters
    ----------
    staypoints : GeoDataFrame
        The staypoints to plot.
    
    out_filename : str, optional
        The file to plot to, if this is not set, the plot will simply be shown.

    radius : float, optional
        The radius in meter with which circles around staypoints should be drawn.

    positionfixes : GeoDataFrame, optional
        If available, some positionfixes that can additionally be plotted.

    plot_osm : bool, default False
        If this is set to True, it will download an OSM street network and plot 
        below the staypoints.

    axis : matplotlib.pyplot.Artist, optional
        axis on which to draw the plot

    Examples
    --------
    >>> df.as_staypoints.plot('output.png', radius=10, positionfixes=pdf, plot_osm=True)
    """
    if axis is None:
        _, ax = regular_figure()
    else:
        ax = axis
    name_geocol = staypoints.geometry.name

    crs_wgs84 = 'EPSG:4326'
    if staypoints.crs is None:
        Warning("Coordinate System (CRS) is not set, default to WGS84.")
        staypoints.crs = crs_wgs84
    elif staypoints.crs != crs_wgs84:
        staypoints = staypoints.to_crs(crs_wgs84)

    if positionfixes is not None:
        ax = positionfixes.as_positionfixes().plot(plot_osm=plot_osm, axis=ax)
    else:
        west = staypoints.geometry.x.min() - 0.03
        east = staypoints.geometry.x.max() + 0.03
        north = staypoints.geometry.y.max() + 0.03
        south = staypoints.geometry.y.min() - 0.03
        plot_osm_streets(north, south, east, west, ax)
        ax.set_xlim([west, east])
        ax.set_ylim([south, north])

    if radius is None:
        radius = 100
    ylim = ax.get_ylim()
    center_angle = (ylim[0] + ylim[1]) / 2
    radius = meters_to_decimal_degrees(radius, center_angle)
    for pt in staypoints.to_dict('records'):
        circle = mpatches.Circle((pt[name_geocol].x, pt[name_geocol].y), radius,
                                 facecolor='none', edgecolor='g', zorder=3)
        ax.add_artist(circle)

    if out_filename is not None:
        save_fig(out_filename, formats=['png'])
    elif axis is None:
        plt.show()
    return ax
