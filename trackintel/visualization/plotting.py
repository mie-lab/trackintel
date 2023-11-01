import logging
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection
from networkx.exception import NetworkXPointlessConcept
from pandas.api.types import is_datetime64_any_dtype
from pint import UnitRegistry

from trackintel.geogr import check_gdf_planar, meters_to_decimal_degrees


def a4_figsize(fig_height_mm=None, columns=2):
    """Generate sizes for a figure that fits on an A4 page.

    The sizes are taken from:
    http://www.springer.com/computer/journal/450 > Artwork and Illustrations Guidelines > Figure Placement and Size

    Parameters
    ----------
    fig_height_mm : float
        If set, uses this height for the figure. Otherwise computes one based on an aesthetic ratio.

    columns : float
        The number of columns this figure should span (1, 1.5 or 2).

    Returns
    -------
    (float, float)
        The width and height in which to plot a figure to fit on an A4 sheet.

    Examples
    --------
    >>> ti.visualization.util.a4_figsize(columns=4)
    """
    if columns == 1:
        fig_width_mm = 84.0
    elif columns == 1.5:
        fig_width_mm = 129.0
    elif columns == 2.0:
        fig_width_mm = 174.0
    else:
        raise ValueError

    if fig_height_mm is None:
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio.
        fig_height_mm = fig_width_mm * golden_mean

    max_figh_height_mm = 234.0
    if fig_height_mm > max_figh_height_mm:
        logging.warning(f"fig_height too large: {fig_height_mm}, so will reduce to {max_figh_height_mm}.")
        fig_height_mm = max_figh_height_mm

    ureg = UnitRegistry()
    fig_height_mm *= ureg.millimeter
    fig_width_mm *= ureg.millimeter
    fig_width = fig_width_mm.to(ureg.inch).magnitude
    fig_height = fig_height_mm.to(ureg.inch).magnitude

    logging.info(f"Creating figure of {fig_width_mm}x{fig_height_mm}.")
    return fig_width, fig_height


def regular_figure():
    """Sets some rc parameters for increased readability and creates an empty figure.

    Returns
    -------
    (figure, axis)
        The figure and its default axis.
    """

    params = {
        "axes.labelsize": 7,  # Fontsize for x and y labels (originally 10).
        "axes.titlesize": 7,
        "font.size": 7,  # Originally 10.
        "legend.fontsize": 7,  # Originally 10.
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "grid.linewidth": 0.8,
        "grid.linestyle": ":",
        "legend.frameon": True,
        "figure.dpi": 600,
    }
    matplotlib.rcParams.update(params)
    fig = plt.figure(figsize=a4_figsize(columns=2))
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False)
    return fig, ax


def save_fig(out_filename, tight="tight", formats=["png", "pdf"]):
    """Saves a figure to a file.

    Parameters
    ----------
    out_filename : str
        The filename of the figure.
    tight : str
        How the bounding box should be drawn.
    formats : list
        A list denoting in which formats this figure should be saved ('png' or 'pdf').

    Examples
    --------
    >>> ti.visualization.util("figure", formats=["png"])
    """

    if out_filename.endswith(".png"):
        outpath = out_filename
    else:
        outpath = out_filename + ".png"
    if "png" in formats:
        logging.info("Creating png...")
        ts = time.time()
        plt.savefig(outpath, dpi=600, bbox_inches=tight, pad_inches=0)
        logging.info(f"...took {round(time.time() - ts, 2)} s!")
    if "pdf" in formats:
        logging.info("Creating pdf...")
        ts = time.time()
        plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches=tight, pad_inches=0)
        logging.info(f"...took {round(time.time() - ts, 2)} s!")
    plt.close()
    logging.info("Finished!")


def plot_osm_streets(north, south, east, west, ax):
    """Plots with osmnx OpenStreetMap streets onto an axis.

    Parameters
    ----------
    north : float
        The northernmost coordinate (to retrieve OSM data for).

    south : float
        The southernmost coordinate.

    east : float
        The easternmost coordinate.

    west : float
        The westernmost coordinate.

    ax : matplotlib.pyplot.Artist, optional
        Axis on which to draw the plot.

    Examples
    --------
    >>> ti.visualization.plotting.plot_osm_street(47.392, 47.364, 8.557, 8.509, ax)
    """
    try:
        G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        lines = []
        for u, v, data in G.edges(keys=False, data=True):
            if "geometry" in data:
                xs, ys = data["geometry"].xy
                lines.append(list(zip(xs, ys)))
            else:
                x1 = G.nodes[u]["x"]
                y1 = G.nodes[u]["y"]
                x2 = G.nodes[v]["x"]
                y2 = G.nodes[v]["y"]
                line = [(x1, y1), (x2, y2)]
                lines.append(line)
        lc = LineCollection(lines, colors="#999999", linewidths=0.5, alpha=1, zorder=0)
        ax.add_collection(lc)
    except NetworkXPointlessConcept as e:
        logging.warn(f"Plotting of OSM graph failed: {e}")


def _prepare_frames(positionfixes, staypoints, triplegs, locations):
    """If not None transform GeoDataFrames to wgs84

    Parameters
    ----------
    positionfixes : Positionfixes
    staypoints : Staypoints
    triplegs : Triplegs
    locations : Locations

    Returns
    -------
    tuple of GeoDataFrames
        tuple with transformed (positionfixes, staypoints, triplegs, locations)
    """
    if positionfixes is not None:
        _, positionfixes = check_gdf_planar(positionfixes, transform=True)
    if staypoints is not None:
        _, staypoints = check_gdf_planar(staypoints, transform=True)
    if triplegs is not None:
        _, triplegs = check_gdf_planar(triplegs, transform=True)
    if locations is not None:
        _, locations = check_gdf_planar(locations, transform=True)
    return positionfixes, staypoints, triplegs, locations


def _calculate_bounds(positionfixes, staypoints, triplegs, locations):
    """Calculate bound of OSM size

    Parameters
    ----------
    positionfixes : Positionfixes
    staypoints : Staypoints
    triplegs : Triplegs
    locations : Locations

    Returns
    -------
    tuple of floats
        tuple with values for (north, south, east, west)
    """
    assert positionfixes is not None or staypoints is not None or triplegs is not None or locations is not None
    # TODO: maybe a relative value instead of 0.03
    if positionfixes is not None:
        north = positionfixes.geometry.y.max()
        south = positionfixes.geometry.y.min()
        east = positionfixes.geometry.x.max()
        west = positionfixes.geometry.x.min()
    elif staypoints is not None:
        north = staypoints.geometry.y.max() + 0.03
        south = staypoints.geometry.y.min() - 0.03
        east = staypoints.geometry.x.max() + 0.03
        west = staypoints.geometry.x.min() - 0.03
    elif triplegs is not None:
        triplegs_bounds = triplegs.bounds
        north = max(triplegs_bounds.maxy) + 0.03
        south = min(triplegs_bounds.miny) - 0.03
        east = max(triplegs_bounds.maxx) + 0.03
        west = min(triplegs_bounds.minx) - 0.03
    else:  # locations is not None
        north = locations.geometry.y.max() + 0.03
        south = locations.geometry.y.min() - 0.03
        east = locations.geometry.x.max() + 0.03
        west = locations.geometry.x.min() - 0.03
    return (north, south, east, west)


def _plot_frames(positionfixes, staypoints, triplegs, locations, radius_sp, radius_locs, ax):
    """Plot frames on axis

    Parameters
    ----------
    positionfixes : Positionfixes
    staypoints : Staypoints
    triplegs : Triplegs
    locations : Locations
    radius_sp : float
    radius_locs : float
    ax : matplotlib.pyplot.Artist
    """
    if positionfixes is not None:
        positionfixes.plot(ax=ax, markersize=0.5)
    if staypoints is not None:
        center_latitude = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        radius = meters_to_decimal_degrees(radius_sp, center_latitude)
        geometry = staypoints.geometry.name
        for pt in staypoints.to_dict("records"):
            circle = mpatches.Circle(
                (pt[geometry].x, pt[geometry].y), radius, facecolor="none", edgecolor="g", zorder=3
            )
            ax.add_artist(circle)
    if triplegs is not None:
        triplegs.plot(ax=ax, cmap="viridis")
    if locations is not None:
        center_latitude = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
        radius = meters_to_decimal_degrees(radius_locs, center_latitude)
        for pt in locations.to_dict("records"):
            circle = mpatches.Circle(
                (pt["center"].x, pt["center"].y), radius, facecolor="none", edgecolor="r", zorder=4
            )
            ax.add_artist(circle)


def plot(
    positionfixes=None,
    staypoints=None,
    triplegs=None,
    locations=None,
    radius_sp=100,
    radius_locs=150,
    filename=None,
    plot_osm=False,
    ax=None,
):
    """Plots positionfixes, staypoints, locations on a map (optionally to a file)

    One of the arguments [positionfixes, staypoints, triplegs, locations] should not be None!

    Parameters
    ----------
    positionfixes : Positionfixes, optional
        Positionfixes to plot, by default None
    staypoints : Staypoints, optional
        Staypoints to plot, by default None
    triplegs : Triplegs, optional
        Triplegs to plot, by default None
    locations : Locations, optional
        Locations to plot, by default None
    radius_sp : float, optional
        Radius in meter for circles around staypoints, default 100
    radius_locs : float, optional
        Radius in meter for circles around locations, default 150
    filename : str, optional
        The file to plot to, else if ax is none plot will be shown, by default None
    plot_osm : bool, optional
        If True, will download OSM street network and plot it as base map, by default False
        If True depending on the extent of your data, this might take a long time!
    ax : matplotlib.pyplot.Artist, optional
        axis on which to draw the plot, by default None

    Examples
    --------
    >>> ti.plot(positionfixes=pfs, filename="output.png", plot_osm=True)
    """
    has_no_ax_input = ax is None
    if ax is None:
        _, ax = regular_figure()
    if positionfixes is None and staypoints is None and triplegs is None and locations is None:
        raise ValueError("At least one GeoDataFrame should not be None.")

    positionfixes, staypoints, triplegs, locations = _prepare_frames(positionfixes, staypoints, triplegs, locations)
    if plot_osm:
        north, south, east, west = _calculate_bounds(positionfixes, staypoints, triplegs, locations)
        plot_osm_streets(north, south, east, west, ax=ax)
        ax.set_xlim([west, east])
        ax.set_ylim([south, north])
    _plot_frames(positionfixes, staypoints, triplegs, locations, radius_sp, radius_locs, ax)

    ax.set_aspect("equal", adjustable="box")
    if filename is not None:
        save_fig(filename, formats=["png"])
    elif has_no_ax_input:
        plt.show()


def plot_modal_split(
    df_modal_split_in,
    out_path=None,
    date_fmt_x_axis="%W",
    fig=None,
    axis=None,
    title=None,
    x_label=None,
    y_label=None,
    x_pad=10,
    y_pad=10,
    title_pad=1.02,
    skip_xticks=0,
    n_col_legend=5,
    borderaxespad=0.5,
    bar_kws=None,
):
    """
    Plot modal split as returned by `trackintel.analysis.calculate_modal_split`

    Parameters
    ----------
    df_modal_split : DataFrame
        DataFrame with modal split information. Format is
    out_path : str, optional
        Path to store the figure
    date_fmt_x_axis : str, default: '%W'
        strftime() date format code that is used for the x-axis
    title : str, optional
    x_label : str, optional
    y_label : str, optional
    fig : matplotlib.figure
        Only used if axis is provided as well.
    axis : matplotlib axes
    x_pad : float, default: 10
        Used to set ax.xaxis.labelpad
    y_pad : float, default: 10
        Used to set ax.yaxis.labelpad
    title_pad : float, default: 1.02
        Passed on to `matplotlib.pyplot.title`
    skip_xticks : int, default: 1
        Every nth x-tick label is kept.
    n_col_legend : int
        Passed on as `ncol` to matplotlib.pyplot.legend()
    borderaxespad : float
        The pad between the axes and legend border, in font-size units.
        Passed on to matplotlib.pyplot.legend()
    bar_kws : dict
        Parameters that control the bar-plot visualization, passed to DataFrame.plot.bar()

    Returns
    -------
    fig : Matplotlib figure handle
    ax : Matplotlib axis handle

    Examples
    --------
    >>> modal_split = calculate_modal_split(triplegs, metric='count', freq='D', per_user=False)
    >>> plot_modal_split(modal_split, out_path=tmp_file, date_fmt_x_axis='%d',
    >>>                  y_label='Percentage of daily count', x_label='days')
    """

    df_modal_split = df_modal_split_in.copy()
    if axis is None:
        fig, ax = regular_figure()
    else:
        ax = axis

    # make sure that modal split is only of a single user
    if isinstance(df_modal_split.index[0], tuple):
        raise ValueError(
            "This function can not support multiindex types. Use 'pandas.MultiIndex.droplevel' or pass "
            "the `per_user=False` flag in 'calculate_modal_split' function."
        )

    if not is_datetime64_any_dtype(df_modal_split.index.dtype):
        raise ValueError(
            "Index of modal split has to be a datetime type. This problem can be solved if the 'freq' "
            "keyword of 'calculate_modal_split is not None'"
        )
    # set date formatter
    df_modal_split.index = df_modal_split.index.map(lambda s: s.strftime(date_fmt_x_axis))

    # plotting
    df_modal_split.plot.bar(stacked=True, ax=ax, **(bar_kws or {}))

    # skip ticks for X axis
    if skip_xticks > 0:
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % skip_xticks != 0:
                tick.set_visible(False)

    # We use a nice trick to put the legend out of the plot and to scale it automatically
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        frameon=False,
        ncol=n_col_legend,
        borderaxespad=borderaxespad,
    )

    if title is not None:
        ax.set_title(title, y=title_pad)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if fig is not None:
        fig.autofmt_xdate()

    plt.tight_layout()

    ax.xaxis.labelpad = x_pad
    ax.yaxis.labelpad = y_pad

    if out_path is not None:
        save_fig(out_path)

    return fig, ax
