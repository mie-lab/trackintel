import time
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from pint import UnitRegistry
from shapely.geometry import Point


ureg = UnitRegistry()


def a4_figsize(fig_height_mm=None, columns=2):
    """
    Generates sizes for a figure that fits on an A4 page.

    The sizes are taken from:
    http://www.springer.com/computer/journal/450 > Artwork and Illustrations Guidelines > Figure Placement and Size

    :param fig_height_mm: If set, uses this height for the figure. Otherwise computes one based on an aesthetic ratio.
    :param columns: The number of columns this figure should span (1, 1.5 or 2).
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
        logging.warning("fig_height too large: %s, so will reduce to %s." % (fig_height_mm, fig_height_mm))
        fig_height_mm = max_figh_height_mm

    fig_height_mm *= ureg.millimeter
    fig_width_mm *= ureg.millimeter

    fig_width = fig_width_mm.to(ureg.inch).magnitude
    fig_height = fig_height_mm.to(ureg.inch).magnitude

    logging.info('Creating figure of %sx%s.' % (fig_width_mm, fig_height_mm))
    return fig_width, fig_height


def regular_figure():
    """
    Sets some rc parameters for increased readability.
    """

    params = {
        'axes.labelsize': 7,  # Fontsize for x and y labels (originally 10).
        'axes.titlesize': 7,
        'font.size': 7,  # Originally 10.
        'legend.fontsize': 7,  # Originally 10.
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'grid.linewidth': 0.8,
        'grid.linestyle': ':',
        'legend.frameon': True,
        'figure.dpi': 600,
    }
    matplotlib.rcParams.update(params)
    fig = plt.figure(figsize=a4_figsize(columns=2))
    ax = fig.gca()
    return fig, ax


def save_fig(out_filename, tight='tight', formats=['png', 'pdf']):
    """
    Saves a figure to a file.

    :param out_filename: The filename of the figure.
    :param tight: How the bounding box should be drawn.
    :param formats: A list denoting in which formats this figure should be saved ('png' or 'pdf').
    """

    if out_filename.endswith('.png'):
        outpath = out_filename
    else:
        outpath = out_filename + '.png'
    if 'png' in formats:
        logging.info("Creating png...")
        ts = time.time()
        plt.savefig(outpath, dpi=600, bbox_inches=tight, pad_inches=0)
        logging.info("...took {} s!".format(round(time.time() - ts, 2)))
    if 'pdf' in formats:
        logging.info("Creating pdf...")
        ts = time.time()
        plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches=tight, pad_inches=0)
        logging.info("...took {} s!".format(round(time.time() - ts, 2)))
    plt.close()
    logging.info("Finished!")
