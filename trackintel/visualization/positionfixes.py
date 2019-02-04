import matplotlib.pyplot as plt

from trackintel.visualization.util import regular_figure, save_fig


def plot_positionfixes(positionfixes, out_filename):
    """
    Plots positionfixes to a file.

    :param positionfixes: The positionfixes to plot.
    :param out_filename: The file to plot to.
    """
    _, ax = regular_figure()
    positionfixes.plot(ax=ax)
    save_fig(out_filename, formats=['png'])
