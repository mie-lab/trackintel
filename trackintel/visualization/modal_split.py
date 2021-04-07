import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype

from trackintel.visualization.util import regular_figure, save_fig


def plot_modal_split(
    df_modal_split_in,
    out_path=None,
    date_fmt_x_axis="%W",
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
):
    """
    Plot modal split as returned by `trackintel.analysis.modal_split.calculate_modal_split`

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
    axis : matplotlib axes
    x_pad : float, default: 10
        used to set ax.xaxis.labelpad
    y_pad : float, default: 10
        used to set ax.yaxis.labelpad
    title_pad : float, default: 1.02
        passed on to `matplotlib.pyplot.title`
    skip_xticks : int, default: 0
        if larger than 0, every nth x-tick label is skipped.
    n_col_legend : int
        passed on as `ncol` to matplotlib.pyplot.legend()
    borderaxespad : float
        passed on to matplotlib.pyplot.legend()

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
    df_modal_split.plot.bar(stacked=True, ax=ax)

    # skip ticks for X axis
    if skip_xticks > 0:
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % skip_xticks == 0:
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
        plt.title(title, y=title_pad)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.autofmt_xdate()

    plt.tight_layout()

    ax.xaxis.labelpad = x_pad
    ax.yaxis.labelpad = y_pad

    if out_path is not None:
        save_fig(out_path)

    return fig, ax
