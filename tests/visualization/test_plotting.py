import datetime
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import trackintel as ti
from trackintel.analysis.modal_split import calculate_modal_split
from trackintel.io.dataset_reader import geolife_add_modes_to_triplegs, read_geolife
from trackintel.visualization.plotting import _calculate_bounds, a4_figsize, plot, plot_modal_split, regular_figure

matplotlib.use("Agg")


@pytest.fixture
def geolife_triplegs_with_modes():
    """Get modal split for a small part of the geolife dataset."""
    pfs, labels = read_geolife(os.path.join("tests", "data", "geolife_modes"))
    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding", dist_threshold=25, time_threshold=5)
    _, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    tpls_with_modes = geolife_add_modes_to_triplegs(tpls, labels)
    return tpls_with_modes


@pytest.fixture
def triplegs_with_modes():
    """Get modal split for randomly generated data."""
    n = 200
    day_1_h1 = pd.Timestamp("1970-01-01 00:00:00", tz="utc")
    one_day = datetime.timedelta(days=1)
    mode_list = ["car", "walk", "train", "bus", "bike", "walk", "bike"]
    df = pd.DataFrame(index=np.arange(n))
    df["mode"] = np.random.choice(mode_list, n)
    df["user_id"] = np.random.randint(1, 5, size=n)
    df["started_at"] = np.random.randint(1, 30, size=n) * one_day
    df["started_at"] = df["started_at"] + day_1_h1
    return df


@pytest.fixture
def test_data():
    """Read tests data from files."""
    pfs_file = os.path.join("examples", "data", "geolife_trajectory.csv")
    pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col=None, crs="EPSG:4326")

    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding")
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", distance_metric="haversine", epsilon=200, num_samples=1
    )
    return pfs, sp, tpls, locs


class TestA4_figsize:
    """Tests for a4_figsize() method."""

    def test_parameter(self, caplog):
        """Test different parameter configurations."""
        fig_width, fig_height = a4_figsize(columns=1)
        assert np.allclose([3.30708661, 2.04389193], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=1.5)
        assert np.allclose([5.07874015, 3.13883403], [fig_width, fig_height])

        fig_width, fig_height = a4_figsize(columns=2)
        assert np.allclose([6.85039370, 4.23377614], [fig_width, fig_height])

        with pytest.raises(ValueError):
            a4_figsize(columns=3)

        a4_figsize(fig_height_mm=250)
        assert "fig_height too large" in caplog.text


class TestPlot_modal_split:
    def test_create_plot_geolife(self, geolife_triplegs_with_modes):
        """Check if we can run the plot function with geolife data without error"""
        modal_split = calculate_modal_split(geolife_triplegs_with_modes, freq="d", per_user=False)
        plot_modal_split(modal_split)

    def test_check_dtype_error(self, geolife_triplegs_with_modes):
        """Check if error is thrown correctly when index is not datetime

        freq=None calculates the modal split over the whole period
        """
        modal_split = calculate_modal_split(geolife_triplegs_with_modes, freq=None, per_user=False)
        with pytest.raises(ValueError):
            plot_modal_split(modal_split)

    def test_multi_user_error(self, triplegs_with_modes):
        """Create a modal split plot based on randomly generated test data"""
        modal_split = calculate_modal_split(triplegs_with_modes, freq="d", per_user=True, norm=True)
        with pytest.raises(ValueError):
            plot_modal_split(modal_split)

        # make sure that there is no error if the data was correctly created
        modal_split = calculate_modal_split(triplegs_with_modes, freq="d", per_user=False, norm=True)
        plot_modal_split(modal_split)

    def test_create_plot_testdata(self, triplegs_with_modes):
        """Create a modal split plot based on randomly generated test data"""
        tmp_file = os.path.join("tests", "data", "modal_split_plot.png")

        modal_split = calculate_modal_split(triplegs_with_modes, freq="d", per_user=False, norm=True)

        modal_split = modal_split[["walk", "bike", "train", "car", "bus"]]  # change order for the looks of the plot
        plot_modal_split(
            modal_split, out_path=tmp_file, date_fmt_x_axis="%d", y_label="Percentage of daily count", x_label="days"
        )

        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
        os.remove(tmp_file.replace("png", "pdf"))

    def test_ax_arg(self, triplegs_with_modes):
        """Test if ax is augmented if passed to function."""
        _, axis = regular_figure()
        modal_split = calculate_modal_split(triplegs_with_modes, freq="d", norm=True)
        xlabel, ylabel, title = "xlabel", "ylabel", "title"
        dateformat = "%d"
        _, ax = plot_modal_split(
            modal_split, date_fmt_x_axis=dateformat, x_label=xlabel, y_label=ylabel, title=title, axis=axis
        )
        assert axis is ax
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        assert ax.get_title() == title

    def test_skip_xticks(self, triplegs_with_modes):
        """Test if function set right ticks invisible."""
        modal_split = calculate_modal_split(triplegs_with_modes, freq="d", norm=True)
        mod = 4  # remove all but the mod 4 ticks
        _, ax = regular_figure()
        _, ax = plot_modal_split(modal_split)
        assert all(t.get_visible() for _, t in enumerate(ax.xaxis.get_major_ticks()))
        _, ax = regular_figure()
        _, ax = plot_modal_split(modal_split, skip_xticks=mod)
        assert all(t.get_visible() == (i % mod == 0) for i, t in enumerate(ax.xaxis.get_major_ticks()))


class Test_calculate_bounds:
    """Test helper function _calculate_bounds"""

    def test_all_None(self):
        """If all arguments are none then would be bug and should raise AssertionError"""
        with pytest.raises(AssertionError):
            _calculate_bounds(None, None, None, None)

    def test_locations(self, test_data):
        """Test reading of locations bounds"""
        _, _, _, locs = test_data
        n, s, e, w = _calculate_bounds(None, None, None, locs)
        assert n >= locs.geometry.y.max()
        assert s <= locs.geometry.y.min()
        assert e >= locs.geometry.x.max()
        assert w <= locs.geometry.x.min()

    def test_triplegs(self, test_data):
        """Test reading of triplegs bounds"""
        _, _, tpls, locs = test_data
        n, s, e, w = _calculate_bounds(None, None, tpls, locs)
        bounds = tpls.bounds
        assert n >= max(bounds.maxy)
        assert s <= min(bounds.miny)
        assert e >= max(bounds.maxx)
        assert w <= min(bounds.minx)

    def test_staypoints(self, test_data):
        """Test reading of staypoint bounds"""
        _, sp, tpls, locs = test_data
        n, s, e, w = _calculate_bounds(None, sp, tpls, locs)
        assert n >= sp.geometry.y.max()
        assert s <= sp.geometry.y.min()
        assert e >= sp.geometry.x.max()
        assert w <= sp.geometry.x.min()

    def test_positionfixes(self, test_data):
        """Test reading of positionfixes bounds"""
        pfs, sp, tpls, locs = test_data
        n, s, e, w = _calculate_bounds(pfs, sp, tpls, locs)
        assert n >= pfs.geometry.y.max()
        assert s <= pfs.geometry.y.min()
        assert e >= pfs.geometry.x.max()
        assert w <= pfs.geometry.x.min()


class TestPlot:
    """Test the plot function"""

    def test_ax(self, test_data):
        """Test if you can pass in the axis kwarg"""
        pfs, sp, tpls, locs = test_data
        _, ax = regular_figure()
        plot(positionfixes=pfs, staypoints=sp, triplegs=tpls, locations=locs, ax=ax)

    def test_all_None(self):
        """Test if Error is raised if all GeoDataFrames are None"""
        with pytest.raises(ValueError, match="At least one GeoDataFrame should not be None."):
            plot()

    def test_plot_file(self, test_data):
        """Test if plotting to file produces a file"""
        pfs, sp, tpls, locs = test_data
        tmp_file = os.path.join("tests", "data", "temp.png")
        plot(positionfixes=pfs, staypoints=sp, triplegs=tpls, locations=locs, filename=tmp_file)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_osm(self, test_data):
        """Test call to plot_osm"""
        pfs, sp, tpls, locs = test_data
        n, s, e, w = _calculate_bounds(*test_data)
        _, ax = regular_figure()
        plot(positionfixes=pfs, staypoints=sp, triplegs=tpls, locations=locs, plot_osm=True, ax=ax)
        assert ax.get_xlim() == (w, e)
        assert ax.get_ylim() == (s, n)

    def test_no_ax_no_file(self, test_data):
        """Test call without set axis nor output file then call plt.show()."""
        pfs, sp, tpls, locs = test_data
        # agg cannot show plt.show() but locally we get a warning for it (except in Linux)
        warnings.filterwarnings("ignore", "Matplotlib is currently using agg", category=UserWarning)
        plot(positionfixes=pfs, staypoints=sp, triplegs=tpls, locations=locs)
        warnings.resetwarnings()
