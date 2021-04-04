import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti


class TestPlot_positionfixes:
    """Tests for plot_positionfixes() method."""

    def test_positionfixes_plot(self):
        """Use trackintel visualization function to plot positionfixes and check if the file exists."""
        tmp_file = os.path.join("tests", "data", "positionfixes_plot.png")
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")
        pfs.as_positionfixes.plot(out_filename=tmp_file, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
