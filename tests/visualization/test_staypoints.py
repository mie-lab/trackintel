import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti


@pytest.fixture
def test_data():
    """Read tests data from files."""
    pfs_file = os.path.join("tests", "data", "positionfixes.csv")
    pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")

    stps_file = os.path.join("tests", "data", "staypoints.csv")
    stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id", crs="EPSG:4326")
    return pfs, stps


class TestPlot_staypoints:
    """Tests for plot_staypoints() method."""

    def test_staypoints_plot(self, test_data):
        """Use trackintel visualization function to plot staypoints and check if the file exists."""
        pfs, stps = test_data
        tmp_file = os.path.join("tests", "data", "staypoints_plot.png")

        stps.as_staypoints.plot(out_filename=tmp_file, radius=0.01, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_parameter(self, test_data):
        """Test other parameter configurations."""
        pfs, stps = test_data
        tmp_file = os.path.join("tests", "data", "staypoints_plot.png")

        # no radius
        stps.as_staypoints.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

        # with osm
        stps.as_staypoints.plot(out_filename="staypoints_plot", plot_osm=True)
        assert os.path.exists("staypoints_plot.png")
        os.remove("staypoints_plot.png")
