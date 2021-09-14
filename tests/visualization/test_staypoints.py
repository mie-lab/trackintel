import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti


@pytest.fixture
def test_data():
    """Read tests data from files."""
    pfs_file = os.path.join("examples", "data", "geolife_trajectory.csv")
    pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col=None, crs="EPSG:4326")

    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding")
    return pfs, sp


class TestPlot_staypoints:
    """Tests for plot_staypoints() method."""

    def test_staypoints_plot(self, test_data):
        """Use trackintel visualization function to plot staypoints and check if the file exists."""
        pfs, sp = test_data
        tmp_file = os.path.join("tests", "data", "staypoints_plot1.png")

        sp.as_staypoints.plot(out_filename=tmp_file, radius=100, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_parameter(self, test_data):
        """Test other parameter configurations."""
        pfs, sp = test_data
        tmp_file = os.path.join("tests", "data", "staypoints_plot2.png")

        # no radius
        sp.as_staypoints.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

        # with osm
        sp.as_staypoints.plot(out_filename=tmp_file, plot_osm=True)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
