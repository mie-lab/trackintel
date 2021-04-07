import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti
from trackintel.visualization.util import regular_figure


@pytest.fixture
def test_data():
    """Read tests data from files."""
    pfs_file = os.path.join("tests", "data", "positionfixes.csv")
    pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")
    stps_file = os.path.join("tests", "data", "staypoints.csv")
    stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id", crs="EPSG:4326")
    locs_file = os.path.join("tests", "data", "locations.csv")
    locs = ti.read_locations_csv(locs_file, sep=";", index_col="id", crs="EPSG:4326")
    return pfs, stps, locs


class TestPlot_locations:
    """Tests for plot_locations() method."""

    def test_locations_plot(self, test_data):
        """Use trackintel visualization function to plot locations and check if the file exists."""
        pfs, stps, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot.png")
        locs.as_locations.plot(
            out_filename=tmp_file, radius=120, positionfixes=pfs, staypoints=stps, staypoints_radius=100, plot_osm=False
        )
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_axis(self, test_data):
        """Test the use of regular_figure() to create axis."""
        pfs, _, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot.png")
        _, ax = regular_figure()

        locs.as_locations.plot(
            out_filename=tmp_file,
            radius=120,
            positionfixes=pfs,
            plot_osm=False,
            axis=ax,
        )
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_parameter(self, test_data):
        """Test other parameter configurations."""
        pfs, _, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot.png")

        # plot only location
        locs.as_locations.plot(out_filename=tmp_file, plot_osm=True)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
