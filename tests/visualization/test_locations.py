import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti
from trackintel.visualization.util import regular_figure


@pytest.fixture
def test_data():
    """Read tests data from files."""
    pfs_file = os.path.join("examples", "data", "geolife_trajectory.csv")
    pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col=None, crs="EPSG:4326")

    pfs, sp = pfs.as_positionfixes.generate_staypoints(method="sliding")
    pfs, _ = pfs.as_positionfixes.generate_triplegs(sp, method="between_staypoints")

    sp, locs = sp.as_staypoints.generate_locations(
        method="dbscan", distance_metric="haversine", epsilon=200, num_samples=1
    )
    return pfs, sp, locs


class TestPlot_locations:
    """Tests for plot_locations() method."""

    def test_locations_plot(self, test_data):
        """Use trackintel visualization function to plot locations and check if the file exists."""
        pfs, sp, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot1.png")
        locs.as_locations.plot(
            out_filename=tmp_file, radius=200, positionfixes=pfs, staypoints=sp, staypoints_radius=100, plot_osm=False
        )
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_axis(self, test_data):
        """Test the use of regular_figure() to create axis."""
        pfs, _, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot2.png")
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
        _, _, locs = test_data
        tmp_file = os.path.join("tests", "data", "locations_plot3.png")

        # plot only location
        locs.as_locations.plot(out_filename=tmp_file, plot_osm=True)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
