import pytest
import os
import matplotlib as mpl

mpl.use("Agg")

import trackintel as ti


class TestIO:
    def test_positionfixes_plot(self):
        """Use trackintel visualization function to plot positionfixes and check if file exists."""

        tmp_file = os.path.join("tests", "data", "positionfixes_plot.png")
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")
        pfs.as_positionfixes.plot(out_filename=tmp_file, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_triplegs_plot(self):
        """Use trackintel visualization function to plot triplegs and check if file exists."""

        tmp_file = os.path.join("tests", "data", "triplegs_plot.png")
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")

        tpls_file = os.path.join("tests", "data", "triplegs.csv")
        tpls = ti.read_triplegs_csv(tpls_file, sep=";", index_col="id", crs="EPSG:4326")
        tpls.as_triplegs.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_staypoints_plot(self):
        """Use trackintel visualization function to plot staypoints and check if file exists."""

        tmp_file = os.path.join("tests", "data", "staypoints_plot.png")
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")

        stps_file = os.path.join("tests", "data", "staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id", crs="EPSG:4326")
        stps.as_staypoints.plot(out_filename=tmp_file, radius=0.01, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_locations_plot(self):
        """Use trackintel visualization function to plot locations and check if file exists."""

        tmp_file = os.path.join("tests", "data", "locations_plot.png")
        pfs_file = os.path.join("tests", "data", "positionfixes.csv")
        pfs = pfs = ti.read_positionfixes_csv(pfs_file, sep=";", index_col="id", crs="EPSG:4326")
        stps_file = os.path.join("tests", "data", "staypoints.csv")
        stps = ti.read_staypoints_csv(stps_file, sep=";", index_col="id", crs="EPSG:4326")
        locs_file = os.path.join("tests", "data", "locations.csv")
        locs = ti.read_locations_csv(locs_file, sep=";", index_col="id", crs="EPSG:4326")
        locs.as_locations.plot(
            out_filename=tmp_file, radius=120, positionfixes=pfs, staypoints=stps, staypoints_radius=100, plot_osm=False
        )
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
