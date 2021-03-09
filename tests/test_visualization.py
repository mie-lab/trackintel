import pytest
import os
import matplotlib as mpl

mpl.use('Agg')

import trackintel as ti


class TestIO:
    def test_positionfixes_plot(self):
        """Use trackintel visualization function to plot positionfixes and check if file exists."""

        tmp_file = os.path.join('tests', 'data', 'positionfixes_plot.png')
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')
        pfs.as_positionfixes.plot(out_filename=tmp_file, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_triplegs_plot(self):
        """Use trackintel visualization function to plot triplegs and check if file exists."""

        tmp_file = os.path.join('tests', 'data', 'triplegs_plot.png')
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')

        tpls_file = os.path.join('tests', 'data', 'triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col='id')
        tpls.as_triplegs.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_staypoints_plot(self):
        """Use trackintel visualization function to plot staypoints and check if file exists."""

        tmp_file = os.path.join('tests', 'data', 'staypoints_plot.png')
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')

        stps_file = os.path.join('tests', 'data', 'staypoints.csv')
        stps = ti.read_staypoints_csv(stps_file, sep=';', index_col='id')
        stps.as_staypoints.plot(out_filename=tmp_file, radius=0.01, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_locations_plot(self):
        """Use trackintel visualization function to plot locations and check if file exists."""

        tmp_file = os.path.join("tests", "data", "locations_plot.png")
        pfs = pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        stps = ti.read_staypoints_csv('tests/data/staypoints.csv', sep=';')
        locs = ti.read_locations_csv('tests/data/locations.csv', sep=';')
        locs.as_locations.plot(out_filename=tmp_file, radius=120, positionfixes=pfs,
                               staypoints=stps, staypoints_radius=100, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_positionfixes_crs_warning(self):
        """Check if warning is raised for data without crs."""

        file = os.path.join('tests', 'data', 'positionfixes.csv')
        _, ax = mpl.pyplot.subplots()
        pfs = ti.read_positionfixes_csv(file, sep=';')
        with pytest.warns(UserWarning):
            pfs.as_positionfixes.plot(axis=ax)

    def test_staypoints_crs_warning(self):
        """Check if warning is raised for data without crs."""

        file = os.path.join('tests', 'data', 'staypoints.csv')
        _, ax = mpl.pyplot.subplots()
        pfs = ti.read_staypoints_csv(file, sep=';')
        with pytest.warns(UserWarning):
            pfs.as_staypoints.plot(axis=ax)

    def test_triplegs_crs_warning(self):
        """Check if warning is raised for data without crs."""

        file = os.path.join('tests', 'data', 'triplegs.csv')
        _, ax = mpl.pyplot.subplots()
        pfs = ti.read_triplegs_csv(file, sep=';')
        with pytest.warns(UserWarning):
            pfs.as_triplegs.plot(axis=ax)

    def test_locations_crs_warning(self):
        """Check if warning is raised for data without crs."""

        file = os.path.join('tests', 'data', 'locations.csv')
        _, ax = mpl.pyplot.subplots()
        pfs = ti.read_locations_csv(file, sep=';')
        with pytest.warns(UserWarning):
            pfs.as_locations.plot(axis=ax)
