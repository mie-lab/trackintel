import pytest
import os
import matplotlib as mpl
mpl.use('Agg')

import trackintel as ti


class TestIO:
    def test_positionfixes_plot(self):
        tmp_file = 'tests/data/positionfixes_plot.png'
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        pfs.as_positionfixes.plot(out_filename=tmp_file, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_triplegs_plot(self):
        tmp_file = 'tests/data/triplegs_plot.png'
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        tpls = ti.read_triplegs_csv('tests/data/triplegs.csv', sep=';')
        tpls.as_triplegs.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_staypoints_plot(self):
        tmp_file = 'tests/data/staypoints_plot.png'
        pfs = ti.read_positionfixes_csv('tests/data/positionfixes.csv', sep=';')
        stps = ti.read_staypoints_csv('tests/data/staypoints.csv', sep=';')
        stps.as_staypoints.plot(out_filename=tmp_file, radius=0.01, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
