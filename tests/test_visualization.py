import pytest
import os
import matplotlib as mpl
mpl.use('Agg')

import trackintel as ti


class TestIO:
    def test_positionfixes_plot(self):
        tmp_file = os.path.join('tests', 'data', 'positionfixes_plot.png')
        
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')
        pfs.as_positionfixes.plot(out_filename=tmp_file, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_triplegs_plot(self):
        tmp_file = os.path.join('tests', 'data', 'triplegs_plot.png')
        
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')
        
        tpls_file = os.path.join('tests', 'data', 'triplegs.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';')
        tpls.as_triplegs.plot(out_filename=tmp_file, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)

    def test_staypoints_plot(self):
        tmp_file = os.path.join('tests', 'data', 'staypoints_plot.png')
        
        pfs_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(pfs_file, sep=';', index_col='id')
        
        stps_file = os.path.join('tests', 'data', 'staypoints.csv')
        stps = ti.read_staypoints_csv('tests/data/staypoints.csv', sep=';', index_col='id')
        stps.as_staypoints.plot(out_filename=tmp_file, radius=0.01, positionfixes=pfs, plot_osm=False)
        assert os.path.exists(tmp_file)
        os.remove(tmp_file)
