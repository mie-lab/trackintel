import os

import numpy as np

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife


class TestReadGeolife:
    def test_loop_read(self):
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife'))
        tmp_file = os.path.join('tests', 'data', 'positionfixes_test.csv')
        pfs.as_positionfixes.to_csv(tmp_file)
        pfs2 = ti.read_positionfixes_csv(tmp_file, index_col='id')[pfs.columns]
        os.remove(tmp_file)
        assert np.isclose(0, (pfs.lat - pfs2.lat).abs().sum())
