import os

import numpy as np
import pandas as pd

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife


class TestReadGeolife:
    def test_loop_read(self):
        """use read_geolife reader, store posfix as .csv, load them again"""

        pfs, _ = read_geolife(os.path.join('tests', 'data', 'geolife'))
        tmp_file = os.path.join('tests', 'data', 'positionfixes_test.csv')
        pfs.as_positionfixes.to_csv(tmp_file)
        pfs2 = ti.read_positionfixes_csv(tmp_file, index_col='id')[pfs.columns]
        os.remove(tmp_file)
        assert np.isclose(0, (pfs.lat - pfs2.lat).abs().sum())

    def test_label_reading(self):
        """test data types of the labels returned by read_geolife"""

        pfs, labels = read_geolife(os.path.join('tests', 'data', 'geolife_modes'))
        # the output is a dictionary
        assert isinstance(labels, dict)

        # it has the keys of the users 10 and 20, the values are pandas dataframes
        for key, value in labels.items():
            assert key in [10, 20]
            assert isinstance(value, pd.DataFrame)

    def test_unavailble_label_reading(self):
        """test data types of the labels returned by read_geolife from a dictionary without label files"""

        pfs, labels = read_geolife(os.path.join('tests', 'data', 'geolife_long'))

        # the output is a dictionary
        assert isinstance(labels, dict)

        # the values are pandas dataframes
        for key, value in labels.items():
            assert isinstance(value, pd.DataFrame)
