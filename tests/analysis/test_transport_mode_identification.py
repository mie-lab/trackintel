# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

import trackintel as ti


class TestTransportModeIdentification:
    def test_check_empty_dataframe(self):
        """Assert that the method does not work for empty DataFrames 
        (but that the rest works fine, e.g., method signature).
        """
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")
        empty_frame = tpls[0:0]
        with pytest.raises(AssertionError):
            empty_frame.as_triplegs.predict_transport_mode(method='simple-coarse')

    def test_simple_coarse_identification_no_crs(self):
        """Assert that the simple-coarse transport mode identification throws the correct 
        warning and and yields the correct results for WGS84.
        """
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")

        with pytest.warns(UserWarning, match='Your data is not projected. WGS84 is assumed and for length calculation'
                                             ' the haversine distance is used'):
            tpls_transport_mode = tpls.as_triplegs.predict_transport_mode(method='simple-coarse')

        assert tpls_transport_mode.iloc[0]['mode'] == 'slow_mobility'
        assert tpls_transport_mode.iloc[1]['mode'] == 'motorized_mobility'
        assert tpls_transport_mode.iloc[2]['mode'] == 'fast_mobility'

    def test_simple_coarse_identification_wgs_84(self):
        """Asserts the correct behaviour with data in wgs84."""
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")
        tpls_2 = tpls.set_crs(epsg=4326)
        tpls_transport_mode_2 = tpls_2.as_triplegs.predict_transport_mode(method='simple-coarse')

        assert tpls_transport_mode_2.iloc[0]['mode'] == 'slow_mobility'
        assert tpls_transport_mode_2.iloc[1]['mode'] == 'motorized_mobility'
        assert tpls_transport_mode_2.iloc[2]['mode'] == 'fast_mobility'

    def test_simple_coarse_identification_projected(self):
        """Asserts the correct behaviour with data in projected coordinate systems."""
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")
        tpls_2 = tpls.set_crs(epsg=4326)
        tpls_3 = tpls_2.to_crs(epsg=2056)
        tpls_transport_mode_3 = tpls_3.as_triplegs.predict_transport_mode(method='simple-coarse')
        assert tpls_transport_mode_3.iloc[0]['mode'] == 'slow_mobility'
        assert tpls_transport_mode_3.iloc[1]['mode'] == 'motorized_mobility'
        assert tpls_transport_mode_3.iloc[2]['mode'] == 'fast_mobility'

    def test_simple_coarse_identification_geographic(self):
        """Asserts that a warning is thrown if data in non-WGS geographic coordinate systems."""
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")
        tpls_2 = tpls.set_crs(epsg=4326)
        tpls_4 = tpls_2.to_crs(epsg=4269)
        with pytest.warns(UserWarning):
            tpls_4.as_triplegs.predict_transport_mode(method='simple-coarse')

    def test_check_categories(self):
        """Asserts the correct identification of valid category dictionaries."""
        tpls_file = os.path.join('tests', 'data', 'triplegs_transport_mode_identification.csv')
        tpls = ti.read_triplegs_csv(tpls_file, sep=';', index_col="id")
        correct_dict = {2: 'cat1', 7: 'cat2', np.inf: 'cat3'}

        assert ti.analysis.transport_mode_identification.check_categories(correct_dict)
        with pytest.raises(ValueError):
            incorrect_dict = {10: 'cat1', 5: 'cat2', np.inf: 'cat3'}
            tpls.as_triplegs.predict_transport_mode(method='simple-coarse', categories=incorrect_dict)
