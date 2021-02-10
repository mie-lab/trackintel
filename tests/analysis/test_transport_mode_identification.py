# -*- coding: utf-8 -*-

import datetime
import os

import pytest
import numpy as np
import geopandas as gpd
from numpy.testing import assert_almost_equal

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife


class TestTransportModeIdentification:
    def test_check_empty_dataframe(self):
        """Assert that the method does not work for empty DataFrames (but
        that the rest works fine, e.g., method signature)."""
        tpls = ti.read_triplegs_csv('tests/data/triplegs_transport_mode_identification.csv', sep=';')
        empty_frame = tpls[0:0]
        with pytest.raises(AssertionError):
            empty_frame.as_triplegs.predict_transport_mode(method='simple-coarse')
   

    def test_check_correct_simple_coarse_identification(self):
        """Assert that the simple-coarse transport mode identification yields the correct results."""
        tpls = ti.read_triplegs_csv('tests/data/triplegs_transport_mode_identification.csv', sep=';')
        tpls_transport_mode = tpls.as_triplegs.predict_transport_mode(method='simple-coarse')

        assert tpls_transport_mode.iloc[0]['mode'] == 'slow_mobility'
        assert tpls_transport_mode.iloc[1]['mode'] == 'motorized_mobility'
        assert tpls_transport_mode.iloc[2]['mode'] == 'fast_mobility'
