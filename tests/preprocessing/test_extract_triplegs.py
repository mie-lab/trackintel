import pytest
import sys

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints
from numpy.testing import assert_almost_equal
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import trackintel as ti
from trackintel.geogr.distances import meters_to_decimal_degrees
from trackintel.io.dataset_reader import read_geolife


class TestExtractTriplegs:
    def test_extract_triplegs_global(self):
        # generate triplegs from raw-data
        pfs = read_geolife(os.path.join('tests','data','geolife_testdata'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        # load pregenerated test-triplegs
        tpls_test = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife_triplegs_test.csv'))

        assert len(tpls) > 0

        assert len(tpls) == len(tpls)
        
        tsamp = tpls.sample(1)
        assert list(tsamp.iloc[0]['geom'].coords) == list(pfs.loc[pfs['tripleg_id']==tsamp.iloc[0].id].geom.apply(lambda x: (x.x,x.y)))
           

        distance_sum = 0
        for i in range(len(tpls)):
            distance = tpls.geom.iloc[i].distance(tpls_test.geom.iloc[i])
            distance_sum = distance_sum + distance

        assert_almost_equal(distance_sum, 0.0)
   

