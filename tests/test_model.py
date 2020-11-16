import pytest
import sys
import os
import filecmp

import pandas as pd
import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints

class TestModel:
    def test_as_positionfixes_accessor(self):
        orig_file = 'tests/data/positionfixes.csv'
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        assert pfs.as_positionfixes

        pfs = pfs.drop(['geom'], axis=1)
        with pytest.raises(AttributeError):
            pfs.as_positionfixes

    def test_positionfixes_center(self):
        orig_file = 'tests/data/positionfixes.csv'
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        assert len(pfs.as_positionfixes.center) == 2

    def test_as_staypoints_accessor(self):        
        orig_file = 'tests/data/staypoints.csv'
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        assert stps.as_staypoints

        stps = stps.drop(['geom'], axis=1)
        with pytest.raises(AttributeError):
            stps.as_staypoints

    def test_staypoints_center(self):     
        orig_file = 'tests/data/staypoints.csv'
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        assert len(stps.as_staypoints.center) == 2

