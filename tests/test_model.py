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

        pfs = pfs.drop(['elevation'], axis=1)
        with pytest.raises(AttributeError):
            pfs.as_positionfixes

    def test_positionfixes_center(self):
        orig_file = 'tests/data/positionfixes.csv'
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        assert len(pfs.as_positionfixes.center) == 2
