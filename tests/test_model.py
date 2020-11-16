import os

import pytest

import trackintel as ti


class TestModel:
    def test_as_positionfixes_accessor(self):
        orig_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        assert pfs.as_positionfixes

        pfs = pfs.drop(['geom'], axis=1)
        with pytest.raises(AttributeError):
            pfs.as_positionfixes

    def test_positionfixes_center(self):
        orig_file = os.path.join('tests', 'data', 'positionfixes.csv')
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        assert len(pfs.as_positionfixes.center) == 2

    def test_as_staypoints_accessor(self):
        orig_file = os.path.join('tests', 'data', 'staypoints.csv')
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        assert stps.as_staypoints

        stps = stps.drop(['geom'], axis=1)
        with pytest.raises(AttributeError):
            stps.as_staypoints

    def test_staypoints_center(self):
        orig_file = os.path.join('tests', 'data', 'staypoints.csv')
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        assert len(stps.as_staypoints.center) == 2
