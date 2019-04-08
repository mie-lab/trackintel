import pytest
import sys
import os
import filecmp

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints


class TestIO:
    def test_positionfixes_from_to_csv(self):
        tmp_file = 'tests/data/trajectory_test.csv'
        pfs = ti.read_positionfixes_csv('tests/data/trajectory.csv', sep=';')
        pfs['tracked_at'] = pfs['tracked_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        ti.write_positionfixes_csv(pfs, tmp_file, sep=';', 
            columns=['user_id', 'tracked_at', 'latitude', 'longitude', 'elevation', 'accuracy'])
        assert filecmp.cmp('tests/data/trajectory.csv', tmp_file)
        os.remove(tmp_file)
        
    def test_positionfixes_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass