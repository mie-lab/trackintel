import copy

import pytest
import sys
import os
import filecmp

import trackintel as ti
from trackintel.preprocessing import positionfixes
from trackintel.preprocessing import staypoints
from trackintel.preprocessing.triplegs import smoothen_triplegs
import pandas as pd
import numpy as np
from matplotlib import pyplot

class TestIO:
    def test_positionfixes_from_to_csv(self):
        orig_file = 'tests/data/positionfixes.csv'
        tmp_file = 'tests/data/positionfixes_test.csv'
        pfs = ti.read_positionfixes_csv(orig_file, sep=';')
        pfs['tracked_at'] = pfs['tracked_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        pfs.as_positionfixes.to_csv(tmp_file, sep=';', 
            columns=['user_id', 'tracked_at', 'latitude', 'longitude', 'elevation', 'accuracy'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_positionfixes_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_triplegs_from_to_csv(self):
        orig_file = 'tests/data/triplegs.csv'
        tmp_file = 'tests/data/triplegs_test.csv'
        tpls = ti.read_triplegs_csv(orig_file, sep=';')
        tpls['started_at'] = tpls['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls['finished_at'] = tpls['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls.as_triplegs.to_csv(tmp_file, sep=';', 
            columns=['user_id', 'started_at', 'finished_at', 'geom'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)

    # def test_Douglas_Peucker_Algorithm_reduces_triplet_length(self):
    #     def plot_line(ax, ob):
    #         x, y = ob.xy
    #         ax.plot(x, y, alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    #
    #     pd.set_option('display.max_columns', 10)
    #     pd.set_option('display.max_rows', 10)
    #     orig_file = 'tests/data/triplegs_with_too_many_points_test.csv'
    #     tpls = ti.read_triplegs_csv(orig_file, sep=';')
    #     tpls_smoothed = smoothen_triplegs(tpls, epsilon=0.0001)
    #     line1 = tpls.iloc[0].geom
    #     line1_smoothed = tpls_smoothed.iloc[0].geom
    #     line2 = tpls.iloc[1].geom
    #     line2_smoothed = tpls_smoothed.iloc[1].geom
    #
    #     print(line1)
    #     print(line1_smoothed)
    #     assert line1.length == line1_smoothed.length
    #     assert line2.length == line2_smoothed.length
    #     assert len(line1.coords) == 10
    #     assert len(line2.coords) == 7
    #     assert len(line1_smoothed.coords) == 4
    #     assert len(line2_smoothed.coords) == 3

    # def test_test_Douglas_Peucker_Algorithm_has_no_side_effects(self):
    #     orig_file = 'tests/data/triplegs_with_too_many_points_test.csv'
    #     tpls = ti.read_triplegs_csv(orig_file, sep=';')
    #     tpls_copy = copy.deepcopy(tpls)
    #     tpls_smoothed = smoothen_triplegs(tpls, epsilon=0.0001)
    #
    #     assert np.all(tpls == tpls_copy)



    def test_triplegs_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_staypoints_from_to_csv(self):
        orig_file = 'tests/data/staypoints.csv'
        tmp_file = 'tests/data/staypoints_test.csv'
        stps = ti.read_staypoints_csv(orig_file, sep=';')
        stps['started_at'] = stps['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps['finished_at'] = stps['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        stps.as_staypoints.to_csv(tmp_file, sep=';', 
            columns=['user_id', 'started_at', 'finished_at', 'elevation', 'geom'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_staypoints_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_places_from_to_csv(self):
        orig_file = 'tests/data/places.csv'
        tmp_file = 'tests/data/places_test.csv'
        plcs = ti.read_places_csv(orig_file, sep=';')
        plcs.as_places.to_csv(tmp_file, sep=';', 
            columns=['user_id', 'elevation', 'center', 'extent'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_places_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

    def test_trips_from_to_csv(self):
        orig_file = 'tests/data/trips.csv'
        tmp_file = 'tests/data/trips_test.csv'
        tpls = ti.read_trips_csv(orig_file, sep=';')
        tpls['started_at'] = tpls['started_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls['finished_at'] = tpls['finished_at'].apply(lambda d: d.isoformat().replace('+00:00', 'Z'))
        tpls.as_trips.to_csv(tmp_file, sep=';', 
            columns=['user_id', 'started_at', 'finished_at', 'origin_staypoint_id', 'destination_staypoint_id'])
        assert filecmp.cmp(orig_file, tmp_file, shallow=False)
        os.remove(tmp_file)
        
    def test_trips_from_to_postgis(self):
        # TODO Implement some tests for PostGIS.
        pass

