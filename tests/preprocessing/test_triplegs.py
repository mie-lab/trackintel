import os

import pandas as pd
import trackintel as ti
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
from trackintel.preprocessing.triplegs import smoothen_triplegs


class TestGenerate_trips():
    def test_general_generation(self):
        """
        Test if we can generate the example trips based on example data
        """
        # load pregenerated trips
        trips_loaded = pd.read_csv(os.path.join('tests', 'data', 'geolife_long', 'trips.csv'))
        trips_loaded['started_at'] = pd.to_datetime(trips_loaded['started_at'])
        trips_loaded['finished_at'] = pd.to_datetime(trips_loaded['finished_at'])
        trips_loaded.rename(columns={'origin': 'origin_staypoint_id',
                                     'destination': 'destination_staypoint_id'}, inplace=True)

        # create trips from geolife (based on positionfixes)
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        spts = spts.as_staypoints.create_activity_flag()
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        spts, tpls, trips = generate_trips(spts, tpls, gap_threshold=15, id_offset=0)
        pd.testing.assert_frame_equal(trips_loaded, trips)

    def test_douglas_peucker_algorithm_reduce_tripleg_length(self):
        """
        test the douglas peucker algorithm in simplifying triplegs
        """
        orig_file = 'tests/data/triplegs_with_too_many_points_test.csv'
        tpls = ti.read_triplegs_csv(orig_file, sep=';')
        tpls_smoothed = smoothen_triplegs(tpls, tolerance=0.0001)
        line1 = tpls.iloc[0].geom
        line1_smoothed = tpls_smoothed.iloc[0].geom
        line2 = tpls.iloc[1].geom
        line2_smoothed = tpls_smoothed.iloc[1].geom

        print(line1)
        print(line1_smoothed)
        assert line1.length == line1_smoothed.length
        assert line2.length == line2_smoothed.length
        assert len(line1.coords) == 10
        assert len(line2.coords) == 7
        assert len(line1_smoothed.coords) == 4
        assert len(line2_smoothed.coords) == 3