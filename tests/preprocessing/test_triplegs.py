import os

import pandas as pd

from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips


class TestGenerate_trips():
    def test_general_generation(self):
        """
        Test if we can generate the example trips based on example data
        """
        # load pregenerated trips
        trips_loaded = pd.read_csv(os.path.join('tests', 'data', 'geolife_long', 'trips.csv'))
        trips_loaded['started_at'] = pd.to_datetime(trips_loaded['started_at'])
        trips_loaded['finished_at'] = pd.to_datetime(trips_loaded['finished_at'])

        # create trips from geolife (based on positionfixes)
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        spts = spts.as_staypoints.create_activity_flag()
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        spts, tpls, trips = generate_trips(spts, tpls, gap_threshold=15, id_offset=0)
        pd.testing.assert_frame_equal(trips_loaded, trips)
