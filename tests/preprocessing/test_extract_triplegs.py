import datetime
import os

import numpy as np
from numpy.testing import assert_almost_equal

import trackintel as ti
from trackintel.io.dataset_reader import read_geolife


class TestExtractTriplegs:
    def test_extract_triplegs_global(self):
        # generate triplegs from raw-data
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        # load pregenerated test-triplegs
        tpls_test = ti.read_triplegs_csv(os.path.join('tests', 'data', 'geolife', 'geolife_triplegs_short.csv'))

        assert len(tpls) > 0

        assert len(tpls) == len(tpls)

        distance_sum = 0

        for i in range(len(tpls)):
            distance = tpls.geom.iloc[i].distance(tpls_test.geom.iloc[i])
            distance_sum = distance_sum + distance
        assert_almost_equal(distance_sum, 0.0)

    def test_check_overlap(self):
        """
        Triplegs and staypoints should not overlap when generated using the default extract triplegs method.
        This test extracts triplegs and staypoints from positionfixes and stores them in a single dataframe.
        The dataframe is sorted by date, then we check if the staypoint/tripleg from the row before was finished when
        the next one started.
        """
        pfs = read_geolife(os.path.join('tests', 'data', 'geolife_long'))
        spts = pfs.as_positionfixes.extract_staypoints(method='sliding', dist_threshold=25, time_threshold=5 * 60)
        tpls = pfs.as_positionfixes.extract_triplegs(spts)

        spts_tpls = spts[['started_at', 'finished_at', 'user_id']].append(
            tpls[['started_at', 'finished_at', 'user_id']])

        spts_tpls.sort_values(by=['user_id', 'started_at'], inplace=True)

        for user_id_this in spts['user_id'].unique():
            spts_tpls_this = spts_tpls[spts_tpls['user_id'] == user_id_this]
            diff = spts_tpls_this['started_at'] - spts_tpls_this['finished_at'].shift(1)

            # transform to numpy array and drop first values (always nan due to shift operation)
            diff = diff.values[1:]

            # all values have to greater or equal to zero. Otherwise there is an overlap
            assert all(diff >= np.timedelta64(datetime.timedelta()))
