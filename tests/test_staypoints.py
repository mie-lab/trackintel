import trackintel as ti
import os
import pandas as pd
import os

import pandas as pd

import trackintel as ti


class TestCreate_activity_flag():
    def test_create_activity_flag(self):
        spts_file = os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv')
        spts_test = ti.read_staypoints_csv(spts_file, tz='utc', index_col='id')

        activity_true = spts_test['activity'].copy()
        spts_test['activity'] = False

        spts_test = spts_test.as_staypoints.create_activity_flag()

        pd.testing.assert_series_equal(spts_test['activity'], activity_true)
