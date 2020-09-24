import pytest
import trackintel as ti
import os
import pandas as pd

class TestCreate_activity_flag():
    def test_create_activity_flag(self):
        spts_test = ti.read_staypoints_csv(os.path.join('tests', 'data', 'geolife', 'geolife_staypoints.csv'))

        activity_true = spts_test['activity'].copy()
        spts_test['activity'] = False

        spts_test = spts_test.as_staypoints.create_activity_flag()

        pd.testing.assert_series_equal(spts_test['activity'], activity_true)
