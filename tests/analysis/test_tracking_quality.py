import sys, os, glob
import pytest
sys.path.append(os.path.join(os.getcwd(), 'trackintel'))
import trackintel as ti


@pytest.fixture
def testdata_stps_tpls_geolife_long():
    pfs = ti.io.dataset_reader.read_geolife(os.path.join('tests', 'data', 'geolife_long'))
    pfs, stps = pfs.as_positionfixes.generate_staypoints(method='sliding',
                                                         dist_threshold=25,
                                                         time_threshold=5 * 60)
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(stps, method='between_staypoints')
    
    tpls['type'] = 'tripleg'
    stps['type'] = 'staypoint'
    stps_tpls = stps.append(tpls).sort_values(by='started_at')
    return stps_tpls

class TestTemporal_tracking_quality():
    def test_temporal_all(self, testdata_stps_tpls_geolife_long):
        """ """
        
        stps_tpls = testdata_stps_tpls_geolife_long
        
        # calculate tracking quality for a sample user
        user_0 = stps_tpls.loc[stps_tpls['user_id'] == 0]
        extent = (user_0['finished_at'].max() - user_0['started_at'].min()).total_seconds()
        tracked = (user_0['finished_at'] - user_0['started_at']).dt.total_seconds().sum()
        quality_manual = tracked/extent
        
        # test if the result of the user agrees
        quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="all")
        print(quality)
        
        assert quality_manual == quality.loc[quality['user_id'] == 0, 'quality'].values[0]
        
    def test_temporal_split_overlaps_days(self, testdata_stps_tpls_geolife_long):
        """ """
        
        stps_tpls = testdata_stps_tpls_geolife_long
        
        # test if the result of the user agrees
        # quality = ti.analysis.tracking_quality.temporal_tracking_quality(stps_tpls, granularity="day")
        
        ti.analysis.tracking_quality._split_overlaps(stps_tpls, granularity="day")
        # assert quality_manual == quality.loc[0].values